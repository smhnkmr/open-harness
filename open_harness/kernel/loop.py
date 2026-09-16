"""The loop. A while-true state machine with named transitions.

Spec: open-harness-spec.md section 4.

Session.run_turn(text) runs one user turn to a Terminal. Every transition,
model resolution, permission decision, tool call and verifier result is
written to the event log. Termination requires two signatures: the model
stops calling tools AND the verifier gate passes.
"""

from __future__ import annotations

import platform
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from open_harness.backend.local import LocalBackend
from open_harness.config import Config
from open_harness.context import compact as compaction
from open_harness.context.prompt import build_system, instructions_message, load_instructions
from open_harness.kernel.gateway import ContextTooLong, Gateway, GatewayError
from open_harness.kernel.log import EventLog
from open_harness.kernel.roles import RoleResolver
from open_harness.kernel.serde import serialize_message
from open_harness.model.types import Block, Message, ModelResponse, ThinkingConfig
from open_harness.policy.engine import decide
from open_harness.policy.reducer import reduce_ask
from open_harness.policy.rules import parse_rule
from open_harness.policy.types import PolicyContext, Rule, ToolCallRequest
from open_harness.tools import default_registry
from open_harness.tools.base import Tool, ToolContext, ToolRegistry, ToolResult
from open_harness.tools.results import persist_if_large
from open_harness.verify.gate import run_gate, run_lint

MAX_TOOL_CONCURRENCY = 10
MAX_OUTPUT_RECOVERY = 3          # [expires] added_for: prototype 2026-09
ESCALATED_MAX_TOKENS = 32_000
DEFAULT_WINDOW = 200_000
MAX_VERIFIER_RETRIES = 5


class Client(Protocol):
    """What the loop needs from a client. The stdio client implements this."""

    def emit(self, record: dict) -> None: ...
    def ask_human(self, req: ToolCallRequest, decision: Any) -> str: ...
    def ask_user(self, question: str, options: list[str]) -> str: ...
    def interrupted(self) -> bool: ...


@dataclass
class Terminal:
    reason: str
    detail: str = ""


@dataclass
class Continue:
    reason: str


@dataclass
class State:
    messages: list[Message]
    turn: int = 0
    max_output_override: int | None = None
    output_recovery_count: int = 0
    reactive_compact_attempted: bool = False
    verifier_retries: int = 0
    compactions: int = 0
    touched_files: list[str] = field(default_factory=list)


class Session:
    def __init__(self, config: Config, cwd: Path, client: Client, log: EventLog,
                 *, registry: ToolRegistry | None = None, gateway: Gateway | None = None) -> None:
        self.config = config
        self.cwd = cwd.resolve()
        self.client = client
        self.log = log
        self.backend = LocalBackend(root=self.cwd)
        self.registry = registry or default_registry()
        self.gateway = gateway or Gateway(resolver=RoleResolver(config), on_event=self._record)
        self.resolver = self.gateway.resolver
        self.read_state: dict[str, Any] = {}
        self.rules: list[Rule] = self._load_rules()
        self.mode = config.policy.mode
        self.messages: list[Message] = log.replay_messages()
        self.log.append("session_start", cwd=str(self.cwd), mode=self.mode, resumed=bool(self.messages))

    # ------------------------------------------------------------------ public

    def run_turn(self, text: str) -> Terminal:
        """Run one user turn. Returns the terminal reason."""
        if not self.messages:
            instr = load_instructions(self.cwd)
            if instr:
                self._push(instructions_message(instr))
        self._push(Message(role="user", blocks=[Block.text_block(text)]))
        state = State(messages=self.messages)
        terminal = self._loop(state)
        self.log.append("turn_end", reason=terminal.reason, detail=terminal.detail, turns=state.turn)
        return terminal

    def final_text(self) -> str:
        for m in reversed(self.messages):
            if m.role == "assistant" and m.text.strip():
                return m.text
        return ""

    # ------------------------------------------------------------------ the loop

    def _loop(self, state: State) -> Terminal:
        while True:
            if self.client.interrupted():
                return Terminal("aborted_streaming", "interrupted before model call")
            if state.turn >= self.config.max_turns:
                return Terminal("max_turns")
            state.turn += 1
            self.log.append("turn_start", turn=state.turn)

            system, tools = self._build_request_parts()
            est = compaction.estimate_tokens(state.messages, system, sum(len(str(t.json_schema)) for t in tools))
            if compaction.should_compact(est, DEFAULT_WINDOW, self.config.max_output_tokens):
                self._compact(state, reason="proactive")

            try:
                resp = self.gateway.call(
                    "main", system=system, messages=state.messages, tools=tools,
                    max_output_tokens=state.max_output_override or self.config.max_output_tokens,
                    thinking=ThinkingConfig(enabled=True),
                )
            except ContextTooLong:
                if state.reactive_compact_attempted:
                    return Terminal("prompt_too_long")
                state.reactive_compact_attempted = True
                self._compact(state, reason="reactive")
                self._transition(Continue("reactive_compact_retry"))
                continue
            except GatewayError as e:
                return Terminal("model_error", str(e))

            self._push(resp.message, state)
            self.log.append("assistant_message", message=serialize_message(resp.message),
                            stop=resp.stop.reason, model=resp.model)
            for bad in resp.invalid_tool_calls:
                self._push(Message(role="user", blocks=[Block.tool_result(
                    bad.id or f"invalid-{bad.index}", f"Invalid tool call arguments: {bad.error}", is_error=True)]), state)

            # max_tokens recovery
            if resp.stop.reason == "max_tokens" and not resp.message.tool_calls:
                if state.max_output_override is None:
                    state.max_output_override = ESCALATED_MAX_TOKENS
                    self._transition(Continue("max_output_tokens_escalate"))
                    continue
                if state.output_recovery_count < MAX_OUTPUT_RECOVERY:
                    state.output_recovery_count += 1
                    self._push(Message(role="user", blocks=[Block.text_block(
                        "Your previous response was cut off. Continue from where you stopped.")],
                        meta={"harness": True}), state)
                    self._transition(Continue("max_output_tokens_recovery"))
                    continue

            calls = resp.message.tool_calls
            if not calls and resp.invalid_tool_calls:
                # the model tried to call a tool but the arguments were malformed;
                # the error result is already pushed, give it another turn.
                self._transition(Continue("next_turn"))
                continue
            if not calls:
                gate = self._verifier_gate(state)
                if gate is None:
                    return Terminal("completed")
                if state.verifier_retries >= MAX_VERIFIER_RETRIES:
                    return Terminal("completed", "verifier still failing after retries; reporting as-is")
                state.verifier_retries += 1
                self._push(Message(role="user", blocks=[Block.text_block(gate)], meta={"harness": True}), state)
                self._transition(Continue("verifier_failed"))
                continue

            results, aborted = self._run_tool_calls(calls, state)
            for r in results:
                self._push(r, state)
            if aborted:
                return Terminal("aborted_tools")
            state.max_output_override = None
            state.output_recovery_count = 0
            self._transition(Continue("next_turn"))

    # ------------------------------------------------------------------ tools

    def _run_tool_calls(self, calls: list[Block], state: State) -> tuple[list[Message], bool]:
        """Batch: consecutive concurrency-safe calls run in parallel, others alone."""
        ctx = self._tool_context()
        batches: list[list[Block]] = []
        for call in calls:
            tool = self.registry.get(call.name or "")
            safe = bool(tool and tool.is_concurrency_safe(call.args or {}))
            if safe and batches and batches[-1] and self._batch_is_safe(batches[-1]):
                batches[-1].append(call)
            else:
                batches.append([call])
        out: list[Message] = []
        for batch in batches:
            if self.client.interrupted():
                out.extend(self._abort_results(batch))
                return out, True
            if len(batch) == 1:
                out.append(self._execute_one(batch[0], ctx, state))
            else:
                with ThreadPoolExecutor(max_workers=min(MAX_TOOL_CONCURRENCY, len(batch))) as pool:
                    out.extend(pool.map(lambda c: self._execute_one(c, ctx, state), batch))
        return out, False

    def _batch_is_safe(self, batch: list[Block]) -> bool:
        return all((t := self.registry.get(c.name or "")) and t.is_concurrency_safe(c.args or {}) for c in batch)

    def _execute_one(self, call: Block, ctx: ToolContext, state: State) -> Message:
        name, args, cid = call.name or "", call.args or {}, call.tool_call_id or ""
        self.log.append("tool_call_start", id=cid, tool=name, args=args)
        tool = self.registry.get(name)
        if tool is None:
            return self._tool_error(cid, f"Unknown tool: {name}")
        v = tool.validate(args, ctx)
        if not v.ok:
            return self._tool_error(cid, f"Invalid input: {v.message}")
        decision = self._permission(tool, args)
        if decision.behavior != "allow":
            self.log.append("tool_call_end", id=cid, tool=name, denied=True, reason=decision.reason)
            return self._tool_error(cid, f"Permission denied: {decision.reason}")
        try:
            result = tool.call(args, ctx)
        except Exception as e:  # noqa: BLE001 - tool failures are results, not crashes
            result = ToolResult(content=f"Tool error: {type(e).__name__}: {e}", is_error=True)
        result = persist_if_large(result, tool, cid, self.log.path.parent)
        if result.persisted_path:
            self.log.append("tool_result_persisted", id=cid, path=result.persisted_path)
        if name in ("edit", "write") and not result.is_error:
            self._after_edit(args.get("file_path", ""), result, state)
        self.log.append("tool_call_end", id=cid, tool=name, is_error=result.is_error,
                        chars=len(result.content) if isinstance(result.content, str) else -1)
        block = Block.tool_result(cid, result.content, is_error=result.is_error)
        return Message(role="user", blocks=[block], meta={"tool": name})

    def _after_edit(self, path: str, result: ToolResult, state: State) -> None:
        """Verification stage 1: lint after every edit; append only errors."""
        if path and path not in state.touched_files:
            state.touched_files.append(path)
        if not self.config.verify.lint:
            return
        errors = run_lint(self.backend, path, self.config.verify, self.cwd)
        self.log.append("verifier_result", stage="lint", file=path, ok=not errors)
        if errors and isinstance(result.content, str):
            result.content += "\n\n" + errors

    def _verifier_gate(self, state: State) -> str | None:
        """Verification stage 2: tests before done. Returns failure text or None."""
        if not state.touched_files:
            return None
        gate = run_gate(self.backend, self.config.verify, self.cwd)
        self.log.append("verifier_result", stage="gate", ok=gate.ok, detail=gate.failures[:2000])
        if gate.ok:
            return None
        return ("<harness-context type=\"verifier\">The turn cannot end: the test command failed. "
                f"Fix the failures below, then stop calling tools.\n{gate.failures}</harness-context>")

    def _permission(self, tool: Tool, args: dict[str, Any]):
        req = ToolCallRequest(tool_name=tool.name, args=args, permission_content=tool.permission_content(args),
                              is_read_only=tool.is_read_only(args), is_destructive=tool.is_destructive(args),
                              paths=self._paths_of(tool, args))
        pctx = PolicyContext(mode=self.mode, cwd=str(self.cwd), additional_dirs=self.config.policy.additional_dirs,
                             rules=self.rules, bypass_available=False)
        decision = decide(req, pctx)
        if decision.behavior == "ask":
            decision = reduce_ask(req, decision, pctx, classifier=None, human=self._human)
            if decision.suggested_rule and decision.behavior == "allow":
                self.rules.append(parse_rule(decision.suggested_rule, "allow", "session"))
        self.log.append("permission_decision", tool=tool.name, behavior=decision.behavior,
                        step=decision.step, reason=decision.reason, immune=decision.immune)
        return decision

    def _human(self, req: ToolCallRequest, decision: Any) -> str:
        self.log.append("approval_request", tool=req.tool_name, content=req.permission_content, reason=decision.reason)
        answer = self.client.ask_human(req, decision)
        self.log.append("approval_response", tool=req.tool_name, behavior=answer)
        return answer

    @staticmethod
    def _paths_of(tool: Tool, args: dict[str, Any]) -> list[str]:
        p = args.get("file_path") or args.get("path")
        return [str(p)] if p else []

    def _tool_context(self) -> ToolContext:
        return ToolContext(cwd=self.cwd, session_dir=self.log.path.parent, backend=self.backend,
                           read_state=self.read_state, is_main_thread=True,
                           abort=self.client.interrupted, ask_user=self.client.ask_user)

    def _tool_error(self, cid: str, text: str) -> Message:
        return Message(role="user", blocks=[Block.tool_result(cid, text, is_error=True)], meta={"error": True})

    def _abort_results(self, batch: list[Block]) -> list[Message]:
        return [self._tool_error(c.tool_call_id or "", "Interrupted by user") for c in batch]

    # ------------------------------------------------------------------ context

    def _build_request_parts(self):
        tools_prompt = "\n\n".join(f"## {t.name}\n{t.prompt()}" for t in self.registry.tools.values())
        main = self.resolver.resolve("main")
        env = {"cwd": str(self.cwd), "platform": platform.platform(), "shell": shutil.which("bash") or "powershell",
               "date": datetime.now(UTC).date().isoformat(), "model": main.spec}
        system = build_system(self.cwd, tools_prompt=tools_prompt, profile_suffix=None, env=env)
        return system, self.registry.specs()

    def _compact(self, state: State, *, reason: str) -> None:
        system, _ = self._build_request_parts()
        to_summarise, keep = compaction.build_compact_request(state.messages, system)
        if not to_summarise:
            return
        prompt_msg = Message(role="user", blocks=[Block.text_block(compaction.SUMMARY_PROMPT)], meta={"harness": True})
        resp: ModelResponse = self.gateway.call("compactor", system=system, messages=to_summarise + [prompt_msg],
                                                tools=[], max_output_tokens=20_000)
        new_messages = compaction.apply_summary(resp.message.text, keep, state.touched_files[-5:])
        self.log.append("compact_boundary", reason=reason, summarised=len(to_summarise), kept=len(keep))
        state.messages[:] = new_messages
        self.messages = state.messages
        for m in new_messages:
            self._record_message(m)
        state.compactions += 1

    # ------------------------------------------------------------------ plumbing

    def _push(self, m: Message, state: State | None = None) -> None:
        target = state.messages if state is not None else self.messages
        target.append(m)
        if state is not None and target is not self.messages:
            self.messages = target
        self._record_message(m)

    def _record_message(self, m: Message) -> None:
        kind = "assistant_message" if m.role == "assistant" else ("tool_result" if m.blocks and m.blocks[0].type == "tool_result" else "user_message")
        if kind != "assistant_message":   # assistant messages are recorded with stop/model by the loop
            self.log.append(kind, message=serialize_message(m))

    def _transition(self, c: Continue) -> None:
        self.log.append("transition", reason=c.reason)

    def _record(self, kind: str, payload: dict) -> None:
        self.log.append(kind, **payload)

    def _load_rules(self) -> list[Rule]:
        out: list[Rule] = []
        for text in self.config.policy.deny:
            out.append(parse_rule(text, "deny", "user"))
        for text in self.config.policy.ask:
            out.append(parse_rule(text, "ask", "user"))
        for text in self.config.policy.allow:
            out.append(parse_rule(text, "allow", "user"))
        return out
