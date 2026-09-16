"""Interactive terminal client: a rich REPL over `kernel.loop.Session`.

Spec: open-harness-spec.md section 12.2 ("Terminal UI ... thin").

`TerminalClient` implements the `kernel.loop.Client` protocol (`emit`,
`ask_human`, `ask_user`, `interrupted`, plus the optional `on_text_delta`
live-streaming hook) by rendering event-log records with `rich`. `run_terminal`
drives the REPL: read a line, dispatch slash commands, otherwise run one turn.
"""

from __future__ import annotations

import json
import signal
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from open_harness.policy.types import Decision, ToolCallRequest

PATH_TOOLS = ("read", "write", "edit", "glob", "grep")
PATTERN_TOOLS = ("glob", "grep")


class TerminalClient:
    """Renders the event log live and drives approvals from the terminal."""

    def __init__(self, *, console: Console | None = None,
                 input_fn: Callable[[str], str] | None = None,
                 show_thinking: bool = False) -> None:
        self.console = console or Console(highlight=False, soft_wrap=True)
        self.input_fn = input_fn or input
        self.show_thinking = show_thinking
        self._streamed = ""
        self._interrupted = False
        self._turn_usage = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}

    # ------------------------------------------------------------------ Client protocol

    def emit(self, record: dict) -> None:
        try:
            self._emit(record)
        except Exception:  # noqa: BLE001 - rendering must never crash the loop
            return

    def ask_human(self, req: ToolCallRequest, decision: Decision) -> str:
        self._render_approval(req, decision)
        while True:
            try:
                raw = self.input_fn("[y] allow once  [a] allow always  [n] deny  [d] deny with message > ")
            except (KeyboardInterrupt, EOFError):
                return "deny"
            answer = raw.strip().lower()
            if answer in ("y", "yes", ""):
                return "allow"
            if answer == "a":
                rule = decision.suggested_rule or req.permission_content
                self._line(f"rule added for this session: {rule}; "
                           "add to open-harness.toml [policy].allow to persist")
                return "allow_always"
            if answer == "n":
                return "deny"
            if answer == "d":
                try:
                    msg = self.input_fn("message to the model > ")
                except (KeyboardInterrupt, EOFError):
                    return "deny"
                msg = msg.strip()
                return f"deny:{msg}" if msg else "deny"
            self._line("unrecognised; enter y, a, n, or d")

    def ask_user(self, question: str, options: list[str]) -> str:
        self.console.print(question, markup=False)
        for i, opt in enumerate(options, 1):
            self.console.print(f"  {i}. {opt}", markup=False)
        try:
            raw = self.input_fn("> ")
        except (KeyboardInterrupt, EOFError):
            return ""
        raw = raw.strip()
        if not raw:
            return options[0] if options else ""
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(options):
                return options[idx - 1]
        return raw

    def interrupted(self) -> bool:
        return self._interrupted

    def on_text_delta(self, role: str, text: str) -> None:
        if role != "main":
            return
        self.console.print(text, end="", markup=False)
        self._streamed += text

    # ------------------------------------------------------------------ state

    def reset_interrupt(self) -> None:
        self._interrupted = False

    def request_interrupt(self) -> None:
        self._interrupted = True

    # ------------------------------------------------------------------ rendering helpers

    def _line(self, text: str, *, style: str | None = "dim") -> None:
        self.console.print(text, style=style, markup=False)

    def _render_approval(self, req: ToolCallRequest, decision: Decision) -> None:
        lines = [
            Text(f"tool: {req.tool_name}"),
            Text(""),
            Text(req.permission_content, style="bold cyan"),
            Text(""),
            Text(f"reason: {decision.reason}"),
            Text(f"step: {decision.step}"),
        ]
        if decision.suggested_rule:
            lines.append(Text(f"suggested rule: {decision.suggested_rule}"))
        body = Text("\n").join(lines)
        self.console.print(Panel(body, title="Approval needed"))

    # ------------------------------------------------------------------ emit dispatch

    def _emit(self, record: dict[str, Any]) -> None:
        kind = record.get("kind")
        handler = _HANDLERS.get(kind)
        if handler is not None:
            handler(self, record)

    def _on_session_start(self, record: dict[str, Any]) -> None:
        self._line(f"session started in {record.get('cwd')} (mode={record.get('mode')})")

    def _on_assistant_message(self, record: dict[str, Any]) -> None:
        for block in record.get("message", {}).get("blocks", []):
            btype = block.get("type")
            if btype == "thinking":
                if self.show_thinking:
                    text = " ".join((block.get("text") or "").split())
                    self._line(text[:120])
            elif btype == "text":
                if self._streamed:
                    self.console.print()
                    self._streamed = ""
                else:
                    text = block.get("text") or ""
                    if text.strip():
                        self.console.print(Markdown(text))
            elif btype == "tool_call":
                name = block.get("name") or ""
                args = block.get("args") or {}
                self.console.print(f"▸ {name} {_compact_args(name, args)}", markup=False)

    def _on_user_message(self, record: dict[str, Any]) -> None:
        message = record.get("message", {})
        if not (message.get("meta") or {}).get("harness"):
            return
        blocks = message.get("blocks", [])
        text = blocks[0].get("text", "") if blocks else ""
        if 'type="instructions"' in text:
            self._line("[instructions loaded]")
        else:
            self._line("[context]")

    def _on_tool_result(self, record: dict[str, Any]) -> None:
        blocks = record.get("message", {}).get("blocks", [])
        if not blocks:
            return
        block = blocks[0]
        text = block.get("text")
        if text is None:
            text = "".join(b.get("text", "") for b in block.get("content") or [] if b.get("type") == "text")
        text = (text or "").strip()
        if block.get("is_error"):
            self._line(f"  ✗ {text[:200]}", style="red")
        else:
            first = text.splitlines()[0] if text else ""
            summary = first if first else f"{len(text)} chars"
            self._line(f"  ✓ {summary}")

    def _on_permission_decision(self, record: dict[str, Any]) -> None:
        behavior = record.get("behavior")
        step = str(record.get("step") or "")
        if behavior == "deny":
            self._line(f"  ⛔ {record.get('tool')} denied: {record.get('reason')}", style="red")
        elif step.startswith("reduce:3"):
            self._line("  ✓ approved")

    def _on_verifier_result(self, record: dict[str, Any]) -> None:
        stage = record.get("stage", "")
        if record.get("ok"):
            label = "lint ok" if stage == "lint" else "tests passed"
            self._line(f"  ⚙ {label}")
        else:
            detail = str(record.get("detail") or "")
            head = "\n".join(detail.splitlines()[:3])
            text = f"  ⚙ {stage} FAILED"
            if head:
                text += f"\n{head}"
            self._line(text, style="yellow")

    def _on_transition(self, record: dict[str, Any]) -> None:
        reason = record.get("reason")
        if reason != "next_turn":
            self._line(f"  ↻ {reason}")

    def _on_compact_boundary(self, record: dict[str, Any]) -> None:
        self._line(f"  ⟲ compacted: {record.get('summarised')} messages summarised, "
                   f"{record.get('kept')} kept")

    def _on_provider_error(self, record: dict[str, Any]) -> None:
        message = str(record.get("message", ""))[:120]
        self._line(f"  ! {record.get('kind')}: {message} (attempt {record.get('attempt')})", style="yellow")

    def _on_usage(self, record: dict[str, Any]) -> None:
        for key in self._turn_usage:
            self._turn_usage[key] += int(record.get(key, 0) or 0)

    def _on_turn_end(self, record: dict[str, Any]) -> None:
        u = self._turn_usage
        self._line(f"── {record.get('reason')} · {record.get('turns')} turns · "
                   f"in {u['input']} out {u['output']} cache {u['cache_read']}")
        self._turn_usage = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}

    def _on_info(self, record: dict[str, Any]) -> None:
        self._line(str(record.get("msg", "")))


_HANDLERS: dict[str, Callable[[TerminalClient, dict[str, Any]], None]] = {
    "session_start": TerminalClient._on_session_start,
    "assistant_message": TerminalClient._on_assistant_message,
    "user_message": TerminalClient._on_user_message,
    "tool_result": TerminalClient._on_tool_result,
    "permission_decision": TerminalClient._on_permission_decision,
    "verifier_result": TerminalClient._on_verifier_result,
    "transition": TerminalClient._on_transition,
    "compact_boundary": TerminalClient._on_compact_boundary,
    "provider_error": TerminalClient._on_provider_error,
    "usage": TerminalClient._on_usage,
    "turn_end": TerminalClient._on_turn_end,
    "info": TerminalClient._on_info,
}


def _compact_args(name: str, args: dict[str, Any]) -> str:
    if name == "shell":
        return str(args.get("command", ""))
    if name in PATH_TOOLS:
        path = str(args.get("file_path") or args.get("path") or "")
        if name in PATTERN_TOOLS:
            pattern = str(args.get("pattern", ""))
            return f"{path} {pattern}".strip()
        return path
    return json.dumps(args, ensure_ascii=False)[:100]


# --------------------------------------------------------------------------- REPL


SLASH_COMMANDS: dict[str, str] = {
    "/help": "show this help",
    "/mode": "/mode <default|accept_edits|plan|bypass|dont_ask> - change the policy mode",
    "/rules": "list active policy rules",
    "/cost": "show cumulative token usage for this session",
    "/thinking": "/thinking on|off - toggle showing model thinking blocks",
    "/quit": "exit the session",
}

_VALID_MODES = ("default", "accept_edits", "plan", "bypass", "dont_ask")


def handle_slash(cmd: str, session: Any, client: TerminalClient) -> bool:
    """Handle a line starting with '/'. Returns True if the REPL should exit."""
    parts = cmd.strip().split(maxsplit=1)
    name = parts[0]
    rest = parts[1].strip() if len(parts) > 1 else ""

    if name == "/quit":
        return True
    if name == "/help":
        _print_help(client)
        return False
    if name == "/mode":
        return _handle_mode(rest, session, client)
    if name == "/rules":
        return _handle_rules(session, client)
    if name == "/cost":
        return _handle_cost(session, client)
    if name == "/thinking":
        return _handle_thinking(rest, client)

    _print_help(client)
    return False


def _handle_mode(rest: str, session: Any, client: TerminalClient) -> bool:
    if rest not in _VALID_MODES:
        client._line(f"usage: {SLASH_COMMANDS['/mode']}")
        return False
    session.set_mode(rest)
    client._line(f"mode set to {rest}")
    return False


def _handle_rules(session: Any, client: TerminalClient) -> bool:
    if not session.rules:
        client._line("(no rules)")
        return False
    for rule in session.rules:
        tool = f"{rule.tool}({rule.content})" if rule.content is not None else rule.tool
        client._line(f"{rule.behavior} {tool} [{rule.source}]")
    return False


def _handle_cost(session: Any, client: TerminalClient) -> bool:
    totals = session.usage_totals()
    client._line(f"input {totals['input']} · output {totals['output']} · "
                 f"cache_read {totals['cache_read']} · cache_write {totals['cache_write']}")
    return False


def _handle_thinking(rest: str, client: TerminalClient) -> bool:
    arg = rest.lower()
    if arg not in ("on", "off"):
        client._line(f"usage: {SLASH_COMMANDS['/thinking']}")
        return False
    client.show_thinking = arg == "on"
    client._line(f"thinking display {'on' if client.show_thinking else 'off'}")
    return False


def _print_help(client: TerminalClient) -> None:
    for name, help_text in SLASH_COMMANDS.items():
        client._line(f"{name:<10} {help_text}")


def run_terminal(config: Any, cwd: Path, log: Any, *, client: TerminalClient | None = None,
                 session_factory: Callable[..., Any] | None = None) -> int:
    """Interactive REPL. Reads lines with `client.input_fn`, handles slash
    commands, and runs `Session.run_turn` for everything else. Returns an
    exit code. Exits on /quit, EOF, or a second Ctrl-C while idle."""
    client = client or TerminalClient()
    # Render every log record as it is written, so the terminal shows exactly
    # what the transcript holds (tool calls, results, verifier, turn end).
    if client.emit not in log.listeners:
        log.listeners.append(client.emit)
    factory = session_factory
    if factory is None:
        from open_harness.kernel.loop import Session as factory
    session = factory(config, cwd, client, log)

    try:
        model_spec = session.resolver.resolve("main").spec
    except Exception:  # noqa: BLE001 - header must never block startup
        model_spec = "unknown"
    client._line(f"session {log.session_id}  ·  log: {log.path}  ·  "
                 f"model: {model_spec}  ·  mode: {session.mode}")
    client._line("type /help")

    state = {"turn_running": False, "idle_ctrl_c": 0}

    def _on_sigint(signum: int, frame: Any) -> None:
        if state["turn_running"]:
            client.request_interrupt()
            return
        state["idle_ctrl_c"] += 1
        if state["idle_ctrl_c"] >= 2:
            raise KeyboardInterrupt

    prev_handler = signal.signal(signal.SIGINT, _on_sigint)
    try:
        _repl(session, client, state)
    finally:
        signal.signal(signal.SIGINT, prev_handler)
    return 0


def _repl(session: Any, client: TerminalClient, state: dict[str, Any]) -> None:
    while True:
        state["idle_ctrl_c"] = 0
        try:
            line = client.input_fn("› ")
        except (EOFError, KeyboardInterrupt):
            return
        line = line.strip()
        if not line:
            continue
        if line.startswith("/"):
            if handle_slash(line, session, client):
                return
            continue
        client.reset_interrupt()
        state["turn_running"] = True
        try:
            session.run_turn(line)
        except KeyboardInterrupt:
            client.request_interrupt()
        except Exception as exc:  # noqa: BLE001 - keep the REPL alive on turn errors
            client._line(f"{type(exc).__name__}: {exc}", style="red")
        finally:
            state["turn_running"] = False
