"""Tests for the terminal client (open_harness/clients/terminal.py), written
against the interface contract before the implementation lands:

    class TerminalClient:
        def __init__(self, *, console=None, input_fn=None, show_thinking=False)
        def emit(self, record: dict) -> None
        def ask_human(self, req: ToolCallRequest, decision: Decision) -> str
        def ask_user(self, question: str, options: list[str]) -> str
        def interrupted(self) -> bool
        def on_text_delta(self, role: str, text: str) -> None
        show_thinking: bool
        def reset_interrupt(self) -> None
        def request_interrupt(self) -> None

    def run_terminal(config, cwd, log, *, client=None, session_factory=None) -> int
    SLASH_COMMANDS: dict[str, str]
    def handle_slash(cmd: str, session, client) -> bool

The module is imported with `pytest.importorskip` so this file collects and
(until the implementation lands) skips cleanly rather than erroring; see
`tests/test_session_controls.py` for the Session-level pieces (add_rule,
set_mode, usage_totals, resume) that are already implemented and run green
now.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import get_args

import pytest
from rich.console import Console

from open_harness.kernel.events import EventKind
from open_harness.kernel.serde import serialize_message
from open_harness.model.types import Block, Message, Stop, TextDelta
from open_harness.policy.types import Decision, Rule, ToolCallRequest
from tests.test_e2e_fake import make_session, tool_call

terminal = pytest.importorskip("open_harness.clients.terminal")
TerminalClient = terminal.TerminalClient
run_terminal = terminal.run_terminal
SLASH_COMMANDS = terminal.SLASH_COMMANDS
handle_slash = terminal.handle_slash

# --------------------------------------------------------------------------- helpers


def make_client(answers: list[str] | None = None, *, show_thinking: bool = False):
    """A TerminalClient wired to a StringIO console and a scripted input_fn
    that raises EOFError once the scripted answers run out (matching a real
    stdin hitting EOF)."""
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, width=100)
    it = iter(answers or [])

    def input_fn(prompt: str = "") -> str:
        try:
            return next(it)
        except StopIteration:
            raise EOFError from None

    client = TerminalClient(console=console, input_fn=input_fn, show_thinking=show_thinking)
    return client, buf


def raising_client(exc: type[BaseException]):
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, width=100)

    def input_fn(prompt: str = "") -> str:
        raise exc

    client = TerminalClient(console=console, input_fn=input_fn)
    return client, buf


def sample_request(command: str = "mkdir made_here") -> ToolCallRequest:
    return ToolCallRequest(tool_name="shell", args={"command": command}, permission_content=command,
                            is_read_only=False, is_destructive=True, paths=[])


def sample_decision(reason: str = "not a fixed repo command") -> Decision:
    return Decision(behavior="ask", reason=reason, step="reduce:3")


def assistant_message_with_all_blocks() -> Message:
    return Message(role="assistant", blocks=[
        Block.text_block("Here is the plan."),
        Block(type="thinking", text="Let me think about this carefully."),
        Block(type="tool_call", tool_call_id="c1", name="shell", args={"command": "mkdir made_here"}),
    ])


def tool_result_message(*, is_error: bool) -> Message:
    text = "boom: command failed" if is_error else "ok"
    return Message(role="user", blocks=[Block.tool_result("c1", text, is_error=is_error)])


def user_message() -> Message:
    return Message(role="user", blocks=[Block.text_block("hello")])


# --------------------------------------------------------------------------- ask_human


@pytest.mark.parametrize("answer,expected", [
    ("y", "allow"),
    ("yes", "allow"),
    ("", "allow"),
    ("a", "allow_always"),
    ("n", "deny"),
])
def test_ask_human_simple_answers(answer: str, expected: str) -> None:
    client, _ = make_client(answers=[answer])
    assert client.ask_human(sample_request(), sample_decision()) == expected


def test_ask_human_deny_with_message() -> None:
    client, _ = make_client(answers=["d", "not now"])
    assert client.ask_human(sample_request(), sample_decision()) == "deny:not now"


def test_ask_human_deny_with_empty_message_is_plain_deny() -> None:
    client, _ = make_client(answers=["d", ""])
    assert client.ask_human(sample_request(), sample_decision()) == "deny"


def test_ask_human_reprompts_on_unrecognised_input() -> None:
    client, _ = make_client(answers=["what", "bogus", "y"])
    assert client.ask_human(sample_request(), sample_decision()) == "allow"


def test_ask_human_keyboard_interrupt_denies() -> None:
    client, _ = raising_client(KeyboardInterrupt)
    assert client.ask_human(sample_request(), sample_decision()) == "deny"


def test_ask_human_eof_denies() -> None:
    client, _ = raising_client(EOFError)
    assert client.ask_human(sample_request(), sample_decision()) == "deny"


def test_ask_human_renders_tool_name_and_command() -> None:
    client, buf = make_client(answers=["y"])
    client.ask_human(sample_request("mkdir made_here"), sample_decision("approval needed"))
    out = buf.getvalue()
    assert "shell" in out
    assert "mkdir made_here" in out


# --------------------------------------------------------------------------- ask_user


def test_ask_user_numeric_selection_picks_second_option() -> None:
    client, buf = make_client(answers=["2"])
    assert client.ask_user("Pick one", ["alpha", "beta", "gamma"]) == "beta"
    out = buf.getvalue()
    assert "alpha" in out
    assert "beta" in out
    assert "gamma" in out


def test_ask_user_empty_answer_selects_first_option() -> None:
    client, _ = make_client(answers=[""])
    assert client.ask_user("Pick one", ["alpha", "beta"]) == "alpha"


def test_ask_user_free_text_is_returned_verbatim() -> None:
    client, _ = make_client(answers=["something else entirely"])
    assert client.ask_user("Pick one", ["alpha", "beta"]) == "something else entirely"


def test_ask_user_eof_returns_empty_string() -> None:
    client, _ = raising_client(EOFError)
    assert client.ask_user("Pick one", ["alpha", "beta"]) == ""


# --------------------------------------------------------------------------- interrupt state


def test_interrupt_lifecycle() -> None:
    client, _ = make_client()
    assert client.interrupted() is False
    client.request_interrupt()
    assert client.interrupted() is True
    client.reset_interrupt()
    assert client.interrupted() is False


# --------------------------------------------------------------------------- on_text_delta


def test_on_text_delta_main_writes_to_console() -> None:
    client, buf = make_client()
    client.on_text_delta("main", "abc")
    assert "abc" in buf.getvalue()


def test_on_text_delta_compactor_writes_nothing() -> None:
    client, buf = make_client()
    client.on_text_delta("compactor", "abc")
    assert buf.getvalue() == ""


# --------------------------------------------------------------------------- show_thinking attribute


def test_show_thinking_defaults_false() -> None:
    client, _ = make_client()
    assert client.show_thinking is False


def test_show_thinking_can_be_enabled_at_construction() -> None:
    client, _ = make_client(show_thinking=True)
    assert client.show_thinking is True


# --------------------------------------------------------------------------- emit: every event kind must not raise

_EXTRA_KINDS = {"user_message", "tool_result", "rule_added", "mode_changed"}
ALL_EVENT_KINDS = sorted(set(get_args(EventKind)) | _EXTRA_KINDS)


def _payload_for(kind: str) -> dict:
    base = {"seq": 1, "ts": "2026-01-01T00:00:00+00:00", "kind": kind}
    by_kind = {
        "session_start": {"cwd": "/tmp/proj", "mode": "default", "resumed": False},
        "turn_start": {"turn": 1},
        "turn_end": {"reason": "completed", "detail": "", "turns": 3},
        "transition": {"reason": "next_turn"},
        "model_resolved": {"role": "main", "spec": "fake:m"},
        "text_delta": {"role": "main", "text": "partial"},
        "thinking_delta": {"role": "main", "text": "thinking..."},
        "assistant_message": {"message": serialize_message(assistant_message_with_all_blocks()),
                              "stop": "tool_use", "model": "fake:m"},
        "user_message": {"message": serialize_message(user_message())},
        "tool_result": {"message": serialize_message(tool_result_message(is_error=False))},
        "usage": {"role": "main", "spec": "fake:m", "input": 5, "output": 2, "cache_read": 0, "cache_write": 0},
        "provider_error": {"kind": "server", "message": "boom", "retryable": False},
        "tool_call_start": {"id": "c1", "tool": "shell", "args": {"command": "mkdir made_here"}},
        "tool_call_end": {"id": "c1", "tool": "shell", "is_error": False, "chars": 10},
        "tool_result_persisted": {"id": "c1", "path": "sessions/s1/c1.txt"},
        "permission_decision": {"tool": "shell", "behavior": "deny", "step": "reduce:3",
                                "reason": "denied by user", "immune": False},
        "approval_request": {"tool": "shell", "content": "mkdir made_here", "reason": "needs approval"},
        "approval_response": {"tool": "shell", "behavior": "allow"},
        "rule_added": {"rule": "shell(mkdir *)", "behavior": "allow"},
        "mode_changed": {"mode": "plan"},
        "verifier_result": {"stage": "gate", "ok": False, "detail": "line one\nline two"},
        "compact_boundary": {"reason": "proactive", "summarised": 3, "kept": 2},
        "attachment": {"path": "notes.txt", "media_type": "text/plain"},
        "task_notification": {"text": "background task finished", "task_id": "t1"},
        "error": {"message": "something broke"},
        "info": {"message": "fyi"},
    }
    return {**base, **by_kind.get(kind, {})}


@pytest.mark.parametrize("kind", ALL_EVENT_KINDS)
def test_emit_does_not_raise(kind: str) -> None:
    client, _ = make_client()
    client.emit(_payload_for(kind))


def test_emit_tool_result_error_renders_content() -> None:
    client, buf = make_client()
    payload = {"seq": 1, "ts": "2026-01-01T00:00:00+00:00", "kind": "tool_result",
               "message": serialize_message(tool_result_message(is_error=True))}
    client.emit(payload)
    assert "boom: command failed" in buf.getvalue()


def test_emit_tool_result_ok_does_not_raise() -> None:
    client, _ = make_client()
    payload = {"seq": 1, "ts": "2026-01-01T00:00:00+00:00", "kind": "tool_result",
               "message": serialize_message(tool_result_message(is_error=False))}
    client.emit(payload)


def test_emit_permission_decision_deny_renders_reason() -> None:
    client, buf = make_client()
    client.emit({"seq": 1, "ts": "2026-01-01T00:00:00+00:00", "kind": "permission_decision",
                 "tool": "shell", "behavior": "deny", "step": "reduce:3",
                 "reason": "denied by user: not now", "immune": False})
    assert "denied by user: not now" in buf.getvalue()


def test_emit_verifier_result_failure_renders_multiline_detail() -> None:
    client, buf = make_client()
    client.emit({"seq": 1, "ts": "2026-01-01T00:00:00+00:00", "kind": "verifier_result",
                 "stage": "gate", "ok": False, "detail": "FAILED test_x\nAssertionError: boom"})
    out = buf.getvalue()
    assert "FAILED test_x" in out
    assert "AssertionError: boom" in out


@pytest.mark.parametrize("reason", ["next_turn", "verifier_failed"])
def test_emit_transition_does_not_raise(reason: str) -> None:
    client, _ = make_client()
    client.emit({"seq": 1, "ts": "2026-01-01T00:00:00+00:00", "kind": "transition", "reason": reason})


def test_emit_usage_then_turn_end_renders_reason_and_token_totals() -> None:
    client, buf = make_client()
    client.emit({"seq": 1, "ts": "2026-01-01T00:00:00+00:00", "kind": "usage",
                 "role": "main", "spec": "fake:m", "input": 100, "output": 40,
                 "cache_read": 0, "cache_write": 0})
    client.emit({"seq": 2, "ts": "2026-01-01T00:00:00+00:00", "kind": "turn_end",
                 "reason": "completed", "detail": "", "turns": 2})
    out = buf.getvalue()
    assert "completed" in out
    assert "100" in out
    assert "40" in out


def test_emit_assistant_message_renders_tool_call_name_and_command() -> None:
    client, buf = make_client()
    client.emit({"seq": 1, "ts": "2026-01-01T00:00:00+00:00", "kind": "assistant_message",
                 "message": serialize_message(assistant_message_with_all_blocks()),
                 "stop": "tool_use", "model": "fake:m"})
    out = buf.getvalue()
    assert "shell" in out
    assert "mkdir made_here" in out


def test_emit_assistant_message_hides_thinking_by_default() -> None:
    client, buf = make_client(show_thinking=False)
    client.emit({"seq": 1, "ts": "2026-01-01T00:00:00+00:00", "kind": "assistant_message",
                 "message": serialize_message(assistant_message_with_all_blocks()),
                 "stop": "tool_use", "model": "fake:m"})
    out = buf.getvalue()
    assert "Here is the plan." in out
    assert "Let me think about this carefully." not in out


def test_emit_assistant_message_shows_thinking_when_enabled() -> None:
    client, buf = make_client(show_thinking=True)
    client.emit({"seq": 1, "ts": "2026-01-01T00:00:00+00:00", "kind": "assistant_message",
                 "message": serialize_message(assistant_message_with_all_blocks()),
                 "stop": "tool_use", "model": "fake:m"})
    assert "Let me think about this carefully." in buf.getvalue()


# --------------------------------------------------------------------------- handle_slash


class FakeSession:
    def __init__(self) -> None:
        self.mode = "default"
        self.mode_calls: list[str] = []
        self.rules: list[Rule] = [Rule(tool="shell", content="mkdir *", behavior="allow", source="session")]
        self.rule_calls: list[tuple[str, str]] = []

    def set_mode(self, mode: str) -> None:
        self.mode_calls.append(mode)
        self.mode = mode

    def usage_totals(self) -> dict[str, int]:
        return {"input": 123, "output": 45, "cache_read": 0, "cache_write": 0}

    def add_rule(self, text: str, behavior: str = "allow") -> Rule:
        self.rule_calls.append((text, behavior))
        rule = Rule(tool=text, content=None, behavior=behavior, source="session")
        self.rules.append(rule)
        return rule


def test_handle_slash_mode_valid_calls_set_mode() -> None:
    session = FakeSession()
    client, _ = make_client()
    handle_slash("/mode plan", session, client)
    assert session.mode_calls == ["plan"]


def test_handle_slash_mode_invalid_does_not_call_set_mode_and_prints_help() -> None:
    session = FakeSession()
    client, buf = make_client()
    handle_slash("/mode bogus", session, client)
    assert session.mode_calls == []
    assert buf.getvalue().strip() != ""


def test_handle_slash_rules_lists_tool_and_content() -> None:
    session = FakeSession()
    client, buf = make_client()
    handle_slash("/rules", session, client)
    out = buf.getvalue()
    assert "shell" in out
    assert "mkdir *" in out


def test_handle_slash_cost_prints_usage_totals() -> None:
    session = FakeSession()
    client, buf = make_client()
    handle_slash("/cost", session, client)
    out = buf.getvalue()
    assert "123" in out
    assert "45" in out


def test_handle_slash_thinking_on_sets_client_flag() -> None:
    session = FakeSession()
    client, _ = make_client(show_thinking=False)
    handle_slash("/thinking on", session, client)
    assert client.show_thinking is True


def test_handle_slash_quit_returns_true() -> None:
    session = FakeSession()
    client, _ = make_client()
    assert handle_slash("/quit", session, client) is True


def test_handle_slash_unknown_command_returns_false() -> None:
    session = FakeSession()
    client, _ = make_client()
    assert handle_slash("/nonsense", session, client) is False


def test_handle_slash_help_lists_every_slash_command() -> None:
    assert SLASH_COMMANDS
    session = FakeSession()
    client, buf = make_client()
    handle_slash("/help", session, client)
    out = buf.getvalue()
    for key in SLASH_COMMANDS:
        assert key in out, f"{key!r} missing from /help output"


# --------------------------------------------------------------------------- run_terminal (end to end)


def test_run_terminal_approval_allow_creates_directory(tmp_path: Path) -> None:
    scripts = [
        tool_call(0, "c1", "shell", {"command": "mkdir made_here"}) + [Stop("tool_use")],
        [TextDelta("made it"), Stop("end_turn")],
    ]
    answers = iter(["make a directory", "y", "/quit"])

    def input_fn(prompt: str = "") -> str:
        try:
            return next(answers)
        except StopIteration:
            raise EOFError from None

    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, width=100)
    client = TerminalClient(console=console, input_fn=input_fn)
    session, _adapter = make_session(tmp_path, scripts, client)

    rc = run_terminal(session.config, session.cwd, session.log, client=client,
                      session_factory=lambda config, cwd, client, log: session)

    assert rc == 0
    assert (session.cwd / "made_here").is_dir()
    out = buf.getvalue()
    assert "Approval needed" in out
    assert "mkdir made_here" in out
    # log records must be rendered too, not only the approval panel
    assert "▸ shell" in out, "tool call line from the log listener is missing"
    assert "completed" in out, "turn_end line from the log listener is missing"
    recs = list(session.log.read())
    approval_responses = [r for r in recs if r["kind"] == "approval_response"]
    assert approval_responses, recs
    assert approval_responses[-1]["behavior"] == "allow"


def test_run_terminal_approval_allow_always_adds_session_rule(tmp_path: Path) -> None:
    scripts = [
        tool_call(0, "c1", "shell", {"command": "mkdir made_here"}) + [Stop("tool_use")],
        [TextDelta("made it"), Stop("end_turn")],
    ]
    answers = iter(["make a directory", "a", "/quit"])

    def input_fn(prompt: str = "") -> str:
        try:
            return next(answers)
        except StopIteration:
            raise EOFError from None

    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, width=100)
    client = TerminalClient(console=console, input_fn=input_fn)
    session, _adapter = make_session(tmp_path, scripts, client)

    rc = run_terminal(session.config, session.cwd, session.log, client=client,
                      session_factory=lambda config, cwd, client, log: session)

    assert rc == 0
    recs = list(session.log.read())
    assert any(r["kind"] == "rule_added" for r in recs), recs
    session_rules = [r for r in session.rules if r.source == "session"]
    assert len(session_rules) == 1
    assert session_rules[0].tool == "shell"


def test_run_terminal_deny_with_message_reaches_tool_result(tmp_path: Path) -> None:
    scripts = [
        tool_call(0, "c1", "shell", {"command": "mkdir made_here"}) + [Stop("tool_use")],
        [TextDelta("okay, skipping that"), Stop("end_turn")],
    ]
    answers = iter(["make a directory", "d", "not now", "/quit"])

    def input_fn(prompt: str = "") -> str:
        try:
            return next(answers)
        except StopIteration:
            raise EOFError from None

    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, width=100)
    client = TerminalClient(console=console, input_fn=input_fn)
    session, _adapter = make_session(tmp_path, scripts, client)

    rc = run_terminal(session.config, session.cwd, session.log, client=client,
                      session_factory=lambda config, cwd, client, log: session)

    assert rc == 0
    assert not (session.cwd / "made_here").exists()
    recs = list(session.log.read())
    tool_result_recs = [r for r in recs if r["kind"] == "tool_result"]
    assert any("not now" in json.dumps(r.get("message")) for r in tool_result_recs), tool_result_recs
