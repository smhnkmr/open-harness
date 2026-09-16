"""Tests for `open_harness.clients.stdio.StdioClient`."""

from __future__ import annotations

import io
import json

from open_harness.clients.stdio import StdioClient
from open_harness.policy.types import Decision, ToolCallRequest


def _lines(*ops: dict) -> io.StringIO:
    return io.StringIO("\n".join(json.dumps(op) for op in ops) + "\n")


def test_emit_writes_one_json_line_per_call() -> None:
    out = io.StringIO()
    client = StdioClient(stdin=io.StringIO(""), stdout=out, background=False)
    client.emit({"kind": "text_delta", "text": "hi"})
    client.emit({"kind": "text_delta", "text": "there"})
    lines = out.getvalue().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == {"kind": "text_delta", "text": "hi"}
    assert json.loads(lines[1]) == {"kind": "text_delta", "text": "there"}


def test_quiet_client_suppresses_emit() -> None:
    out = io.StringIO()
    client = StdioClient(stdin=io.StringIO(""), stdout=out, background=False, quiet=True)
    client.emit({"kind": "text_delta", "text": "hidden"})
    assert out.getvalue() == ""


def test_next_op_returns_none_on_eof() -> None:
    client = StdioClient(stdin=io.StringIO(""), stdout=io.StringIO())
    assert client.next_op() is None


def test_ask_human_resolves_and_queues_unrelated_ops() -> None:
    scripted = _lines(
        {"op": "interrupt"},
        {"op": "approve", "request_id": "req-1", "behavior": "allow"},
        {"op": "turn_input", "text": "next thing"},
    )
    out = io.StringIO()
    client = StdioClient(stdin=scripted, stdout=out)

    req = ToolCallRequest(
        tool_name="shell", args={"command": "rm -rf build"}, permission_content="rm -rf build",
        is_read_only=False, is_destructive=True,
    )
    decision = Decision(behavior="ask", reason="destructive command", suggested_rule="shell(rm -rf*)")

    answer = client.ask_human(req, decision, request_id="req-1")
    assert answer == "allow"

    emitted = [json.loads(line) for line in out.getvalue().splitlines() if line.strip()]
    assert len(emitted) == 1
    assert emitted[0]["kind"] == "approval_request"
    assert emitted[0]["tool"] == "shell"
    assert emitted[0]["content"] == "rm -rf build"
    assert emitted[0]["reason"] == "destructive command"
    assert emitted[0]["suggested_rule"] == "shell(rm -rf*)"
    request_id = emitted[0]["request_id"]
    assert isinstance(request_id, str) and request_id

    # the interrupt op that arrived first set the flag but was not queued;
    # unrelated ops (turn_input) queue in arrival order for later next_op().
    assert client.interrupted() is True

    queued = client.next_op()
    assert queued == {"op": "turn_input", "text": "next thing"}


def test_ask_human_returns_deny_on_eof_without_matching_approve() -> None:
    scripted = _lines({"op": "turn_input", "text": "irrelevant"})
    client = StdioClient(stdin=scripted, stdout=io.StringIO())

    req = ToolCallRequest(
        tool_name="write", args={}, permission_content="write a.py",
        is_read_only=False, is_destructive=False,
    )
    decision = Decision(behavior="ask", reason="new file")

    answer = client.ask_human(req, decision)
    assert answer == "deny"


def test_ask_user_emits_user_question_and_resolves() -> None:
    scripted = _lines({"op": "approve", "request_id": "q-1", "behavior": "yes"})
    out = io.StringIO()
    client = StdioClient(stdin=scripted, stdout=out)

    answer = client.ask_user("Proceed with deploy?", ["yes", "no"], request_id="q-1")
    assert answer == "yes"

    emitted = json.loads(out.getvalue().splitlines()[0])
    assert emitted["kind"] == "user_question"
    assert emitted["request_id"] == "q-1"
    assert emitted["question"] == "Proceed with deploy?"
    assert emitted["options"] == ["yes", "no"]


def test_interrupted_true_after_interrupt_op() -> None:
    scripted = _lines({"op": "turn_input", "text": "hello"}, {"op": "interrupt"})
    client = StdioClient(stdin=scripted, stdout=io.StringIO())

    first = client.next_op()
    assert first == {"op": "turn_input", "text": "hello"}

    # "interrupt" is not surfaced as a queued op -- interrupted() is the sole
    # signal for it -- so the next read hits EOF.
    second = client.next_op()
    assert second is None
    assert client.interrupted() is True
