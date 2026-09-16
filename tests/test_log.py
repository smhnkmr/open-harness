"""Tests for `open_harness.kernel.log.EventLog` and `new_session`."""

from __future__ import annotations

import uuid
from pathlib import Path

from open_harness.kernel.log import EventLog, new_session
from open_harness.kernel.serde import serialize_message
from open_harness.model.types import Block, Message


def test_append_assigns_increasing_seq(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "s1.jsonl")
    r1 = log.append("info", note="a")
    r2 = log.append("info", note="b")
    assert r1["seq"] == 1
    assert r2["seq"] == 2
    assert "ts" in r1 and "ts" in r2
    assert r1["kind"] == "info"


def test_read_round_trips_records(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "s2.jsonl")
    log.append("info", note="hello")
    records = list(log.read())
    assert len(records) == 1
    assert records[0]["kind"] == "info"
    assert records[0]["note"] == "hello"


def test_append_flushes_without_close(tmp_path: Path) -> None:
    path = tmp_path / "s3.jsonl"
    log = EventLog(path)
    log.append("info", note="x")
    # Read through a fresh handle, not through the EventLog, to prove the
    # write actually reached disk without needing log.close().
    content = path.read_text(encoding="utf-8")
    assert "x" in content
    assert content.count("\n") == 1


def test_reopen_continues_sequence(tmp_path: Path) -> None:
    path = tmp_path / "s4.jsonl"
    log1 = EventLog(path)
    log1.append("info", note="a")
    log1.close()

    log2 = EventLog(path)
    r = log2.append("info", note="b")
    assert r["seq"] == 2
    assert [rec["note"] for rec in log2.read()] == ["a", "b"]


def test_new_session_creates_uuid_named_file(tmp_path: Path) -> None:
    log = new_session(tmp_path)
    assert log.path.parent == tmp_path
    assert log.path.suffix == ".jsonl"
    assert log.session_id == log.path.stem
    uuid.UUID(log.session_id)  # does not raise


def test_replay_round_trips_tool_call_tool_result_and_image(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "s5.jsonl")

    user_msg = Message(role="user", blocks=[Block.text_block("please read a file")])
    log.append("user_message", message=serialize_message(user_msg))

    tool_call = Block(type="tool_call", tool_call_id="tc1", name="read", args={"path": "a.py"})
    assistant_msg = Message(role="assistant", blocks=[Block.text_block("On it."), tool_call])
    log.append("assistant_message", message=serialize_message(assistant_msg))

    image_block = Block(type="image", media_type="image/png", data="AAAA")
    result_block = Block.tool_result("tc1", [Block.text_block("file contents"), image_block])
    result_msg = Message(role="user", blocks=[result_block], meta={"tool": "read"})
    log.append("tool_result", message=serialize_message(result_msg))

    messages = log.replay_messages()
    assert len(messages) == 3

    assert messages[0].role == "user"
    assert messages[0].text == "please read a file"

    assert messages[1].role == "assistant"
    assert messages[1].text == "On it."
    assert len(messages[1].tool_calls) == 1
    assert messages[1].tool_calls[0].name == "read"
    assert messages[1].tool_calls[0].args == {"path": "a.py"}

    assert messages[2].role == "user"
    result = messages[2].blocks[0]
    assert result.type == "tool_result"
    assert result.tool_call_id == "tc1"
    assert result.content is not None
    assert result.content[0].type == "text"
    assert result.content[0].text == "file contents"
    assert result.content[1].type == "image"
    assert result.content[1].media_type == "image/png"
    assert result.content[1].data == "AAAA"


def test_separate_tool_results_stay_separate_messages(tmp_path: Path) -> None:
    """The loop pushes each tool result as its own message (one record each);
    replay must preserve that, not fold them into a single message."""
    log = EventLog(tmp_path / "s6.jsonl")
    for i in range(2):
        block = Block.tool_result(f"tc{i}", f"result {i}")
        msg = Message(role="user", blocks=[block], meta={"tool": "read"})
        log.append("tool_result", message=serialize_message(msg))

    messages = log.replay_messages()
    assert len(messages) == 2
    assert all(m.role == "user" for m in messages)
    assert messages[0].blocks[0].tool_call_id == "tc0"
    assert messages[1].blocks[0].tool_call_id == "tc1"


def test_last_boundary_and_replay_starts_after_it(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "s7.jsonl")
    msg1 = Message(role="user", blocks=[Block.text_block("first")])
    log.append("user_message", message=serialize_message(msg1))
    assert log.last_boundary() is None

    boundary = log.append("compact_boundary", removed=1)
    assert log.last_boundary() == boundary["seq"]

    msg2 = Message(role="user", blocks=[Block.text_block("after boundary")])
    log.append("user_message", message=serialize_message(msg2))

    messages = log.replay_messages()
    assert len(messages) == 1
    assert messages[0].text == "after boundary"


def test_last_boundary_none_when_absent(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "s8.jsonl")
    log.append("info", note="no boundary here")
    assert log.last_boundary() is None
