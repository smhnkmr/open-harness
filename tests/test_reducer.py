"""Tests for open_harness.model.reducer (spec 5.3)."""

from __future__ import annotations

import pytest

from open_harness.model.reducer import ProviderErrorRaised, reduce
from open_harness.model.types import (
    ProviderError,
    Stop,
    TextDelta,
    ThinkingDelta,
    ToolCallFragment,
    Usage,
)


def test_merges_consecutive_text_deltas() -> None:
    events = [TextDelta("Hello, "), TextDelta("world"), TextDelta("!"), Stop(reason="end_turn")]
    resp = reduce(events, model="anthropic:claude", role="main")
    assert len(resp.message.blocks) == 1
    assert resp.message.blocks[0].type == "text"
    assert resp.message.blocks[0].text == "Hello, world!"


def test_text_then_thinking_then_text_produces_three_blocks() -> None:
    events = [
        TextDelta("a"),
        ThinkingDelta(text="thinking..."),
        TextDelta("b"),
        Stop(reason="end_turn"),
    ]
    resp = reduce(events, model="m", role="main")
    types = [b.type for b in resp.message.blocks]
    assert types == ["text", "thinking", "text"]
    assert resp.message.blocks[0].text == "a"
    assert resp.message.blocks[2].text == "b"


def test_thinking_delta_merges_text_and_native() -> None:
    events = [
        ThinkingDelta(text="foo", native={"type": "thinking", "thinking": "foo"}),
        ThinkingDelta(text="bar", native={"type": "thinking", "thinking": "foobar"}),
        ThinkingDelta(text="", native={"signature": "sig123"}),
        Stop(reason="end_turn"),
    ]
    resp = reduce(events, model="m", role="main")
    block = resp.message.blocks[0]
    assert block.type == "thinking"
    assert block.text == "foobar"
    assert block.native == {"type": "thinking", "thinking": "foobar", "signature": "sig123"}


def test_tool_call_accumulates_across_three_fragments() -> None:
    events = [
        ToolCallFragment(index=0, id="call_1", name="read_file", args_fragment=""),
        ToolCallFragment(index=0, args_fragment='{"path": '),
        ToolCallFragment(index=0, args_fragment='"a.py", '),
        ToolCallFragment(index=0, args_fragment='"limit": 10}'),
        Stop(reason="end_turn"),
    ]
    resp = reduce(events, model="m", role="main")
    assert len(resp.invalid_tool_calls) == 0
    assert len(resp.message.blocks) == 1
    call = resp.message.blocks[0]
    assert call.type == "tool_call"
    assert call.tool_call_id == "call_1"
    assert call.name == "read_file"
    assert call.args == {"path": "a.py", "limit": 10}


def test_single_complete_fragment() -> None:
    events = [
        ToolCallFragment(index=0, id="call_1", name="ping", args_fragment="{}"),
        Stop(reason="end_turn"),
    ]
    resp = reduce(events, model="m", role="main")
    assert resp.message.blocks[0].args == {}
    assert resp.message.blocks[0].name == "ping"


def test_two_tool_calls_keep_first_non_none_id_and_name() -> None:
    events = [
        ToolCallFragment(index=0, id="call_a", name="foo", args_fragment=""),
        ToolCallFragment(index=1, id="call_b", name="bar", args_fragment=""),
        ToolCallFragment(index=0, args_fragment='{"x": 1}'),
        ToolCallFragment(index=1, args_fragment='{"y": 2}'),
        Stop(reason="end_turn"),
    ]
    resp = reduce(events, model="m", role="main")
    by_id = {b.tool_call_id: b for b in resp.message.blocks}
    assert by_id["call_a"].name == "foo"
    assert by_id["call_a"].args == {"x": 1}
    assert by_id["call_b"].name == "bar"
    assert by_id["call_b"].args == {"y": 2}


def test_truncated_object_is_invalid_tool_call() -> None:
    events = [
        ToolCallFragment(index=0, id="call_1", name="foo", args_fragment='{"a":'),
        Stop(reason="end_turn"),
    ]
    resp = reduce(events, model="m", role="main")
    assert resp.message.blocks == []
    assert len(resp.invalid_tool_calls) == 1
    bad = resp.invalid_tool_calls[0]
    assert bad.id == "call_1"
    assert bad.raw_args == '{"a":'
    assert "invalid JSON" in bad.error


def test_non_object_json_is_invalid_tool_call() -> None:
    events = [
        ToolCallFragment(index=0, id="call_1", name="foo", args_fragment='"just a string"'),
        Stop(reason="end_turn"),
    ]
    resp = reduce(events, model="m", role="main")
    assert resp.message.blocks == []
    assert len(resp.invalid_tool_calls) == 1
    bad = resp.invalid_tool_calls[0]
    assert "JSON object" in bad.error


def test_empty_args_fragment_is_valid_empty_dict() -> None:
    events = [
        ToolCallFragment(index=0, id="call_1", name="foo", args_fragment=""),
        Stop(reason="end_turn"),
    ]
    resp = reduce(events, model="m", role="main")
    assert resp.message.blocks[0].args == {}


def test_stop_forced_to_tool_use_when_tool_calls_present_even_if_end_turn() -> None:
    events = [
        ToolCallFragment(index=0, id="call_1", name="foo", args_fragment="{}"),
        Stop(reason="end_turn", native_reason="end_turn"),
    ]
    resp = reduce(events, model="m", role="main")
    assert resp.stop.reason == "tool_use"


def test_stop_not_overridden_when_no_valid_tool_calls() -> None:
    events = [TextDelta("hi"), Stop(reason="end_turn")]
    resp = reduce(events, model="m", role="main")
    assert resp.stop.reason == "end_turn"


def test_usage_last_wins_but_cache_fields_sum() -> None:
    events = [
        Usage(input_tokens=100, output_tokens=0, cache_read_tokens=10, cache_write_tokens=5),
        Usage(input_tokens=100, output_tokens=42, cache_read_tokens=0, cache_write_tokens=0),
        Stop(reason="end_turn"),
    ]
    resp = reduce(events, model="m", role="main")
    assert resp.usage.input_tokens == 100
    assert resp.usage.output_tokens == 42
    assert resp.usage.cache_read_tokens == 10
    assert resp.usage.cache_write_tokens == 5


def test_provider_error_raises_provider_error_raised() -> None:
    err = ProviderError(kind="rate_limit", message="slow down", retryable=True, retry_after=1.0)
    events = [TextDelta("partial"), err]
    with pytest.raises(ProviderErrorRaised) as exc_info:
        reduce(events, model="m", role="main")
    assert exc_info.value.error is err


def test_missing_stop_defaults_to_end_turn() -> None:
    resp = reduce([TextDelta("hi")], model="m", role="main")
    assert resp.stop.reason == "end_turn"
