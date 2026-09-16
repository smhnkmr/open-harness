"""Tests for the anthropic and openai-compatible adapters (spec 5.2, 5.4, 5.5).

No network calls: the vendor SDK clients are monkeypatched with fakes that
return canned SSE-event / chunk iterators shaped like the real SDK objects.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import anthropic
import httpx2
import openai

from open_harness.model.adapters._errors import map_provider_exception
from open_harness.model.adapters.anthropic import AnthropicAdapter
from open_harness.model.adapters.openai_compatible import OpenAICompatibleAdapter
from open_harness.model.reducer import reduce
from open_harness.model.types import (
    BOUNDARY,
    Block,
    Message,
    ModelRequest,
    ThinkingConfig,
    ToolSpec,
)

# --------------------------------------------------------------------------- fakes


class _FakeAnthropicMessages:
    def __init__(self, events: list[Any]) -> None:
        self._events = events
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return iter(self._events)


class _FakeAnthropicClient:
    def __init__(self, events: list[Any]) -> None:
        self.messages = _FakeAnthropicMessages(events)


class _RaisingAnthropicMessages:
    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def create(self, **kwargs: Any) -> Any:
        raise self._exc


class _RaisingAnthropicClient:
    def __init__(self, exc: Exception) -> None:
        self.messages = _RaisingAnthropicMessages(exc)


class _FakeOpenAICompletions:
    def __init__(self, respond: Any) -> None:
        self._respond = respond
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self._respond(kwargs)


class _FakeOpenAIClient:
    def __init__(self, respond: Any) -> None:
        self.chat = SimpleNamespace(completions=_FakeOpenAICompletions(respond))


def _rate_limit_error(*, headers: dict[str, str] | None = None) -> anthropic.RateLimitError:
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    body = {"error": {"type": "rate_limit_error", "message": "slow down"}}
    resp = httpx2.Response(429, request=request, headers=headers or {}, json=body)
    return anthropic.RateLimitError("slow down", response=resp, body=body)


# --------------------------------------------------------------------------- anthropic: request conversion


def _basic_request(**overrides: Any) -> ModelRequest:
    defaults: dict[str, Any] = {
        "system": [
            Block.text_block("static identity"),
            Block.text_block("more static"),
            Block.text_block(BOUNDARY),
            Block.text_block("dynamic tail"),
        ],
        "messages": [
            Message(role="user", blocks=[Block.text_block("first")]),
            Message(role="user", blocks=[Block.text_block("second")]),
            Message(role="assistant", blocks=[Block.text_block("reply")]),
        ],
        "tools": [],
        "max_output_tokens": 1024,
    }
    defaults.update(overrides)
    return ModelRequest(**defaults)


def test_anthropic_cache_control_on_last_system_block_before_boundary() -> None:
    adapter = AnthropicAdapter(api_key="k")
    payload = adapter.build_payload("claude-x", _basic_request())
    system = payload["system"]
    # boundary marker itself must be dropped
    assert all(b["text"] != BOUNDARY for b in system)
    assert system == [
        {"type": "text", "text": "static identity"},
        {"type": "text", "text": "more static", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "dynamic tail"},
    ]


def test_anthropic_merges_consecutive_user_messages() -> None:
    adapter = AnthropicAdapter(api_key="k")
    payload = adapter.build_payload("claude-x", _basic_request())
    messages = payload["messages"]
    # the two consecutive user messages collapse into one
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert [c["text"] for c in messages[0]["content"]] == ["first", "second"]


def test_anthropic_cache_control_on_last_message_last_block() -> None:
    adapter = AnthropicAdapter(api_key="k")
    payload = adapter.build_payload("claude-x", _basic_request())
    last_block = payload["messages"][-1]["content"][-1]
    assert last_block["cache_control"] == {"type": "ephemeral"}


def test_anthropic_tool_call_and_tool_result_conversion() -> None:
    req = _basic_request(
        messages=[
            Message(
                role="assistant",
                blocks=[Block(type="tool_call", tool_call_id="call_1", name="read", args={"path": "a"})],
            ),
            Message(
                role="user",
                blocks=[Block.tool_result("call_1", "file contents", is_error=False)],
            ),
        ]
    )
    adapter = AnthropicAdapter(api_key="k")
    payload = adapter.build_payload("claude-x", req)
    assistant_block = payload["messages"][0]["content"][0]
    assert assistant_block == {
        "type": "tool_use",
        "id": "call_1",
        "name": "read",
        "input": {"path": "a"},
    }
    user_block = payload["messages"][1]["content"][0]
    assert user_block["type"] == "tool_result"
    assert user_block["tool_use_id"] == "call_1"
    assert user_block["content"] == "file contents"


def test_anthropic_thinking_block_resends_native_verbatim() -> None:
    native = {"type": "thinking", "thinking": "hmm", "signature": "sig"}
    req = _basic_request(
        messages=[
            Message(role="assistant", blocks=[Block(type="thinking", native=native)]),
            Message(role="user", blocks=[Block.text_block("go on")]),
        ]
    )
    adapter = AnthropicAdapter(api_key="k")
    payload = adapter.build_payload("claude-x", req)
    # the assistant message's thinking block is not the last block of the
    # last message here, so it must be untouched by cache-control placement.
    assert payload["messages"][0]["content"][0] is native


def test_anthropic_thinking_without_native_is_omitted() -> None:
    req = _basic_request(
        messages=[
            Message(
                role="assistant",
                blocks=[Block(type="thinking", text="unsent"), Block.text_block("kept")],
            )
        ]
    )
    adapter = AnthropicAdapter(api_key="k")
    payload = adapter.build_payload("claude-x", req)
    content = payload["messages"][0]["content"]
    assert len(content) == 1
    assert content[0]["text"] == "kept"


def test_anthropic_thinking_config_sets_budget_and_bumps_max_tokens() -> None:
    req = _basic_request(max_output_tokens=100, thinking=ThinkingConfig(enabled=True, budget_tokens=None))
    adapter = AnthropicAdapter(api_key="k")
    payload = adapter.build_payload("claude-x", req)
    assert payload["thinking"] == {"type": "enabled", "budget_tokens": 4096}
    assert payload["max_tokens"] > 4096


# --------------------------------------------------------------------------- anthropic: streaming


def test_anthropic_stream_maps_sse_events_to_neutral_events_and_reduces() -> None:
    events = [
        SimpleNamespace(
            type="message_start",
            message=SimpleNamespace(
                usage=SimpleNamespace(
                    input_tokens=50, cache_read_input_tokens=5, cache_creation_input_tokens=2
                )
            ),
        ),
        SimpleNamespace(type="content_block_start", index=0, content_block=SimpleNamespace(type="text")),
        SimpleNamespace(
            type="content_block_delta", index=0, delta=SimpleNamespace(type="text_delta", text="Hello ")
        ),
        SimpleNamespace(
            type="content_block_delta", index=0, delta=SimpleNamespace(type="text_delta", text="there")
        ),
        SimpleNamespace(type="content_block_stop", index=0),
        SimpleNamespace(
            type="content_block_start",
            index=1,
            content_block=SimpleNamespace(type="tool_use", id="call_1", name="read_file"),
        ),
        SimpleNamespace(
            type="content_block_delta",
            index=1,
            delta=SimpleNamespace(type="input_json_delta", partial_json='{"path": '),
        ),
        SimpleNamespace(
            type="content_block_delta",
            index=1,
            delta=SimpleNamespace(type="input_json_delta", partial_json='"a.py"}'),
        ),
        SimpleNamespace(type="content_block_stop", index=1),
        SimpleNamespace(
            type="message_delta",
            delta=SimpleNamespace(stop_reason="end_turn"),
            usage=SimpleNamespace(output_tokens=12),
        ),
        SimpleNamespace(type="message_stop"),
    ]
    adapter = AnthropicAdapter(api_key="k")
    adapter._client_instance = _FakeAnthropicClient(events)

    stream_events = list(adapter.stream("claude-x", _basic_request(tools=[])))
    resp = reduce(stream_events, model="anthropic:claude-x", role="main")

    assert resp.message.text == "Hello there"
    tool_calls = [b for b in resp.message.blocks if b.type == "tool_call"]
    assert len(tool_calls) == 1
    assert tool_calls[0].tool_call_id == "call_1"
    assert tool_calls[0].name == "read_file"
    assert tool_calls[0].args == {"path": "a.py"}
    assert resp.usage.input_tokens == 50
    assert resp.usage.output_tokens == 12
    assert resp.usage.cache_read_tokens == 5
    assert resp.usage.cache_write_tokens == 2
    # content decides: a tool call is present, so stop is tool_use even
    # though the adapter's raw stop_reason was end_turn.
    assert resp.stop.reason == "tool_use"


def test_anthropic_stream_thinking_signature_round_trips() -> None:
    events = [
        SimpleNamespace(
            type="message_start", message=SimpleNamespace(usage=SimpleNamespace(input_tokens=1))
        ),
        SimpleNamespace(
            type="content_block_start", index=0, content_block=SimpleNamespace(type="thinking")
        ),
        SimpleNamespace(
            type="content_block_delta",
            index=0,
            delta=SimpleNamespace(type="thinking_delta", thinking="pondering"),
        ),
        SimpleNamespace(
            type="content_block_delta",
            index=0,
            delta=SimpleNamespace(type="signature_delta", signature="sig-abc"),
        ),
        SimpleNamespace(
            type="message_delta", delta=SimpleNamespace(stop_reason="end_turn"), usage=SimpleNamespace()
        ),
    ]
    adapter = AnthropicAdapter(api_key="k")
    adapter._client_instance = _FakeAnthropicClient(events)
    resp = reduce(list(adapter.stream("claude-x", _basic_request())), model="m", role="main")
    block = resp.message.blocks[0]
    assert block.type == "thinking"
    assert block.native == {"type": "thinking", "thinking": "pondering", "signature": "sig-abc"}


def test_anthropic_stream_never_raises_yields_provider_error_instead() -> None:
    adapter = AnthropicAdapter(api_key="k")
    adapter._client_instance = _RaisingAnthropicClient(_rate_limit_error(headers={"retry-after": "2"}))
    events = list(adapter.stream("claude-x", _basic_request()))
    assert len(events) == 1
    err = events[0]
    assert err.kind == "rate_limit"
    assert err.retryable is True
    assert err.retry_after == 2.0


# --------------------------------------------------------------------------- error mapping (shared)


def test_map_rate_limit_error() -> None:
    err = map_provider_exception(_rate_limit_error(headers={"retry-after": "3.5"}))
    assert err.kind == "rate_limit"
    assert err.retryable is True
    assert err.retry_after == 3.5
    assert err.status == 429


def test_map_overloaded_error() -> None:
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    body = {"error": {"type": "overloaded_error", "message": "overloaded"}}
    resp = httpx2.Response(529, request=request, json=body)
    exc = anthropic.OverloadedError("overloaded", response=resp, body=body)
    err = map_provider_exception(exc)
    assert err.kind == "overloaded"
    assert err.retryable is True


def test_map_context_too_long() -> None:
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    body = {"error": {"message": "prompt is too long: 250000 tokens > 200000 maximum"}}
    resp = httpx2.Response(400, request=request, json=body)
    exc = anthropic.BadRequestError(body["error"]["message"], response=resp, body=body)
    err = map_provider_exception(exc)
    assert err.kind == "context_too_long"
    assert err.retryable is False


def test_map_auth_error() -> None:
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    body = {"error": {"message": "invalid api key"}}
    resp = httpx2.Response(401, request=request, json=body)
    exc = anthropic.AuthenticationError("invalid api key", response=resp, body=body)
    err = map_provider_exception(exc)
    assert err.kind == "auth"


def test_map_server_error() -> None:
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    body = {"error": {"message": "internal error"}}
    resp = httpx2.Response(500, request=request, json=body)
    exc = anthropic.InternalServerError("internal error", response=resp, body=body)
    err = map_provider_exception(exc)
    assert err.kind == "server"
    assert err.retryable is True


def test_map_connection_error() -> None:
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    exc = anthropic.APIConnectionError(request=request)
    err = map_provider_exception(exc)
    assert err.kind == "network"
    assert err.retryable is True


# --------------------------------------------------------------------------- openai-compatible: request conversion


def _oa_request(**overrides: Any) -> ModelRequest:
    defaults: dict[str, Any] = {
        "system": [Block.text_block("system rules"), Block.text_block(BOUNDARY), Block.text_block("tail")],
        "messages": [
            Message(role="user", blocks=[Block.text_block("hi")]),
            Message(
                role="assistant",
                blocks=[Block(type="tool_call", tool_call_id="call_1", name="search", args={"q": "cats"})],
            ),
            Message(role="user", blocks=[Block.tool_result("call_1", "results here")]),
        ],
        "tools": [ToolSpec(name="search", description="search the web", json_schema={})],
        "max_output_tokens": 512,
    }
    defaults.update(overrides)
    return ModelRequest(**defaults)


def test_openai_system_message_drops_boundary() -> None:
    adapter = OpenAICompatibleAdapter(api_key="k")
    payload = adapter.build_payload("gpt-x", _oa_request())
    system_messages = [m for m in payload["messages"] if m["role"] == "system"]
    assert len(system_messages) == 1
    assert BOUNDARY not in system_messages[0]["content"]
    assert "system rules" in system_messages[0]["content"]
    assert "tail" in system_messages[0]["content"]


def test_openai_tool_result_becomes_tool_message() -> None:
    adapter = OpenAICompatibleAdapter(api_key="k")
    payload = adapter.build_payload("gpt-x", _oa_request())
    tool_messages = [m for m in payload["messages"] if m["role"] == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["tool_call_id"] == "call_1"
    assert tool_messages[0]["content"] == "results here"


def test_openai_assistant_tool_call_shape() -> None:
    adapter = OpenAICompatibleAdapter(api_key="k")
    payload = adapter.build_payload("gpt-x", _oa_request())
    assistant_messages = [m for m in payload["messages"] if m["role"] == "assistant"]
    assert len(assistant_messages) == 1
    tool_calls = assistant_messages[0]["tool_calls"]
    assert tool_calls == [
        {"id": "call_1", "type": "function", "function": {"name": "search", "arguments": '{"q": "cats"}'}}
    ]


def test_openai_tools_use_function_dialect() -> None:
    adapter = OpenAICompatibleAdapter(api_key="k")
    payload = adapter.build_payload("gpt-x", _oa_request())
    assert payload["tools"][0]["type"] == "function"
    assert payload["tools"][0]["function"]["name"] == "search"


def test_openai_payload_hook_can_patch_payload() -> None:
    class Vendor(OpenAICompatibleAdapter):
        def payload_hook(self, payload: dict[str, Any]) -> dict[str, Any]:
            payload["vendor_flag"] = True
            return payload

    adapter = Vendor(api_key="k")
    payload = adapter.build_payload("gpt-x", _oa_request())
    assert payload["vendor_flag"] is True


# --------------------------------------------------------------------------- openai-compatible: streaming


def _text_chunk(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content=text, tool_calls=None, reasoning_content=None),
                finish_reason=None,
            )
        ],
        usage=None,
    )


def _tool_call_chunk(index: int, *, id_: str | None, name: str | None, args_fragment: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(
                    content=None,
                    reasoning_content=None,
                    tool_calls=[
                        SimpleNamespace(
                            index=index,
                            id=id_,
                            function=SimpleNamespace(name=name, arguments=args_fragment),
                        )
                    ],
                ),
                finish_reason=None,
            )
        ],
        usage=None,
    )


def _finish_chunk(reason: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=None, tool_calls=None, reasoning_content=None), finish_reason=reason)],
        usage=None,
    )


def _usage_chunk(prompt: int, completion: int) -> SimpleNamespace:
    return SimpleNamespace(choices=[], usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion))


def test_openai_stream_maps_chunks_to_events_and_reduces() -> None:
    chunks = [
        _text_chunk("Hello "),
        _text_chunk("world"),
        _tool_call_chunk(0, id_="call_1", name="search", args_fragment='{"q": '),
        _tool_call_chunk(0, id_=None, name=None, args_fragment='"cats"}'),
        _finish_chunk("tool_calls"),
        _usage_chunk(20, 8),
    ]
    adapter = OpenAICompatibleAdapter(api_key="k")
    adapter._client_instance = _FakeOpenAIClient(lambda kwargs: iter(chunks))

    events = list(adapter.stream("gpt-x", _oa_request(tools=[])))
    resp = reduce(events, model="m", role="main")

    assert resp.message.text == "Hello world"
    tool_calls = [b for b in resp.message.blocks if b.type == "tool_call"]
    assert tool_calls[0].tool_call_id == "call_1"
    assert tool_calls[0].name == "search"
    assert tool_calls[0].args == {"q": "cats"}
    assert resp.usage.input_tokens == 20
    assert resp.usage.output_tokens == 8
    assert resp.stop.reason == "tool_use"


def test_openai_stream_finish_reason_length_maps_to_max_tokens() -> None:
    chunks = [_text_chunk("partial"), _finish_chunk("length")]
    adapter = OpenAICompatibleAdapter(api_key="k")
    adapter._client_instance = _FakeOpenAIClient(lambda kwargs: iter(chunks))
    resp = reduce(list(adapter.stream("gpt-x", _oa_request(tools=[]))), model="m", role="main")
    assert resp.stop.reason == "max_tokens"


def test_openai_stream_retries_once_without_include_usage() -> None:
    request = httpx2.Request("POST", "https://api.openai.com/v1/chat/completions")
    body = {"error": {"message": "stream_options is not supported"}}
    resp = httpx2.Response(400, request=request, json=body)
    err = openai.BadRequestError("stream_options is not supported", response=resp, body=body)

    def respond(kwargs: dict[str, Any]) -> Any:
        if "stream_options" in kwargs:
            raise err
        return iter([_text_chunk("ok"), _finish_chunk("stop")])

    adapter = OpenAICompatibleAdapter(api_key="k")
    fake_completions = _FakeOpenAICompletions(respond)
    adapter._client_instance = SimpleNamespace(chat=SimpleNamespace(completions=fake_completions))

    resp_events = list(adapter.stream("gpt-x", _oa_request(tools=[])))
    reduced = reduce(resp_events, model="m", role="main")
    assert reduced.message.text == "ok"
    assert len(fake_completions.calls) == 2
    assert "stream_options" in fake_completions.calls[0]
    assert "stream_options" not in fake_completions.calls[1]


def test_openai_stream_never_raises_yields_provider_error_instead() -> None:
    request = httpx2.Request("POST", "https://api.openai.com/v1/chat/completions")
    body = {"error": {"message": "rate limited"}}
    resp = httpx2.Response(429, request=request, headers={"retry-after": "1"}, json=body)
    err = openai.RateLimitError("rate limited", response=resp, body=body)

    def respond(kwargs: dict[str, Any]) -> Any:
        raise err

    adapter = OpenAICompatibleAdapter(api_key="k")
    adapter._client_instance = SimpleNamespace(
        chat=SimpleNamespace(completions=_FakeOpenAICompletions(respond))
    )
    events = list(adapter.stream("gpt-x", _oa_request(tools=[])))
    assert len(events) == 1
    assert events[0].kind == "rate_limit"
