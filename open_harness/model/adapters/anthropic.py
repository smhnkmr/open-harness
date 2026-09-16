"""Anthropic Messages API adapter.

Spec: open-harness-spec.md sections 5.2, 5.4, 5.5.

Streams via the raw SSE event iterator (`client.messages.create(..., stream=True)`
iterated directly), not the `MessageStream` convenience wrapper, so every
event maps 1:1 onto the neutral `Event` union with no hidden accumulation.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import anthropic

from open_harness.model.adapter import Adapter
from open_harness.model.adapters._errors import map_provider_exception
from open_harness.model.schema import to_anthropic_tool
from open_harness.model.types import (
    BOUNDARY,
    Block,
    CapabilityFlags,
    Event,
    Message,
    ModelRequest,
    Stop,
    TextDelta,
    ThinkingDelta,
    ToolCallFragment,
    Usage,
)

_STOP_REASON_MAP = {
    "end_turn": "end_turn",
    "tool_use": "tool_use",
    "max_tokens": "max_tokens",
    "stop_sequence": "stop_sequence",
}

_DEFAULT_THINKING_BUDGET = 4096


class AnthropicAdapter(Adapter):
    """Adapter for api.anthropic.com's Messages API."""

    name = "anthropic"
    flags = CapabilityFlags(
        requires_role_alternation=True,
        supports_cache_control=True,
        supports_thinking=True,
    )

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 600.0,
        extra: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(api_key=api_key, base_url=base_url, timeout=timeout, extra=extra)
        self._client_instance: Any | None = None

    def _client(self) -> Any:
        if self._client_instance is None:
            self._client_instance = anthropic.Anthropic(
                api_key=self.api_key, base_url=self.base_url, timeout=self.timeout, max_retries=0
            )
        return self._client_instance

    # ----------------------------------------------------------------- request

    def build_payload(self, model: str, req: ModelRequest) -> dict[str, Any]:
        """Build the `messages.create` kwargs. Exposed for testing without a
        network call."""
        system = _convert_system(req.system)
        messages = _merge_alternating(_convert_messages(req.messages))
        if messages and messages[-1]["content"]:
            messages[-1]["content"][-1] = {
                **messages[-1]["content"][-1],
                "cache_control": {"type": "ephemeral"},
            }

        payload: dict[str, Any] = {
            "model": model,
            "system": system,
            "messages": messages,
            "max_tokens": req.max_output_tokens,
        }
        if req.tools:
            payload["tools"] = [to_anthropic_tool(t) for t in req.tools]
        if req.stop_sequences:
            payload["stop_sequences"] = req.stop_sequences
        if req.thinking is not None and req.thinking.enabled:
            budget = req.thinking.budget_tokens or _DEFAULT_THINKING_BUDGET
            payload["thinking"] = {"type": "enabled", "budget_tokens": budget}
            payload["max_tokens"] = max(payload["max_tokens"], budget + 1)
        return payload

    # ------------------------------------------------------------------ stream

    def stream(self, model: str, req: ModelRequest) -> Iterator[Event]:
        try:
            payload = self.build_payload(model, req)
            raw_stream = self._client().messages.create(**payload, stream=True)
        except Exception as exc:  # noqa: BLE001 - never raise out of stream()
            yield map_provider_exception(exc)
            return

        input_tokens = 0
        cache_read = 0
        cache_write = 0
        output_tokens = 0
        thinking_text_by_index: dict[int, str] = {}
        stop_reason = "end_turn"
        native_stop_reason: str | None = None

        try:
            for sse in raw_stream:
                kind = sse.type
                if kind == "message_start":
                    usage = sse.message.usage
                    input_tokens = getattr(usage, "input_tokens", 0) or 0
                    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
                    cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
                elif kind == "content_block_start":
                    block = sse.content_block
                    if block.type == "tool_use":
                        yield ToolCallFragment(
                            index=sse.index, id=block.id, name=block.name, args_fragment=""
                        )
                elif kind == "content_block_delta":
                    delta = sse.delta
                    if delta.type == "text_delta":
                        yield TextDelta(text=delta.text)
                    elif delta.type == "thinking_delta":
                        buf = thinking_text_by_index.get(sse.index, "") + delta.thinking
                        thinking_text_by_index[sse.index] = buf
                        yield ThinkingDelta(
                            text=delta.thinking, native={"type": "thinking", "thinking": buf}
                        )
                    elif delta.type == "signature_delta":
                        yield ThinkingDelta(text="", native={"signature": delta.signature})
                    elif delta.type == "input_json_delta":
                        yield ToolCallFragment(index=sse.index, args_fragment=delta.partial_json)
                elif kind == "message_delta":
                    native_stop_reason = sse.delta.stop_reason
                    stop_reason = _STOP_REASON_MAP.get(native_stop_reason, "other")
                    usage = getattr(sse, "usage", None)
                    if usage is not None:
                        output_tokens = getattr(usage, "output_tokens", 0) or 0
                # message_stop and content_block_stop carry no extra data.
        except Exception as exc:  # noqa: BLE001 - never raise out of stream()
            yield map_provider_exception(exc)
            return

        yield Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
        )
        yield Stop(reason=stop_reason, native_reason=native_stop_reason)


# --------------------------------------------------------------------------- request conversion


def _convert_system(system: list[Block]) -> list[dict[str, Any]]:
    marker_index = next((i for i, b in enumerate(system) if b.text == BOUNDARY), None)
    out: list[dict[str, Any]] = []
    for i, block in enumerate(system):
        if i == marker_index:
            continue
        entry: dict[str, Any] = {"type": "text", "text": block.text or ""}
        if marker_index is not None and i == marker_index - 1:
            entry["cache_control"] = {"type": "ephemeral"}
        out.append(entry)
    return out


def _convert_messages(messages: list[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for msg in messages:
        content = _convert_blocks(msg.blocks)
        if content:
            out.append({"role": msg.role, "content": content})
    return out


def _convert_blocks(blocks: list[Block]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for block in blocks:
        if block.type == "text":
            out.append({"type": "text", "text": block.text or ""})
        elif block.type == "tool_call":
            out.append(
                {
                    "type": "tool_use",
                    "id": block.tool_call_id,
                    "name": block.name,
                    "input": block.args or {},
                }
            )
        elif block.type == "tool_result":
            out.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.tool_call_id,
                    "content": _convert_tool_result_content(block),
                    "is_error": block.is_error,
                }
            )
        elif block.type == "thinking":
            if block.native is not None:
                out.append(block.native)
            # else: omit -- cannot round-trip a thinking block without its
            # native signature.
        elif block.type == "image":
            out.append(_convert_image(block))
        # file, citation, non_standard: no Anthropic-native shape defined by
        # the spec text; dropped rather than guessed at.
    return out


def _convert_tool_result_content(block: Block) -> Any:
    if block.content is not None:
        parts: list[dict[str, Any]] = []
        for nested in block.content:
            if nested.type == "image":
                parts.append(_convert_image(nested))
            else:
                parts.append({"type": "text", "text": nested.text or ""})
        return parts
    return block.text or ""


def _convert_image(block: Block) -> dict[str, Any]:
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": block.media_type, "data": block.data},
    }


def _merge_alternating(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge consecutive same-role messages (adapter requires alternation)."""
    merged: list[dict[str, Any]] = []
    for msg in messages:
        if merged and merged[-1]["role"] == msg["role"]:
            merged[-1]["content"] = [*merged[-1]["content"], *msg["content"]]
        else:
            merged.append({"role": msg["role"], "content": list(msg["content"])})
    return merged
