"""Base adapter for OpenAI-shaped Chat Completions endpoints.

Spec: open-harness-spec.md sections 5.2, 5.4, 5.5.

Vendors (Ollama, Groq, DeepSeek, OpenRouter, ...) subclass and override
`payload_hook` only; message conversion and streaming never move.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import openai

from open_harness.model.adapter import Adapter
from open_harness.model.adapters._errors import map_provider_exception
from open_harness.model.schema import to_openai_tool
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

_FINISH_REASON_MAP = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "length": "max_tokens",
}


class OpenAICompatibleAdapter(Adapter):
    """Adapter for any OpenAI Chat Completions-compatible host."""

    name = "openai-compatible"
    flags = CapabilityFlags()  # defaults: openai tool dialect, no cache/thinking round-trip

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
            self._client_instance = openai.OpenAI(
                api_key=self.api_key, base_url=self.base_url, timeout=self.timeout, max_retries=0
            )
        return self._client_instance

    def payload_hook(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Vendor subclasses override this to patch quirks. Default: identity."""
        return payload

    # ----------------------------------------------------------------- request

    def build_payload(self, model: str, req: ModelRequest) -> dict[str, Any]:
        """Build the `chat.completions.create` kwargs. Exposed for testing
        without a network call."""
        messages = _convert_system(req.system) + _convert_messages(req.messages)
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": req.max_output_tokens,
        }
        if req.tools:
            payload["tools"] = [to_openai_tool(t) for t in req.tools]
        if req.stop_sequences:
            payload["stop"] = req.stop_sequences
        return self.payload_hook(payload)

    # ------------------------------------------------------------------ stream

    def stream(self, model: str, req: ModelRequest) -> Iterator[Event]:
        try:
            payload = self.build_payload(model, req)
            raw_stream = self._create_stream(payload)
        except Exception as exc:  # noqa: BLE001 - never raise out of stream()
            yield map_provider_exception(exc)
            return

        input_tokens = 0
        output_tokens = 0
        stop_reason = "end_turn"
        native_finish_reason: str | None = None
        tool_call_index_for_id: dict[str, int] = {}

        try:
            for chunk in raw_stream:
                usage = getattr(chunk, "usage", None)
                if usage is not None:
                    input_tokens = getattr(usage, "prompt_tokens", 0) or 0
                    output_tokens = getattr(usage, "completion_tokens", 0) or 0
                for choice in getattr(chunk, "choices", None) or []:
                    delta = choice.delta
                    if getattr(delta, "content", None):
                        yield TextDelta(text=delta.content)
                    reasoning = getattr(delta, "reasoning_content", None) or getattr(
                        delta, "reasoning", None
                    )
                    if reasoning:
                        yield ThinkingDelta(text=reasoning)
                    for tc in getattr(delta, "tool_calls", None) or []:
                        index = tc.index if tc.index is not None else tool_call_index_for_id.setdefault(
                            tc.id or "", len(tool_call_index_for_id)
                        )
                        function = getattr(tc, "function", None)
                        yield ToolCallFragment(
                            index=index,
                            id=tc.id,
                            name=getattr(function, "name", None),
                            args_fragment=getattr(function, "arguments", None) or "",
                        )
                    if choice.finish_reason:
                        native_finish_reason = choice.finish_reason
                        stop_reason = _FINISH_REASON_MAP.get(choice.finish_reason, "other")
        except Exception as exc:  # noqa: BLE001 - never raise out of stream()
            yield map_provider_exception(exc)
            return

        yield Usage(input_tokens=input_tokens, output_tokens=output_tokens)
        yield Stop(reason=stop_reason, native_reason=native_finish_reason)

    def _create_stream(self, payload: dict[str, Any]) -> Any:
        """`stream=True` with `include_usage`, tolerating servers that reject it."""
        try:
            return self._client().chat.completions.create(
                **payload, stream=True, stream_options={"include_usage": True}
            )
        except openai.BadRequestError as exc:
            message = (getattr(exc, "message", None) or str(exc)).lower()
            if "stream_options" in message or "include_usage" in message:
                return self._client().chat.completions.create(**payload, stream=True)
            raise


# --------------------------------------------------------------------------- request conversion


def _convert_system(system: list[Block]) -> list[dict[str, Any]]:
    text = "\n\n".join(b.text or "" for b in system if b.text != BOUNDARY and b.text)
    return [{"role": "system", "content": text}] if text else []


def _convert_messages(messages: list[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for msg in messages:
        tool_results = [b for b in msg.blocks if b.type == "tool_result"]
        other = [b for b in msg.blocks if b.type != "tool_result"]
        for result in tool_results:
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": result.tool_call_id,
                    "content": _stringify_tool_result(result),
                }
            )
        if msg.role == "assistant":
            converted = _convert_assistant(other)
            if converted is not None:
                out.append(converted)
        elif other:
            out.append({"role": msg.role, "content": _convert_user_content(other)})
    return out


def _stringify_tool_result(block: Block) -> str:
    if block.content is not None:
        parts = []
        for nested in block.content:
            if nested.type == "image":
                parts.append("[image]")
            else:
                parts.append(nested.text or "")
        return "\n".join(parts)
    return block.text or ""


def _convert_assistant(blocks: list[Block]) -> dict[str, Any] | None:
    text_parts = [b.text or "" for b in blocks if b.type == "text"]
    tool_calls = [b for b in blocks if b.type == "tool_call"]
    if not text_parts and not tool_calls:
        return None
    message: dict[str, Any] = {"role": "assistant", "content": "".join(text_parts) or None}
    if tool_calls:
        message["tool_calls"] = [
            {
                "id": tc.tool_call_id,
                "type": "function",
                "function": {"name": tc.name, "arguments": json.dumps(tc.args or {})},
            }
            for tc in tool_calls
        ]
    return message


def _convert_user_content(blocks: list[Block]) -> Any:
    if len(blocks) == 1 and blocks[0].type == "text":
        return blocks[0].text or ""
    parts: list[dict[str, Any]] = []
    for block in blocks:
        if block.type == "image":
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{block.media_type};base64,{block.data}"},
                }
            )
        else:
            parts.append({"type": "text", "text": block.text or ""})
    return parts
