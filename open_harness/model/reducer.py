"""Kernel-owned stream reduction. Adapters never parse tool-call JSON.

Spec: open-harness-spec.md section 5.3.

`reduce()` walks the neutral `Event` stream a single adapter call produced and
folds it into one `ModelResponse`: text/thinking blocks merge across
consecutive same-type deltas, tool-call fragments are concatenated by index
and parsed with a partial-JSON-tolerant parser, usage is accumulated, and the
stop reason is corrected so that content -- not the adapter's stated reason --
decides whether the turn ends in `tool_use`.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from open_harness.model.types import (
    Block,
    Event,
    InvalidToolCall,
    Message,
    ModelResponse,
    ProviderError,
    Stop,
    TextDelta,
    ThinkingDelta,
    ToolCallFragment,
    Usage,
)


class ProviderErrorRaised(Exception):
    """Raised out of `reduce()` when the event stream carries a ProviderError.

    The gateway catches this to drive retries/fallback; the reducer itself
    never swallows or retries.
    """

    def __init__(self, error: ProviderError) -> None:
        super().__init__(error.message)
        self.error = error


@dataclass
class _ToolAccum:
    """Mutable in-progress tool call, kept in the ordered block list by
    reference so later fragments for the same index mutate it in place."""

    index: int
    id: str | None = None
    name: str | None = None
    raw_args: str = ""

    def absorb(self, frag: ToolCallFragment) -> None:
        if self.id is None and frag.id is not None:
            self.id = frag.id
        if self.name is None and frag.name is not None:
            self.name = frag.name
        self.raw_args += frag.args_fragment


def reduce(events: Iterable[Event], *, model: str, role: str) -> ModelResponse:
    """Fold a stream of `Event`s into one `ModelResponse`.

    Raises `ProviderErrorRaised` the instant a `ProviderError` event is seen;
    nothing built so far is returned in that case.
    """
    ordered: list[Block | _ToolAccum] = []
    tool_by_index: dict[int, _ToolAccum] = {}
    usage = Usage()
    stop = Stop(reason="end_turn", native_reason=None)
    saw_stop = False

    for ev in events:
        if isinstance(ev, TextDelta):
            _merge_text(ordered, ev)
        elif isinstance(ev, ThinkingDelta):
            _merge_thinking(ordered, ev)
        elif isinstance(ev, ToolCallFragment):
            accum = tool_by_index.get(ev.index)
            if accum is None:
                accum = _ToolAccum(index=ev.index)
                tool_by_index[ev.index] = accum
                ordered.append(accum)
            accum.absorb(ev)
        elif isinstance(ev, Usage):
            usage.input_tokens = ev.input_tokens
            usage.output_tokens = ev.output_tokens
            usage.cache_read_tokens += ev.cache_read_tokens
            usage.cache_write_tokens += ev.cache_write_tokens
        elif isinstance(ev, Stop):
            stop = ev
            saw_stop = True
        elif isinstance(ev, ProviderError):
            raise ProviderErrorRaised(ev)
        else:  # pragma: no cover - exhaustive by Event union
            raise TypeError(f"unknown event type: {ev!r}")

    blocks: list[Block] = []
    invalid: list[InvalidToolCall] = []
    for item in ordered:
        if isinstance(item, Block):
            blocks.append(item)
            continue
        parsed = _parse_partial_json(item.raw_args)
        if isinstance(parsed, dict):
            blocks.append(
                Block(type="tool_call", tool_call_id=item.id, name=item.name, args=parsed)
            )
        else:
            invalid.append(
                InvalidToolCall(
                    index=item.index,
                    id=item.id,
                    name=item.name,
                    raw_args=item.raw_args,
                    error=_error_for(item.raw_args, parsed),
                )
            )

    if not saw_stop:
        stop = Stop(reason="end_turn", native_reason=None)
    if any(b.type == "tool_call" for b in blocks) and stop.reason != "tool_use":
        stop = Stop(reason="tool_use", native_reason=stop.native_reason)

    message = Message(role="assistant", blocks=blocks)
    return ModelResponse(
        message=message, invalid_tool_calls=invalid, usage=usage, stop=stop, model=model, role=role
    )


def _merge_text(ordered: list[Block | _ToolAccum], ev: TextDelta) -> None:
    if ordered and isinstance(ordered[-1], Block) and ordered[-1].type == "text":
        target = ordered[-1]
        target.text = (target.text or "") + ev.text
    else:
        ordered.append(Block(type="text", text=ev.text))


def _merge_thinking(ordered: list[Block | _ToolAccum], ev: ThinkingDelta) -> None:
    if ordered and isinstance(ordered[-1], Block) and ordered[-1].type == "thinking":
        target = ordered[-1]
    else:
        target = Block(type="thinking", text="")
        ordered.append(target)
    if ev.text:
        target.text = (target.text or "") + ev.text
    if ev.native is not None:
        target.native = {**(target.native or {}), **ev.native}


# --------------------------------------------------------------------------- partial JSON


def _parse_partial_json(raw: str) -> Any:
    """Try a strict parse, then a best-effort repair of a truncated document.

    Returns the parsed value (of any JSON type) on success, or a sentinel
    `_ParseFailure` on failure so the caller can report why.
    """
    text = raw.strip()
    if text == "":
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError as first_exc:
        repaired = _repair(text)
        if repaired is not None:
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                pass
        return _ParseFailure(str(first_exc))


@dataclass
class _ParseFailure:
    reason: str


def _repair(text: str) -> str | None:
    """Close unterminated strings, arrays and objects. Best-effort only:
    a value missing after a trailing `:` or `,` is not synthesised."""
    stack: list[str] = []
    in_string = False
    escaped = False
    for ch in text:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]" and stack:
            stack.pop()
    closing = []
    if in_string:
        closing.append('"')
    for opener in reversed(stack):
        closing.append("}" if opener == "{" else "]")
    if not closing and not in_string:
        return None  # nothing to repair; the original parse failure stands
    return text + "".join(closing)


def _error_for(raw: str, parsed: Any) -> str:
    if isinstance(parsed, _ParseFailure):
        return f"invalid JSON in tool call arguments: {parsed.reason}"
    return f"tool call arguments must be a JSON object, got {type(parsed).__name__}"
