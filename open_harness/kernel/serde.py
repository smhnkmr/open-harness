"""Serialisation between `Message`/`Block` dataclasses and JSON-safe dicts.

Spec: open-harness-spec.md section 5.1 (Block/Message shapes). Used by
`kernel.log` to persist and replay messages, and by any client that needs to
round-trip a `Message` through JSON.
"""

from __future__ import annotations

from typing import Any

from open_harness.model.types import Block, Message


def serialize_block(block: Block) -> dict[str, Any]:
    """Flatten a `Block` to a JSON-safe dict, recursing into nested content."""
    data: dict[str, Any] = {"type": block.type}
    if block.text is not None:
        data["text"] = block.text
    if block.tool_call_id is not None:
        data["tool_call_id"] = block.tool_call_id
    if block.name is not None:
        data["name"] = block.name
    if block.args is not None:
        data["args"] = block.args
    if block.is_error:
        data["is_error"] = block.is_error
    if block.content is not None:
        data["content"] = [serialize_block(b) for b in block.content]
    if block.media_type is not None:
        data["media_type"] = block.media_type
    if block.data is not None:
        data["data"] = block.data
    if block.path is not None:
        data["path"] = block.path
    if block.native is not None:
        data["native"] = block.native
    if block.needs_server_state:
        data["needs_server_state"] = block.needs_server_state
    if block.cache_breakpoint:
        data["cache_breakpoint"] = block.cache_breakpoint
    return data


def deserialize_block(data: dict[str, Any]) -> Block:
    """Inverse of `serialize_block`."""
    content = data.get("content")
    return Block(
        type=data["type"],
        text=data.get("text"),
        tool_call_id=data.get("tool_call_id"),
        name=data.get("name"),
        args=data.get("args"),
        is_error=data.get("is_error", False),
        content=[deserialize_block(b) for b in content] if content is not None else None,
        media_type=data.get("media_type"),
        data=data.get("data"),
        path=data.get("path"),
        native=data.get("native"),
        needs_server_state=data.get("needs_server_state", False),
        cache_breakpoint=data.get("cache_breakpoint", False),
    )


def serialize_message(message: Message) -> dict[str, Any]:
    return {
        "role": message.role,
        "blocks": [serialize_block(b) for b in message.blocks],
        "meta": message.meta,
    }


def deserialize_message(data: dict[str, Any]) -> Message:
    return Message(
        role=data["role"],
        blocks=[deserialize_block(b) for b in data.get("blocks", [])],
        meta=dict(data.get("meta") or {}),
    )
