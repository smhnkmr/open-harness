"""Tool schema normalisation and dialect conversion.

Spec: open-harness-spec.md section 5.3 (tool schema normalisation) and 5.2
(`tool_schema_dialect` capability flag). `ToolSpec.json_schema` is always the
canonical OpenAI-shaped parameters schema; adapters convert from that shape
to their own dialect. The kernel and tool authors never target a vendor
dialect directly.
"""

from __future__ import annotations

from typing import Any

from open_harness.model.types import ToolSpec


def normalise_schema(schema: dict[str, Any] | None) -> dict[str, Any]:
    """Ensure a JSON schema has `type: object`, `properties` and `required`.

    Non-destructive: existing keys are preserved, only missing ones are
    filled in with sensible empty defaults.
    """
    out = dict(schema) if schema else {}
    out.setdefault("type", "object")
    out.setdefault("properties", {})
    out.setdefault("required", [])
    return out


def to_openai_tool(spec: ToolSpec) -> dict[str, Any]:
    """Canonical shape is already OpenAI's; just wrap and normalise."""
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": normalise_schema(spec.json_schema),
        },
    }


def to_anthropic_tool(spec: ToolSpec) -> dict[str, Any]:
    return {
        "name": spec.name,
        "description": spec.description,
        "input_schema": normalise_schema(spec.json_schema),
    }
