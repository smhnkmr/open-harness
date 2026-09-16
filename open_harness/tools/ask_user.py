"""ask_user tool. Spec: open-harness-spec.md section 6.2.

Delegates to ctx.ask_user, the surface-provided callback. Not read-only and
not concurrency-safe: it blocks on a human and should never be batched with
another ask_user call.
"""

from __future__ import annotations

from typing import Any

from open_harness.tools.base import Tool, ToolContext, ToolResult, ValidationResult


class AskUserTool(Tool):
    name = "ask_user"
    description = "Ask the user a clarifying question, optionally offering suggested options."
    search_hint = "ask user question clarify confirm human input"

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "options": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["question"],
        }

    def is_read_only(self, args: dict[str, Any]) -> bool:
        return False

    def is_concurrency_safe(self, args: dict[str, Any]) -> bool:
        return False

    def permission_content(self, args: dict[str, Any]) -> str:
        return str(args.get("question", ""))

    def validate(self, args: dict[str, Any], ctx: ToolContext) -> ValidationResult:
        if not args.get("question"):
            return ValidationResult(ok=False, message="question is required")
        return ValidationResult(ok=True)

    def call(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        if ctx.ask_user is None:
            return ToolResult(content="Error: no user available to ask.", is_error=True)
        question = str(args["question"])
        options = [str(o) for o in args.get("options", [])]
        answer = ctx.ask_user(question, options)
        return ToolResult(content=answer)
