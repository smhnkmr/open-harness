"""glob tool. Spec: open-harness-spec.md section 6.3.

Wraps the stdlib `glob` module (which gives pathlib-rglob-like `**`
semantics via recursive=True). Results are capped, sorted by mtime
descending, and reported relative to cwd.
"""

from __future__ import annotations

import glob as glob_module
import os
from pathlib import Path
from typing import Any

from open_harness.tools.base import Tool, ToolContext, ToolResult, ValidationResult

MAX_RESULTS = 100


class GlobTool(Tool):
    name = "glob"
    description = "Find files matching a glob pattern, most recently modified first."
    search_hint = "glob find files pattern wildcard filename"

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "e.g. '**/*.py'"},
                "path": {"type": "string", "description": "Base directory; defaults to cwd."},
            },
            "required": ["pattern"],
        }

    def is_read_only(self, args: dict[str, Any]) -> bool:
        return True

    def permission_content(self, args: dict[str, Any]) -> str:
        return str(args.get("pattern", ""))

    def validate(self, args: dict[str, Any], ctx: ToolContext) -> ValidationResult:
        if not args.get("pattern"):
            return ValidationResult(ok=False, message="pattern is required")
        return ValidationResult(ok=True)

    def call(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        pattern = str(args["pattern"])
        raw_path = args.get("path") or "."
        base = Path(raw_path)
        base = base if base.is_absolute() else (ctx.cwd / base)
        try:
            base = base.resolve()
        except OSError:
            return ToolResult(content=f"Error: cannot resolve path: {raw_path}", is_error=True)
        if not base.exists():
            return ToolResult(content=f"Error: path not found: {base}", is_error=True)

        full_pattern = str(base / pattern)
        try:
            raw_matches = glob_module.glob(full_pattern, recursive=True)
        except (OSError, ValueError) as exc:
            return ToolResult(content=f"Error: invalid pattern: {exc}", is_error=True)

        matches = [Path(p) for p in raw_matches if Path(p).is_file()]
        matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        truncated = len(matches) > MAX_RESULTS
        matches = matches[:MAX_RESULTS]
        rels = [os.path.relpath(p, ctx.cwd) for p in matches]

        if not rels:
            return ToolResult(content=f"({self.name} completed with no output)")

        content = "\n".join(rels)
        if truncated:
            content += f"\n... [truncated to {MAX_RESULTS} most recently modified results]"
        return ToolResult(content=content, data={"truncated": truncated, "count": len(rels)})
