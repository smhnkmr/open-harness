"""write tool. Spec: open-harness-spec.md section 6.3.

Overwriting an existing file requires having read it this session, with an
unchanged mtime since. New files may be written without a prior read.
"""

from __future__ import annotations

import hashlib
from typing import Any

from open_harness.tools._shared import (
    CHANGED_ON_DISK,
    NOT_READ,
    check_read_before_write,
    resolve_path,
)
from open_harness.tools.base import ReadRecord, Tool, ToolContext, ToolResult, ValidationResult


class WriteTool(Tool):
    name = "write"
    description = "Write content to a file, creating it (and parent dirs) or overwriting it."
    search_hint = "write file create save overwrite"

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Absolute path, or relative to cwd."},
                "content": {"type": "string"},
            },
            "required": ["file_path", "content"],
        }

    def is_read_only(self, args: dict[str, Any]) -> bool:
        return False

    def is_destructive(self, args: dict[str, Any]) -> bool:
        return True

    def permission_content(self, args: dict[str, Any]) -> str:
        return str(args.get("file_path", ""))

    def validate(self, args: dict[str, Any], ctx: ToolContext) -> ValidationResult:
        if not args.get("file_path"):
            return ValidationResult(ok=False, message="file_path is required")
        if "content" not in args:
            return ValidationResult(ok=False, message="content is required")
        return ValidationResult(ok=True)

    def call(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        raw_path = str(args["file_path"])
        content = str(args["content"])
        resolved = resolve_path(raw_path, ctx.cwd)

        code = check_read_before_write(resolved, ctx)
        if code == NOT_READ:
            return ToolResult(
                content=f"Error [NOT_READ]: read the file first before writing to it: {resolved}",
                is_error=True,
            )
        if code == CHANGED_ON_DISK:
            return ToolResult(
                content=f"Error [CHANGED_ON_DISK]: file changed on disk since it was last read: {resolved}",
                is_error=True,
            )

        resolved.parent.mkdir(parents=True, exist_ok=True)
        data = content.encode("utf-8")
        resolved.write_bytes(data)

        stat = resolved.stat()
        ctx.read_state[str(resolved)] = ReadRecord(
            mtime_ns=stat.st_mtime_ns,
            size=stat.st_size,
            content_hash=hashlib.sha256(data).hexdigest(),
            full_read=True,
        )
        return ToolResult(
            content=f"Wrote {len(data)} bytes to {resolved}",
            data={"path": str(resolved), "bytes": len(data)},
        )
