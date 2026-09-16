"""read tool. Spec: open-harness-spec.md section 6.3.

Caps by file size before reading and by (approximate) tokens after. Repeated
reads of the same path/offset/limit with an unchanged mtime return a stub
instead of the content. Binary files error, except recognised image
extensions which are returned as an image block.
"""

from __future__ import annotations

import base64
import hashlib
from pathlib import Path
from typing import Any

from open_harness.model.types import Block
from open_harness.tools.base import ReadRecord, Tool, ToolContext, ToolResult, ValidationResult

_IMAGE_MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


class ReadTool(Tool):
    name = "read"
    description = "Read a file from the filesystem, optionally a line range."
    search_hint = "read file view cat open contents show"
    max_result_chars = float("inf")

    MAX_FILE_BYTES = 256 * 1024
    # ~25,000 tokens, approximated as chars/4.
    MAX_OUTPUT_CHARS = 25_000 * 4

    def __init__(self) -> None:
        # (resolved_path, offset, limit) -> (mtime_ns, size) of the last read
        # with that exact range. Used only to detect "unchanged since last
        # read"; the read-before-write invariant is enforced separately via
        # ctx.read_state (the contract-shaped record).
        self._last_reads: dict[tuple[str, int | None, int | None], tuple[int, int]] = {}

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute path, or a path relative to cwd.",
                },
                "offset": {"type": "integer", "description": "1-based line number to start from."},
                "limit": {"type": "integer", "description": "Maximum number of lines to read."},
            },
            "required": ["file_path"],
        }

    def is_read_only(self, args: dict[str, Any]) -> bool:
        return True

    def permission_content(self, args: dict[str, Any]) -> str:
        return str(args.get("file_path", ""))

    def validate(self, args: dict[str, Any], ctx: ToolContext) -> ValidationResult:
        if not args.get("file_path"):
            return ValidationResult(ok=False, message="file_path is required")
        return ValidationResult(ok=True)

    def call(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        raw_path = str(args["file_path"])
        offset = args.get("offset")
        limit = args.get("limit")
        offset = int(offset) if offset is not None else None
        limit = int(limit) if limit is not None else None

        path = Path(raw_path)
        resolved = path if path.is_absolute() else (ctx.cwd / path)
        try:
            resolved = resolved.resolve()
        except OSError:
            return ToolResult(content=f"Error: cannot resolve path: {raw_path}", is_error=True)

        if not resolved.exists():
            return ToolResult(content=f"Error: file not found: {resolved}", is_error=True)
        if not resolved.is_file():
            return ToolResult(content=f"Error: not a file: {resolved}", is_error=True)

        stat = resolved.stat()
        cache_key = (str(resolved), offset, limit)
        if self._last_reads.get(cache_key) == (stat.st_mtime_ns, stat.st_size):
            return ToolResult(content="(file unchanged since last read)")

        ext = resolved.suffix.lower()
        if ext in _IMAGE_MEDIA_TYPES:
            return self._read_image(resolved, stat, ext, cache_key, ctx)

        if offset is None and limit is None and stat.st_size > self.MAX_FILE_BYTES:
            return ToolResult(
                content=(
                    f"Error: {resolved} is {stat.st_size} bytes, over the "
                    f"{self.MAX_FILE_BYTES}-byte limit for a full read. "
                    "Pass offset/limit to read a slice instead."
                ),
                is_error=True,
            )

        raw_bytes = resolved.read_bytes()
        try:
            text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return ToolResult(
                content=f"Error: {resolved} is not valid UTF-8 text (binary file).",
                is_error=True,
            )

        lines = text.splitlines()
        start = max(offset - 1, 0) if offset is not None else 0
        if limit is not None:
            end = min(start + limit, len(lines))
        else:
            end = len(lines)
        selected = lines[start:end]

        numbered = [f"{i + start + 1:6d}\t{line}" for i, line in enumerate(selected)]
        output = "\n".join(numbered)

        truncated = False
        if len(output) > self.MAX_OUTPUT_CHARS:
            head = output[: self.MAX_OUTPUT_CHARS]
            cut = head.rfind("\n")
            if cut > 0:
                head = head[:cut]
            output = head + "\n... [output truncated at ~25,000 tokens; use offset/limit to read more]"
            truncated = True

        full_read = offset is None and limit is None
        ctx.read_state[str(resolved)] = ReadRecord(
            mtime_ns=stat.st_mtime_ns,
            size=stat.st_size,
            content_hash=hashlib.sha256(raw_bytes).hexdigest(),
            full_read=full_read,
        )
        self._last_reads[cache_key] = (stat.st_mtime_ns, stat.st_size)

        if not selected:
            return ToolResult(content=f"({self.name} completed with no output)")
        return ToolResult(
            content=output,
            data={"path": str(resolved), "lines": len(selected), "truncated": truncated},
        )

    def _read_image(
        self,
        resolved: Path,
        stat: Any,
        ext: str,
        cache_key: tuple[str, int | None, int | None],
        ctx: ToolContext,
    ) -> ToolResult:
        data = resolved.read_bytes()
        ctx.read_state[str(resolved)] = ReadRecord(
            mtime_ns=stat.st_mtime_ns,
            size=stat.st_size,
            content_hash=hashlib.sha256(data).hexdigest(),
            full_read=True,
        )
        self._last_reads[cache_key] = (stat.st_mtime_ns, stat.st_size)
        block = Block(type="image", media_type=_IMAGE_MEDIA_TYPES[ext], data=base64.b64encode(data).decode("ascii"))
        return ToolResult(content=[block])
