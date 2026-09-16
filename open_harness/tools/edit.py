"""edit tool. Spec: open-harness-spec.md section 6.3.

Exact string match only (no fuzzy matching, no model repair). Requires the
file to have been read this session with an unchanged mtime since -- same
invariant as write. CRLF files are matched using their own line ending: the
tool normalises old_string/new_string to the file's newline convention
before comparing, rather than normalising the file's content.
"""

from __future__ import annotations

import difflib
import hashlib
from typing import Any

from open_harness.tools._shared import (
    CHANGED_ON_DISK,
    NOT_READ,
    check_read_before_write,
    resolve_path,
)
from open_harness.tools.base import ReadRecord, Tool, ToolContext, ToolResult, ValidationResult

MAX_DIFF_LINES = 40


def _to_newline(s: str, newline: str) -> str:
    normalized = s.replace("\r\n", "\n")
    return normalized if newline == "\n" else normalized.replace("\n", newline)


class EditTool(Tool):
    name = "edit"
    description = "Replace an exact string occurrence in a file with another string."
    search_hint = "edit replace modify change patch file"

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Absolute path, or relative to cwd."},
                "old_string": {"type": "string"},
                "new_string": {"type": "string"},
                "replace_all": {"type": "boolean", "default": False},
            },
            "required": ["file_path", "old_string", "new_string"],
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
        old_string = args.get("old_string")
        if not old_string:
            return ValidationResult(ok=False, message="old_string is required and must be non-empty")
        if old_string == args.get("new_string"):
            return ValidationResult(ok=False, message="old_string and new_string must differ")
        return ValidationResult(ok=True)

    def call(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        raw_path = str(args["file_path"])
        old_string = str(args["old_string"])
        new_string = str(args["new_string"])
        replace_all = bool(args.get("replace_all", False))
        resolved = resolve_path(raw_path, ctx.cwd)

        if not resolved.exists():
            return ToolResult(content=f"Error [NOT_FOUND]: file not found: {resolved}", is_error=True)
        if not resolved.is_file():
            return ToolResult(content=f"Error [NOT_FOUND]: not a file: {resolved}", is_error=True)

        code = check_read_before_write(resolved, ctx)
        if code == NOT_READ:
            return ToolResult(
                content=f"Error [NOT_READ]: read the file first before editing it: {resolved}",
                is_error=True,
            )
        if code == CHANGED_ON_DISK:
            return ToolResult(
                content=f"Error [CHANGED_ON_DISK]: file changed on disk since it was last read: {resolved}",
                is_error=True,
            )

        raw_bytes = resolved.read_bytes()
        try:
            content = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return ToolResult(content=f"Error: {resolved} is not valid UTF-8 text.", is_error=True)

        newline = "\r\n" if "\r\n" in content else "\n"
        match_old = _to_newline(old_string, newline)
        match_new = _to_newline(new_string, newline)

        count = content.count(match_old)
        if count == 0:
            return ToolResult(
                content=f"Error [NO_MATCH]: old_string not found in {resolved}",
                is_error=True,
            )
        if count > 1 and not replace_all:
            return ToolResult(
                content=(
                    f"Error [MULTIPLE_MATCHES]: old_string matches {count} locations in {resolved}; "
                    "pass replace_all=true, or supply a larger, unique old_string."
                ),
                is_error=True,
            )

        if replace_all:
            new_content = content.replace(match_old, match_new)
            replacements = count
        else:
            new_content = content.replace(match_old, match_new, 1)
            replacements = 1

        data = new_content.encode("utf-8")
        resolved.write_bytes(data)

        stat = resolved.stat()
        ctx.read_state[str(resolved)] = ReadRecord(
            mtime_ns=stat.st_mtime_ns,
            size=stat.st_size,
            content_hash=hashlib.sha256(data).hexdigest(),
            full_read=True,
        )

        diff_lines = list(
            difflib.unified_diff(
                content.replace(newline, "\n").splitlines(keepends=True),
                new_content.replace(newline, "\n").splitlines(keepends=True),
                fromfile=str(resolved),
                tofile=str(resolved),
                lineterm="",
            )
        )
        truncated = len(diff_lines) > MAX_DIFF_LINES
        if truncated:
            diff_lines = diff_lines[:MAX_DIFF_LINES] + ["... (diff truncated)"]
        diff_text = "\n".join(line.rstrip("\n") for line in diff_lines)

        return ToolResult(
            content=diff_text or f"Replaced {replacements} occurrence(s) in {resolved}",
            data={"path": str(resolved), "replacements": replacements, "diff_truncated": truncated},
        )
