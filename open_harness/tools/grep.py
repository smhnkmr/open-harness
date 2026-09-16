"""grep tool. Spec: open-harness-spec.md section 6.3.

Uses ripgrep when it's on PATH, else a conservative Python fallback that
walks the tree, skipping .git/node_modules and files that look binary.
"""

from __future__ import annotations

import fnmatch
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from open_harness.tools.base import Tool, ToolContext, ToolResult, ValidationResult

_SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".hg", ".svn"}
_BINARY_SKIP_EXTS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".pdf", ".zip", ".gz",
    ".tar", ".7z", ".exe", ".dll", ".so", ".dylib", ".pyc", ".class",
    ".woff", ".woff2", ".ttf", ".eot", ".mp3", ".mp4", ".mov", ".avi",
}
_RG_TIMEOUT_S = 30


def _is_probably_binary(path: Path) -> bool:
    if path.suffix.lower() in _BINARY_SKIP_EXTS:
        return True
    try:
        with open(path, "rb") as f:
            chunk = f.read(1024)
    except OSError:
        return True
    return b"\x00" in chunk


def _iter_candidate_files(base: Path):
    if base.is_file():
        yield base
        return
    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
        for fname in files:
            yield Path(root) / fname


def _relativize(line: str, abs_base: Path, cwd: Path) -> str:
    prefix = str(abs_base)
    if line.startswith(prefix):
        rel = os.path.relpath(abs_base, cwd)
        return rel + line[len(prefix):]
    return line


def _run_ripgrep(
    pattern: str,
    abs_base: Path,
    cwd: Path,
    output_mode: str,
    ignore_case: bool,
    context: int | None,
    glob_pat: str | None,
    multiline: bool,
) -> list[str]:
    cmd = ["rg", "--no-heading", "--color=never"]
    if output_mode == "files_with_matches":
        cmd.append("-l")
    elif output_mode == "count":
        cmd.append("-c")
    else:
        cmd.append("-n")
        if context:
            cmd += ["-C", str(int(context))]
    if ignore_case:
        cmd.append("-i")
    if multiline:
        cmd += ["-U", "--multiline-dotall"]
    if glob_pat:
        cmd += ["--glob", glob_pat]
    cmd += [pattern, str(abs_base)]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=_RG_TIMEOUT_S, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if proc.returncode not in (0, 1):
        return []
    lines = [ln for ln in proc.stdout.splitlines() if ln and ln != "--"]
    return [_relativize(ln, abs_base, cwd) for ln in lines]


def _python_grep(
    pattern: str,
    abs_base: Path,
    cwd: Path,
    output_mode: str,
    ignore_case: bool,
    context: int | None,
    glob_pat: str | None,
    multiline: bool,
) -> list[str]:
    flags = re.IGNORECASE if ignore_case else 0
    if multiline:
        flags |= re.MULTILINE | re.DOTALL
    try:
        regex = re.compile(pattern, flags)
    except re.error as exc:
        return [f"Error: invalid pattern: {exc}"]

    results: list[str] = []
    for path in sorted(_iter_candidate_files(abs_base)):
        if glob_pat and not fnmatch.fnmatch(path.name, glob_pat):
            continue
        if _is_probably_binary(path):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        rel = os.path.relpath(path, cwd)
        lines = text.splitlines()

        if multiline:
            if not regex.search(text):
                continue
            match_lines = {text.count("\n", 0, m.start()) + 1 for m in regex.finditer(text)}
        else:
            match_lines = {i + 1 for i, line in enumerate(lines) if regex.search(line)}
            if not match_lines:
                continue

        if output_mode == "files_with_matches":
            results.append(rel)
            continue
        if output_mode == "count":
            results.append(f"{rel}:{len(match_lines)}")
            continue

        shown = match_lines
        if context:
            shown = set()
            for ln in match_lines:
                for n in range(max(1, ln - context), min(len(lines), ln + context) + 1):
                    shown.add(n)
        for ln in sorted(shown):
            results.append(f"{rel}:{ln}:{lines[ln - 1]}")

    return results


class GrepTool(Tool):
    name = "grep"
    description = "Search file contents for a regex pattern (ripgrep-backed)."
    search_hint = "grep search find text pattern regex content ripgrep"

    def __init__(self, force_python_fallback: bool = False) -> None:
        self._force_python_fallback = force_python_fallback

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string", "description": "Defaults to cwd."},
                "glob": {"type": "string"},
                "output_mode": {
                    "type": "string",
                    "enum": ["content", "files_with_matches", "count"],
                    "default": "files_with_matches",
                },
                "-i": {"type": "boolean", "description": "Case-insensitive."},
                "-C": {"type": "integer", "description": "Lines of context (content mode only)."},
                "head_limit": {"type": "integer", "default": 250},
                "multiline": {"type": "boolean", "default": False},
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

    def _use_ripgrep(self) -> bool:
        return not self._force_python_fallback and shutil.which("rg") is not None

    def call(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        pattern = str(args["pattern"])
        raw_path = args.get("path") or "."
        base = Path(raw_path)
        abs_base = base if base.is_absolute() else (ctx.cwd / base)
        try:
            abs_base = abs_base.resolve()
        except OSError:
            return ToolResult(content=f"Error: cannot resolve path: {raw_path}", is_error=True)
        if not abs_base.exists():
            return ToolResult(content=f"Error: path not found: {abs_base}", is_error=True)

        output_mode = args.get("output_mode", "files_with_matches")
        ignore_case = bool(args.get("-i", False))
        context = args.get("-C")
        context = int(context) if context is not None else None
        head_limit = int(args.get("head_limit") or 250)
        multiline = bool(args.get("multiline", False))
        glob_pat = args.get("glob")

        if self._use_ripgrep():
            lines = _run_ripgrep(pattern, abs_base, ctx.cwd, output_mode, ignore_case, context, glob_pat, multiline)
        else:
            lines = _python_grep(pattern, abs_base, ctx.cwd, output_mode, ignore_case, context, glob_pat, multiline)

        truncated = len(lines) > head_limit
        shown = lines[:head_limit]
        content = "\n".join(shown)
        if truncated:
            content += f"\n... [showing {head_limit} of {len(lines)} results]"

        if not content:
            return ToolResult(content=f"({self.name} completed with no output)")
        return ToolResult(content=content, data={"truncated": truncated, "count": len(shown)})
