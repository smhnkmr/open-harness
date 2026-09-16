"""shell tool. Spec: open-harness-spec.md section 6.3.

Runs a command through ctx.backend.execute. `is_read_only` uses a
conservative allowlist heuristic over each `&&`/`||`/`;`/`|`-separated
segment -- unknown commands, redirection, and any write-ish token make the
whole command non-read-only.
"""

from __future__ import annotations

import re
import shlex
from typing import Any

from open_harness.tools.base import Tool, ToolContext, ToolResult, ValidationResult

DEFAULT_TIMEOUT_MS = 120_000
MAX_TIMEOUT_MS = 600_000
OUTPUT_CAP = 30_000
HEAD_CAP = 20_000
TAIL_CAP = 10_000

_SEGMENT_SPLIT_RE = re.compile(r"&&|\|\||;|\|")

_READ_ONLY_COMMANDS = {
    "ls", "dir", "cat", "head", "tail", "grep", "rg", "find", "echo", "pwd",
    "which", "type", "wc", "sort", "uniq", "diff", "file", "stat",
    "printenv", "env", "whoami", "date", "true", "false", "test", "cd",
    "export",  # only sets a variable in the child shell
}

# `NAME=value` words before the program name (`PYTHONPATH=src python -m x`).
# Values with `$`, backticks or quotes are left alone so the safety check
# still sees them.
_ENV_PREFIX_RE = re.compile(r"^(?:[A-Za-z_][A-Za-z0-9_]*=[^\s\"'`$]*\s+)+")


def strip_env_prefix(segment: str) -> str:
    """Drop leading `NAME=value` assignments so rules and the read-only
    check see the program that actually runs. The eval matrix found Sonnet
    writing `PYTHONPATH=src <python> -m pytest` and being denied because
    no rule starts with `PYTHONPATH=`."""
    return _ENV_PREFIX_RE.sub("", segment.lstrip())

_READ_ONLY_GIT_SUBCOMMANDS = {"status", "log", "diff", "show", "branch"}

# Tokens that always make a segment non-read-only, even if they could
# theoretically appear read-only in isolation (kept explicit per spec 6.3).
_WRITE_TOKENS = {
    "rm", "mv", "cp", "curl", "wget", "pip", "pip3", "npm", "npx", "yarn",
    "pnpm", "make", "touch", "mkdir", "rmdir", "chmod", "chown", "sed",
    "tee", "dd", "kill", "shutdown", "reboot", "install",
}


def _split_segments(command: str) -> list[str]:
    return [seg.strip() for seg in _SEGMENT_SPLIT_RE.split(command) if seg.strip()]


def _segment_is_read_only(segment: str) -> bool:
    # `2>&1` and `2>/dev/null` only reroute stderr; they write nothing.
    stripped = re.sub(r"\d?>&\d|\d>\s*/dev/null|2>\s*NUL\b", "", segment)
    if ">" in stripped:  # covers both > and >>
        return False
    try:
        tokens = shlex.split(strip_env_prefix(segment))
    except ValueError:
        return False
    if not tokens:
        return True
    head = tokens[0]
    if head in _WRITE_TOKENS:
        return False
    if head == "git":
        sub = tokens[1] if len(tokens) > 1 else ""
        return sub in _READ_ONLY_GIT_SUBCOMMANDS
    return head in _READ_ONLY_COMMANDS


def command_is_read_only(command: str) -> bool:
    segments = _split_segments(command)
    if not segments:
        return True
    return all(_segment_is_read_only(seg) for seg in segments)


def _cap_output(text: str) -> tuple[str, bool]:
    if len(text) <= OUTPUT_CAP:
        return text, False
    omitted = len(text) - HEAD_CAP - TAIL_CAP
    marker = f"\n... [{omitted} chars omitted] ...\n"
    return text[:HEAD_CAP] + marker + text[-TAIL_CAP:], True


class ShellTool(Tool):
    name = "shell"
    description = "Run a shell command in the working directory and return its output."
    search_hint = "shell run command execute bash terminal exec"

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "timeout_ms": {
                    "type": "integer",
                    "default": DEFAULT_TIMEOUT_MS,
                    "maximum": MAX_TIMEOUT_MS,
                },
                "description": {"type": "string", "description": "Short human-readable purpose."},
            },
            "required": ["command"],
        }

    def is_read_only(self, args: dict[str, Any]) -> bool:
        return command_is_read_only(str(args.get("command", "")))

    def is_destructive(self, args: dict[str, Any]) -> bool:
        return not self.is_read_only(args)

    def permission_content(self, args: dict[str, Any]) -> str:
        return str(args.get("command", ""))

    def validate(self, args: dict[str, Any], ctx: ToolContext) -> ValidationResult:
        if not args.get("command"):
            return ValidationResult(ok=False, message="command is required")
        timeout_ms = args.get("timeout_ms")
        if timeout_ms is not None and int(timeout_ms) > MAX_TIMEOUT_MS:
            return ValidationResult(ok=False, message=f"timeout_ms may not exceed {MAX_TIMEOUT_MS}")
        return ValidationResult(ok=True)

    def call(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        command = str(args["command"])
        timeout_ms = int(args.get("timeout_ms") or DEFAULT_TIMEOUT_MS)
        timeout_ms = min(timeout_ms, MAX_TIMEOUT_MS)

        # TODO(P1): only the main thread may persist a `cd` across shell
        # calls. The prototype does not track/apply cwd changes at all --
        # every call runs with ctx.cwd regardless of a leading `cd ... &&`.
        result = ctx.backend.execute(command, timeout=timeout_ms / 1000, cwd=ctx.cwd)

        combined = result.stdout
        if result.stderr:
            combined = f"{combined}\n[stderr]\n{result.stderr}" if combined else f"[stderr]\n{result.stderr}"
        combined, _truncated = _cap_output(combined)
        if result.timed_out:
            note = f"[command timed out after {timeout_ms}ms and was killed]"
            combined = f"{combined}\n{note}" if combined else note

        return ToolResult(
            content=combined,
            is_error=(result.exit_code != 0),
            data={"exit_code": result.exit_code, "timed_out": result.timed_out},
        )
