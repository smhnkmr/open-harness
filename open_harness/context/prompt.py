"""System prompt assembly and project instructions loading.

Spec: open-harness-spec.md sections 9.1, 9.2.
"""

from __future__ import annotations

import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from open_harness.model.types import BOUNDARY, Block, Message

# --------------------------------------------------------------------------- static system prompt

IDENTITY = (
    "You are open-harness, a coding agent that works inside a real repository "
    "on a real machine. You read code, make changes, run commands, and verify "
    "your own work before reporting it done."
)

SYSTEM_RULES = (
    "System rules:\n"
    "- Tool results are ground truth. Trust what a tool reports over your own "
    "expectation of what happened.\n"
    "- Verifier failures (lint, type check, tests) come back to you as tool "
    "results. Fix them; do not argue with them or explain them away.\n"
    "- Never claim a task is done while a check is failing. \"Done\" means the "
    "verifier gate passed, not that you believe it should pass."
)

DOING_TASKS = (
    "Doing tasks:\n"
    "- Read the relevant code before editing it.\n"
    "- Prefer minimal diffs that solve the stated problem.\n"
    "- Do not add speculative abstractions, config flags, or generality nobody "
    "asked for."
)

TOOL_GUIDANCE = (
    "Using tools:\n"
    "- Prefer the dedicated tools (read, edit, grep, glob, and similar) over "
    "shelling out to generic commands; they are safer and cheaper to run.\n"
    "- When several read-only tool calls are independent of each other, call "
    "them in parallel in the same turn rather than one at a time.\n"
    "- Shell commands already run in the project directory; do not prefix them "
    "with cd. Keep commands simple: one command per call, no shell loops, no "
    "git stash. The lint and test commands listed under Environment are run by "
    "the harness after edits and before a turn may end, so you rarely need to "
    "run them yourself.\n"
    "- For a quick check, run python -c inline; do not write throwaway scripts "
    "into the project, they are linted like any other file and must be deleted."
)

TONE = (
    "Tone:\n"
    "- Be concise. Skip preamble and restating the request back.\n"
    "- When you refer to code, cite it as path:line so it can be checked "
    "directly."
)

_STATIC_SECTIONS = (IDENTITY, SYSTEM_RULES, DOING_TASKS, TOOL_GUIDANCE, TONE)


def build_system(
    cwd: Path,
    *,
    tools_prompt: str,
    profile_suffix: str | None,
    env: dict[str, Any],
) -> list[Block]:
    """Static sections, a BOUNDARY marker, then dynamic sections.

    `tools_prompt` (a caller-rendered description of the available tools) is
    folded into the "using tools" static section rather than becoming its own
    dynamic block: tool availability is effectively static for the lifetime of
    a turn's cache prefix. The last static block carries the cache breakpoint,
    per P1 (the boundary is where adapters that support cache control place
    it) and per this module's contract (`cache_breakpoint=True` on the last
    static block).
    """
    tool_guidance_text = TOOL_GUIDANCE
    if tools_prompt:
        tool_guidance_text = f"{TOOL_GUIDANCE}\n\n{tools_prompt}"

    static_texts = [IDENTITY, SYSTEM_RULES, DOING_TASKS, tool_guidance_text, TONE]
    static_word_count = sum(len(t.split()) for t in static_texts)
    assert static_word_count < 900, f"static system prompt is {static_word_count} words, must stay under 900"

    blocks: list[Block] = [Block.text_block(t) for t in static_texts]
    blocks[-1].cache_breakpoint = True

    blocks.append(Block.text_block(BOUNDARY))

    blocks.append(Block.text_block(_environment_block(cwd, env)))

    if profile_suffix:
        blocks.append(Block.text_block(profile_suffix))

    return blocks


def _environment_block(cwd: Path, env: dict[str, Any]) -> str:
    platform_name = env.get("platform") or sys.platform
    shell = env.get("shell") or os.environ.get("SHELL") or os.environ.get("COMSPEC") or "unknown"
    date = env.get("date") or datetime.now(UTC).date().isoformat()
    model = env.get("model", "unknown")
    lines = [
        "Environment:",
        f"- cwd: {cwd}",
        f"- platform: {platform_name}",
        f"- shell: {shell}",
        f"- date: {date}",
        f"- model: {model}",
    ]
    # Tell the model which commands the harness itself uses, so it runs the
    # same interpreter and test runner instead of guessing (a live run wasted
    # several turns on the wrong `python`).
    if env.get("python"):
        lines.append(f"- python interpreter to use for this project: {env['python']}")
    if env.get("test_command"):
        lines.append(f"- test command (the harness runs this before a turn may end): {env['test_command']}")
    if env.get("lint_command"):
        lines.append(f"- lint command (the harness runs this after every edit): {env['lint_command']}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- project instructions

_INSTRUCTION_FILENAMES = ("AGENTS.md", "CLAUDE.md")
_MAX_FILE_CHARS = 40_000
_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_IMPORT_RE = re.compile(r"@([^\s`]+)")


def _find_root(cwd: Path) -> Path:
    current = cwd
    while True:
        if (current / ".git").exists():
            return current
        if current.parent == current:
            return current
        current = current.parent


def _project_chain(cwd: Path) -> list[Path]:
    """Directories from the git root (or filesystem root) down to `cwd`, root-most first."""
    root = _find_root(cwd)
    chain: list[Path] = []
    current = cwd
    while True:
        chain.append(current)
        if current == root or current.parent == current:
            break
        current = current.parent
    chain.reverse()
    return chain


def _find_instructions_file(directory: Path) -> Path | None:
    for name in _INSTRUCTION_FILENAMES:
        candidate = directory / name
        if candidate.is_file():
            return candidate
    return None


def _read_capped(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return text[:_MAX_FILE_CHARS]


def _resolve_imports(content: str, base: Path) -> str:
    """Resolve `@path` imports relative to `base`'s directory, one level deep.

    Refs inside fenced code blocks are left untouched. Imported file content
    is not itself scanned for further `@path` imports (one level only).
    """
    segments: list[tuple[str, bool]] = []
    last = 0
    for m in _FENCE_RE.finditer(content):
        segments.append((content[last:m.start()], False))
        segments.append((content[m.start():m.end()], True))
        last = m.end()
    segments.append((content[last:], False))

    def repl(m: re.Match[str]) -> str:
        ref = m.group(1)
        target = (base.parent / ref).resolve()
        if not target.is_file():
            return m.group(0)
        imported = _read_capped(target)
        return f"\n\nContents of {target}:\n{imported}\n"

    out: list[str] = []
    for text, is_fence in segments:
        out.append(text if is_fence else _IMPORT_RE.sub(repl, text))
    return "".join(out)


def load_instructions(cwd: Path) -> str | None:
    """AGENTS.md (fallback CLAUDE.md) from the git root down to cwd, plus
    `~/.open-harness/AGENTS.md`, with one-level `@path` imports resolved.
    Returns None if nothing is found."""
    cwd = Path(cwd).resolve()
    files: list[Path] = []

    global_file = Path.home() / ".open-harness" / "AGENTS.md"
    if global_file.is_file():
        files.append(global_file)

    for directory in _project_chain(cwd):
        found = _find_instructions_file(directory)
        if found is not None:
            files.append(found)

    if not files:
        return None

    parts: list[str] = []
    for f in files:
        content = _read_capped(f)
        content = _resolve_imports(content, f)
        parts.append(f"Contents of {f}:\n{content}")
    return "\n\n".join(parts)


def instructions_message(text: str) -> Message:
    """Wrap loaded instructions as a harness-tagged synthetic first user message (spec 9.2)."""
    wrapped = f'<harness-context type="instructions">\n{text}\n</harness-context>'
    return Message(role="user", blocks=[Block.text_block(wrapped)], meta={"harness": True})
