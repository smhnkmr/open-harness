"""Result persistence: large tool outputs spill to disk, replaced by a
preview. Kernel-side concern, separate from individual tools.

Spec: open-harness-spec.md section 6.4.
"""

from __future__ import annotations

import re
from pathlib import Path

from open_harness.tools.base import (
    AGGREGATE_MAX_CHARS,
    DEFAULT_MAX_RESULT_CHARS,
    PREVIEW_BYTES,
    Tool,
    ToolResult,
)

_UNSAFE_ID_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _safe_id(tool_call_id: str) -> str:
    safe = _UNSAFE_ID_RE.sub("_", tool_call_id).strip("_.")
    return safe or "result"


def _preview(text: str) -> str:
    """First PREVIEW_BYTES characters, cut back to the last newline so the
    preview doesn't end mid-line."""
    head = text[:PREVIEW_BYTES]
    if len(text) > PREVIEW_BYTES:
        cut = head.rfind("\n")
        if cut > 0:
            head = head[:cut]
    return head


def _write_once(file_path: Path, content: str) -> None:
    """Write `content` to `file_path` only if it doesn't already exist.
    Concurrent/duplicate persist attempts for the same tool_call_id keep
    whatever was written first."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(file_path, "x", encoding="utf-8", newline="") as f:
            f.write(content)
    except FileExistsError:
        pass


def _persist(content: str, tool_call_id: str, session_dir: Path) -> tuple[str, str]:
    """Write `content` to tool-results/<safe id>.txt and build the replacement
    preview content. Returns (new_content, persisted_path)."""
    results_dir = session_dir / "tool-results"
    file_path = results_dir / f"{_safe_id(tool_call_id)}.txt"
    _write_once(file_path, content)

    preview = _preview(content)
    new_content = (
        f'<persisted-output path="{file_path}">\n'
        f"{preview}\n"
        "...</persisted-output>\n"
        f'The full output was too large to show and was saved to "{file_path}". '
        "Use the read tool (with offset/limit) to view more of it."
    )
    return new_content, str(file_path)


def persist_if_large(
    result: ToolResult, tool: Tool, tool_call_id: str, session_dir: Path
) -> ToolResult:
    """If `result.content` is a string over the tool's (or default) result
    cap, spill it to `session_dir/tool-results/` and replace it with a
    preview. List content (which may contain images) is never persisted.
    Empty string content is replaced by a fixed "no output" message."""
    content = result.content

    if content == "":
        return ToolResult(
            content=f"({tool.name} completed with no output)",
            is_error=result.is_error,
            data=result.data,
            persisted_path=result.persisted_path,
        )

    if not isinstance(content, str):
        # list[Block]: may contain images; never persisted (spec 6.4).
        return result

    if tool.max_result_chars == float("inf"):
        # An infinite cap means "never persist" (spec 6.3: read never persists,
        # to avoid a read -> file -> read loop). The tool caps its own output.
        return result
    limit = min(tool.max_result_chars, DEFAULT_MAX_RESULT_CHARS)
    if len(content) <= limit:
        return result

    new_content, persisted_path = _persist(content, tool_call_id, session_dir)
    return ToolResult(
        content=new_content,
        is_error=result.is_error,
        data=result.data,
        persisted_path=persisted_path,
    )


def enforce_aggregate_budget(
    results: list[tuple[Tool, str, ToolResult]], session_dir: Path
) -> list[ToolResult]:
    """Per-message aggregate cap. Given the (already per-result-processed)
    results for one message, force-persist the largest not-yet-persisted
    string results, largest first, until the total is under
    AGGREGATE_MAX_CHARS or nothing more can be persisted. Earlier persist
    decisions (results already on disk) are frozen and not reconsidered."""
    out = [r for _, _, r in results]

    def size(res: ToolResult) -> int:
        return len(res.content) if isinstance(res.content, str) else 0

    total = sum(size(r) for r in out)
    if total <= AGGREGATE_MAX_CHARS:
        return out

    candidates = [
        i
        for i, r in enumerate(out)
        if r.persisted_path is None and isinstance(r.content, str) and r.content != ""
    ]
    candidates.sort(key=lambda i: size(out[i]), reverse=True)

    for i in candidates:
        if total <= AGGREGATE_MAX_CHARS:
            break
        _tool, tool_call_id, _ = results[i]
        before = size(out[i])
        current = out[i]
        assert isinstance(current.content, str)
        new_content, persisted_path = _persist(current.content, tool_call_id, session_dir)
        out[i] = ToolResult(
            content=new_content,
            is_error=current.is_error,
            data=current.data,
            persisted_path=persisted_path,
        )
        total -= before - size(out[i])

    return out
