"""Parse an open-harness session log (`<session_root>/<uuid>.jsonl`) into
per-turn metrics.

The log is append-only (`open_harness/kernel/log.py`): resuming a session for
turn 2 of a task appends more records to the *same* file rather than starting
a new one, so a naive whole-file parse would double-count turn 1's usage,
tool calls, etc. when a driver calls this again after turn 2 finishes.

Instead we split the log into per-turn segments -- each segment starts at a
`user_message` record (one per `session.run_turn()` / CLI invocation) and
runs up to and including the next `turn_end` record (which carries that
call's own loop-turn count, not a cumulative one; see the loop's `turn_start
{"turn": n}` numbering, which also restarts at 1 for every resumed call).
`turn_index` selects which segment (0 = first prompt, 1 = the first resume,
...); `None` selects the most recently completed segment, or the trailing
incomplete one if the session ended mid-turn (crash).

Malformed lines (not JSON, or JSON that is not an object) are skipped.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from evals.parse_cc import _load_lines

if TYPE_CHECKING:
    from evals.runners import ParsedTurn


def _read_records(path: Path) -> list[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    return _load_lines(text.splitlines())


def _segments(records: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    segments: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] | None = None
    for rec in records:
        kind = rec.get("kind")
        if kind == "user_message":
            if current is not None and all(r.get("kind") == "user_message" for r in current):
                # The loop pushes project instructions (AGENTS.md) as a
                # harness-tagged user message right before the first real
                # prompt; both belong to the same turn.
                current.append(rec)
                continue
            if current is not None:
                # Previous segment never saw a turn_end (crashed mid-turn);
                # keep it so callers can still inspect what happened.
                segments.append(current)
            current = [rec]
            continue
        if current is None:
            # Records before the first user_message (e.g. session_start)
            # don't belong to any prompt turn.
            continue
        current.append(rec)
        if kind == "turn_end":
            segments.append(current)
            current = None
    if current is not None:
        segments.append(current)
    return segments


def parse_oh_log(path: Path, *, turn_index: int | None = None) -> ParsedTurn:
    from evals.runners import ParsedTurn
    from evals.types import TokenUsage

    path = Path(path)
    records = _read_records(path)
    segments = _segments(records)

    if not segments:
        return ParsedTurn(is_error=True, error_text=f"no turn records found in {path}")

    if turn_index is not None:
        if turn_index < 0 or turn_index >= len(segments):
            return ParsedTurn(
                is_error=True,
                error_text=f"no records for turn index {turn_index} in {path} "
                f"(found {len(segments)} turn(s))",
            )
        segment = segments[turn_index]
    else:
        segment = segments[-1]

    tokens = TokenUsage()
    usage_by_model: dict[str, TokenUsage] = {}
    api_calls = 0
    tool_calls = 0
    denials = 0
    verifier_failures = 0
    turns = 0
    last_assistant_rec: dict[str, Any] | None = None
    turn_end_rec: dict[str, Any] | None = None
    last_provider_error: dict[str, Any] | None = None

    for rec in segment:
        kind = rec.get("kind")
        if kind == "usage":
            api_calls += 1
            tokens.input += int(rec.get("input", 0) or 0)
            tokens.output += int(rec.get("output", 0) or 0)
            tokens.cache_read += int(rec.get("cache_read", 0) or 0)
            tokens.cache_write += int(rec.get("cache_write", 0) or 0)
            spec = str(rec.get("spec") or "unknown")
            per_model = usage_by_model.setdefault(spec.split(":", 1)[-1], TokenUsage())
            per_model.input += int(rec.get("input", 0) or 0)
            per_model.output += int(rec.get("output", 0) or 0)
            per_model.cache_read += int(rec.get("cache_read", 0) or 0)
            per_model.cache_write += int(rec.get("cache_write", 0) or 0)
        elif kind == "tool_call_start":
            tool_calls += 1
        elif kind == "permission_decision":
            if rec.get("behavior") == "deny":
                denials += 1
        elif kind == "verifier_result":
            if rec.get("ok") is False:
                verifier_failures += 1
        elif kind == "provider_error":
            last_provider_error = rec
        elif kind == "assistant_message":
            last_assistant_rec = rec
        elif kind == "turn_end":
            turn_end_rec = rec
            turns += int(rec.get("turns", 0) or 0)

    final_text = ""
    if last_assistant_rec is not None:
        blocks = (last_assistant_rec.get("message") or {}).get("blocks") or []
        final_text = "".join(
            b.get("text", "") for b in blocks if isinstance(b, dict) and b.get("type") == "text"
        )

    if turn_end_rec is not None:
        reason = turn_end_rec.get("reason", "")
        is_error = reason != "completed"
        error_text = "" if not is_error else f"{reason}: {turn_end_rec.get('detail', '')}".strip(": ")
    else:
        is_error = True
        if last_provider_error is not None:
            error_text = f"incomplete turn (no turn_end): {last_provider_error.get('message', '')}"
        else:
            error_text = "incomplete turn (no turn_end record found)"

    return ParsedTurn(
        api_calls=api_calls,
        tool_calls=tool_calls,
        tokens=tokens,
        usage_by_model=usage_by_model,
        reported_cost_usd=None,  # open-harness does not report a cost itself
        denials=denials,
        verifier_failures=verifier_failures,
        turns=turns,
        final_text=final_text,
        is_error=is_error,
        error_text=error_text,
    )
