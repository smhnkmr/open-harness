"""Parse a Claude Code `--output-format stream-json` transcript.

Claude Code emits one JSON object per stdout line. `assistant` records carry
one *content block* each (`message.content` has exactly one entry) but
several such records share the same `message.id` when the model emitted
several blocks (thinking / text / tool_use) for a single API response, so the
number of API calls is the number of distinct `message.id`s seen, not the
number of `assistant` records. The `result` record at the end of the stream
carries the harness-reported cost, token totals (per model, in `modelUsage`,
which also captures Haiku sub-calls such as the compactor), turn count,
permission denials and the final answer text.

Malformed lines (not JSON, or JSON that is not an object) are skipped.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from evals.runners import ParsedTurn


def _load_lines(lines: Iterable[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for raw in lines:
        raw = raw.strip()
        if not raw:
            continue
        try:
            rec = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict):
            records.append(rec)
    return records


def parse_cc_stream(lines: Iterable[str]) -> ParsedTurn:
    """Parse the stdout lines of one `claude -p ... --output-format stream-json`
    invocation (one turn) into a `ParsedTurn`."""
    from evals.runners import ParsedTurn
    from evals.types import TokenUsage

    message_ids: set[str] = set()
    tool_calls = 0
    result_rec: dict[str, Any] | None = None

    for rec in _load_lines(lines):
        kind = rec.get("type")
        if kind == "assistant":
            message = rec.get("message") or {}
            mid = message.get("id")
            if mid:
                message_ids.add(mid)
            for block in message.get("content") or []:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    tool_calls += 1
        elif kind == "result":
            result_rec = rec

    tokens = TokenUsage()
    usage_by_model: dict[str, TokenUsage] = {}
    if result_rec is not None:
        model_usage = result_rec.get("modelUsage") or {}
        for model_name, usage in model_usage.items():
            per_model = TokenUsage(
                input=int(usage.get("inputTokens", 0) or 0),
                output=int(usage.get("outputTokens", 0) or 0),
                cache_read=int(usage.get("cacheReadInputTokens", 0) or 0),
                cache_write=int(usage.get("cacheCreationInputTokens", 0) or 0),
            )
            usage_by_model[model_name] = per_model
            tokens.input += per_model.input
            tokens.output += per_model.output
            tokens.cache_read += per_model.cache_read
            tokens.cache_write += per_model.cache_write

        reported_cost_usd = result_rec.get("total_cost_usd")
        denials = len(result_rec.get("permission_denials") or [])
        turns = int(result_rec.get("num_turns", 0) or 0)
        final_text = result_rec.get("result") or ""
        is_error = bool(result_rec.get("is_error", False))
        error_text = final_text if is_error else ""
    else:
        reported_cost_usd = None
        denials = 0
        turns = 0
        final_text = ""
        is_error = True
        error_text = "no result record found in stream"

    return ParsedTurn(
        api_calls=len(message_ids),
        tool_calls=tool_calls,
        tokens=tokens,
        usage_by_model=usage_by_model,
        reported_cost_usd=reported_cost_usd,
        denials=denials,
        verifier_failures=0,  # Claude Code has no verifier gate
        turns=turns,
        final_text=final_text,
        is_error=is_error,
        error_text=error_text,
    )
