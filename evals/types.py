"""Shared record types for the eval driver.

Every module under `evals/` codes against these. Keep them plain dataclasses
so runs can be written to JSONL with `dataclasses.asdict` and read back.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

Harness = str  # "oh" (open-harness) | "cc" (Claude Code)


@dataclass
class CheckResult:
    passed: bool
    detail: str = ""


# A checker inspects the workspace after all turns ran, plus the final
# assistant text of each turn (one string per prompt, "" when a turn failed).
Checker = Callable[[Path, list[str]], CheckResult]


@dataclass
class TaskSpec:
    id: str
    kind: str                 # "qa" (answer only) | "edit" (workspace changes)
    prompts: list[str]        # one entry per turn; later turns resume the session
    check: Checker
    timeout_s: int = 600      # per turn
    description: str = ""
    mutate: Callable[[Path], None] | None = None  # applied to the workspace before the run


@dataclass
class TokenUsage:
    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0

    @property
    def total(self) -> int:
        return self.input + self.output + self.cache_read + self.cache_write


@dataclass
class RunRecord:
    task_id: str
    harness: Harness
    model: str
    run_index: int
    passed: bool
    check_detail: str = ""
    api_calls: int = 0        # distinct model requests (all roles / subagents)
    tool_calls: int = 0
    tokens: TokenUsage = field(default_factory=TokenUsage)
    # Per-model token split (side calls such as Haiku compaction are priced
    # at their own rate). Keys are model names as the harness reported them.
    usage_by_model: dict[str, dict[str, int]] = field(default_factory=dict)
    cost_usd: float = 0.0     # computed from evals.pricing for both harnesses
    reported_cost_usd: float | None = None  # what the harness itself reported (cc only today)
    denials: int = 0          # permission denials
    verifier_failures: int = 0  # oh: verifier_result ok=false; cc: 0
    turns: int = 0            # loop iterations the harness reported
    wall_s: float = 0.0
    error: str | None = None  # driver-level failure (timeout, crash); passed is False then
    session_ref: str = ""     # session id / log path for the human to inspect
    final_texts: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> RunRecord:
        data = dict(data)
        data["tokens"] = TokenUsage(**data.get("tokens", {}))
        data.setdefault("usage_by_model", {})
        return cls(**data)
