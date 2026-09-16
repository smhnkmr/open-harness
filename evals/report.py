"""Markdown report generation for eval runs.

Spec: SPEC.md section 18.2 step 1 and 18.3 (the comparison record this
report is modeled on). Pure functions over `RunRecord` lists so the driver
(and tests) can call `summarize` without touching disk, and `load_records`
to read a `runs.jsonl` file back in.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from evals.types import RunRecord

HARNESS_ORDER = ["oh", "cc"]
HARNESS_NAMES = {"oh": "open-harness", "cc": "Claude Code"}


def load_records(path: Path) -> list[RunRecord]:
    """Read a `runs.jsonl` file (one `RunRecord.to_json()` per line) back in.

    Missing file or blank lines are tolerated; returns [] for a missing file.
    """
    path = Path(path)
    if not path.exists():
        return []
    records: list[RunRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(RunRecord.from_json(json.loads(line)))
    return records


def _harness_sort_key(h: str) -> tuple[int, str]:
    if h in HARNESS_ORDER:
        return (HARNESS_ORDER.index(h), h)
    return (len(HARNESS_ORDER), h)


def _mean(values: list[float | int]) -> float | None:
    values = [v for v in values if v is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _fmt_num(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _fmt_money(value: float | None) -> str:
    return "n/a" if value is None else f"${value:.4f}"


def _fmt_pass_rate(passed: int, total: int) -> str:
    return f"{passed}/{total}" if total else "0/0"


def _fmt_delta_money(value: float | None) -> str:
    if value is None:
        return "n/a"
    sign = "+" if value >= 0 else "-"
    return f"{sign}${abs(value):.4f}"


def _fmt_delta_num(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    sign = "+" if value >= 0 else "-"
    return f"{sign}{abs(value):.{digits}f}"


def _truncate(s: str | None, n: int = 200) -> str:
    s = s or ""
    return s if len(s) <= n else s[: n - 3] + "..."


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def summarize(records: list[RunRecord], *, model: str = "") -> str:
    """Build the markdown eval report for a list of `RunRecord`s.

    Records with `error` set are excluded from the "mean" columns (they
    have no meaningful metrics) but are still counted in pass-rate
    denominators, total denials/verifier-failures, and the error count.
    """
    lines: list[str] = ["# Eval report", ""]
    lines.append(f"- Model: {model or 'n/a'}")
    generated = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")

    if not records:
        lines.append("- Runs recorded: 0")
        lines.append(f"- Generated: {generated}")
        lines.append("")
        lines.append("No runs recorded yet.")
        lines.append("")
        return "\n".join(lines)

    harnesses = sorted({r.harness for r in records}, key=_harness_sort_key)
    tasks = sorted({r.task_id for r in records})
    per_harness_counts = {h: sum(1 for r in records if r.harness == h) for h in harnesses}
    counts_str = ", ".join(f"{h}={per_harness_counts[h]}" for h in harnesses)

    lines.append(f"- Runs recorded: {len(records)} ({counts_str})")
    lines.append(f"- Generated: {generated}")
    lines.append("")

    # --- Overall table -----------------------------------------------
    lines.append("## Overall")
    lines.append("")
    header = [
        "Harness",
        "Pass rate",
        "Mean API calls",
        "Mean tool calls",
        "Mean total tokens",
        "Mean cost (computed)",
        "Mean cost (reported)",
        "Mean wall (s)",
        "Denials (total)",
        "Verifier failures (total)",
        "Errors",
    ]
    rows: list[list[str]] = []
    for h in harnesses:
        group = [r for r in records if r.harness == h]
        ok = [r for r in group if r.error is None]
        reported = [r.reported_cost_usd for r in ok if r.reported_cost_usd is not None]
        rows.append(
            [
                f"{h} ({HARNESS_NAMES.get(h, h)})",
                _fmt_pass_rate(sum(1 for r in group if r.passed), len(group)),
                _fmt_num(_mean([r.api_calls for r in ok])),
                _fmt_num(_mean([r.tool_calls for r in ok])),
                _fmt_num(_mean([r.tokens.total for r in ok]), 0),
                _fmt_money(_mean([r.cost_usd for r in ok])),
                _fmt_money(_mean(reported) if reported else None),
                _fmt_num(_mean([r.wall_s for r in ok])),
                str(sum(r.denials for r in group)),
                str(sum(r.verifier_failures for r in group)),
                str(sum(1 for r in group if r.error is not None)),
            ]
        )
    lines.extend(_table(header, rows))
    lines.append("")

    # --- Per-task table -------------------------------------------------
    lines.append("## Per task")
    lines.append("")
    both = set(harnesses) == {"oh", "cc"}
    header = ["Task"]
    for h in harnesses:
        header.extend([f"{h} pass", f"{h} cost", f"{h} api", f"{h} wall"])
    if both:
        header.extend(["delta cost (oh-cc)", "delta api (oh-cc)"])

    rows = []
    for task_id in tasks:
        row = [task_id]
        means: dict[str, dict[str, float | None]] = {}
        for h in harnesses:
            group = [r for r in records if r.harness == h and r.task_id == task_id]
            ok = [r for r in group if r.error is None]
            mean_cost = _mean([r.cost_usd for r in ok])
            mean_api = _mean([r.api_calls for r in ok])
            mean_wall = _mean([r.wall_s for r in ok])
            means[h] = {"cost": mean_cost, "api": mean_api}
            row.extend(
                [
                    _fmt_pass_rate(sum(1 for r in group if r.passed), len(group)),
                    _fmt_money(mean_cost),
                    _fmt_num(mean_api),
                    _fmt_num(mean_wall),
                ]
            )
        if both:
            oh_cost, cc_cost = means.get("oh", {}).get("cost"), means.get("cc", {}).get("cost")
            oh_api, cc_api = means.get("oh", {}).get("api"), means.get("cc", {}).get("api")
            delta_cost = None if oh_cost is None or cc_cost is None else oh_cost - cc_cost
            delta_api = None if oh_api is None or cc_api is None else oh_api - cc_api
            row.extend([_fmt_delta_money(delta_cost), _fmt_delta_num(delta_api)])
        rows.append(row)
    lines.extend(_table(header, rows))
    lines.append("")

    # --- Failures ---------------------------------------------------
    lines.append("## Failures")
    lines.append("")
    failures = [r for r in records if not r.passed]
    if not failures:
        lines.append("No failures.")
    else:
        for r in failures:
            status = "ERROR" if r.error else "FAIL"
            detail = _truncate(r.error if r.error else r.check_detail)
            session = _truncate(r.session_ref)
            lines.append(
                f"- {r.task_id} {r.harness} run{r.run_index}: {status} - {detail} "
                f"(session: {session})"
            )
    lines.append("")

    return "\n".join(lines)
