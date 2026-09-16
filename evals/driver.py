"""Eval driver: runs the same tasks on open-harness and Claude Code and reports.

Spec: SPEC.md section 18.2, roadmap step 1. Not part of the installed
package; run from the repo root as `python -m evals.driver`.

The driver codes against three modules owned by other work in progress:
`evals.tasks` (TASKS registry), `evals.fixture` (build_workspace) and
`evals.runners` (run_oh / run_cc). They are imported lazily, through the
`get_*` functions below, so this module (and its tests) can load even
before those modules exist -- tests monkeypatch the `get_*` functions to
inject fakes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import TextIO

from evals.report import summarize
from evals.types import RunRecord, TaskSpec

REPO_ROOT = Path(__file__).resolve().parent.parent
VALID_HARNESSES = ("oh", "cc")


# --------------------------------------------------------------------------
# Lazy imports of the other agents' modules. Tests monkeypatch these
# functions directly (e.g. `monkeypatch.setattr(driver, "get_tasks", ...)`).
# --------------------------------------------------------------------------


def get_tasks() -> dict[str, TaskSpec]:
    from evals.tasks import TASKS

    return TASKS


def get_build_workspace() -> Callable[..., Path]:
    from evals.fixture import build_workspace

    return build_workspace


def get_run_oh() -> Callable[..., RunRecord]:
    from evals.runners import run_oh

    return run_oh


def get_run_cc() -> Callable[..., RunRecord]:
    from evals.runners import run_cc

    return run_cc


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def _select_tasks(tasks: dict[str, TaskSpec], spec: str) -> list[TaskSpec]:
    if spec == "all":
        ids = sorted(tasks)
    else:
        ids = [t.strip() for t in spec.split(",") if t.strip()]
    missing = [i for i in ids if i not in tasks]
    if missing:
        raise ValueError(f"unknown task id(s): {', '.join(missing)}")
    return [tasks[i] for i in ids]


def _read_existing(runs_path: Path) -> set[tuple[str, str, int]]:
    seen: set[tuple[str, str, int]] = set()
    if not runs_path.exists():
        return seen
    for line in runs_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        seen.add((data["task_id"], data["harness"], data["run_index"]))
    return seen


def _append_record(runs_path: Path, record: RunRecord) -> None:
    runs_path.parent.mkdir(parents=True, exist_ok=True)
    with runs_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record.to_json()) + "\n")


def _write_report(out_dir: Path, model: str) -> None:
    from evals.report import load_records

    records = load_records(out_dir / "runs.jsonl")
    (out_dir / "report.md").write_text(summarize(records, model=model), encoding="utf-8")


def _progress_line(record: RunRecord) -> str:
    if record.error:
        status = "ERROR"
    elif record.passed:
        status = "PASS"
    else:
        status = "FAIL"
    return (
        f"{record.task_id} {record.harness} run{record.run_index} {status} "
        f"{record.cost_usd:.2f}$ {record.wall_s:.0f}s "
        f"api={record.api_calls} tools={record.tool_calls}"
    )


# --------------------------------------------------------------------------
# Core run
# --------------------------------------------------------------------------


def _run_one(
    *,
    task: TaskSpec,
    harness: str,
    model: str,
    run_index: int,
    run_dir: Path,
    workspace_dir: Path,
    build_workspace_fn: Callable[..., Path],
    run_oh_fn: Callable[..., RunRecord],
    run_cc_fn: Callable[..., RunRecord],
    harness_python: str,
    repo_root: Path,
    timeout_s: float | None,
) -> RunRecord:
    """Build the workspace, invoke the harness, and apply the checker.

    Never raises: any exception (workspace build, harness crash, checker
    bug) is caught and turned into a RunRecord with `error` set, so one bad
    run cannot take down the rest of the matrix.
    """
    t0 = time.monotonic()
    mutate = getattr(task, "mutate", None)
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        build_workspace_fn(workspace_dir, mutate=mutate, python=harness_python)
        if harness == "oh":
            record = run_oh_fn(
                task,
                workspace_dir,
                model,
                run_index,
                run_dir=run_dir,
                harness_python=harness_python,
                repo_root=repo_root,
                timeout_s=timeout_s,
            )
        elif harness == "cc":
            record = run_cc_fn(
                task,
                workspace_dir,
                model,
                run_index,
                run_dir=run_dir,
                timeout_s=timeout_s,
                harness_python=harness_python,
            )
        else:
            raise ValueError(f"unknown harness: {harness!r}")
    except Exception as exc:  # noqa: BLE001 - deliberately broad, see docstring
        wall = time.monotonic() - t0
        return RunRecord(
            task_id=task.id,
            harness=harness,
            model=model,
            run_index=run_index,
            passed=False,
            wall_s=wall,
            error=f"{type(exc).__name__}: {exc}",
        )

    if record.error:
        return record

    try:
        result = task.check(workspace_dir, record.final_texts)
    except Exception as exc:  # noqa: BLE001 - same rationale as above
        record.passed = False
        record.error = f"checker raised {type(exc).__name__}: {exc}"
        return record

    record.passed = result.passed
    record.check_detail = result.detail
    return record


def run_matrix(
    *,
    harnesses: list[str],
    model: str,
    tasks: list[TaskSpec],
    runs: int,
    out_dir: Path,
    harness_python: str | None = None,
    repo_root: Path | None = None,
    timeout_s: float | None = None,
    resume: bool = False,
    build_workspace_fn: Callable[..., Path] | None = None,
    run_oh_fn: Callable[..., RunRecord] | None = None,
    run_cc_fn: Callable[..., RunRecord] | None = None,
    stream: TextIO | None = None,
) -> list[RunRecord]:
    """Run every (task, run_index, harness) combination and report as it goes.

    Order is interleaved: for each task, for each run index, for each
    harness -- so both harnesses see the same API weather for a given run.
    Each RunRecord is appended to `<out_dir>/runs.jsonl` immediately, and
    `<out_dir>/report.md` is regenerated after every record.
    """
    repo_root = repo_root or REPO_ROOT
    harness_python = harness_python or sys.executable
    build_workspace_fn = build_workspace_fn or get_build_workspace()
    run_oh_fn = run_oh_fn or get_run_oh()
    run_cc_fn = run_cc_fn or get_run_cc()
    stream = stream if stream is not None else sys.stdout

    out_dir.mkdir(parents=True, exist_ok=True)
    runs_path = out_dir / "runs.jsonl"
    existing = _read_existing(runs_path) if resume else set()

    new_records: list[RunRecord] = []
    for task in tasks:
        for run_index in range(1, runs + 1):
            for harness in harnesses:
                if (task.id, harness, run_index) in existing:
                    continue
                run_dir = out_dir / task.id / harness / f"run{run_index}"
                workspace_dir = run_dir / "workspace"
                record = _run_one(
                    task=task,
                    harness=harness,
                    model=model,
                    run_index=run_index,
                    run_dir=run_dir,
                    workspace_dir=workspace_dir,
                    build_workspace_fn=build_workspace_fn,
                    run_oh_fn=run_oh_fn,
                    run_cc_fn=run_cc_fn,
                    harness_python=harness_python,
                    repo_root=repo_root,
                    timeout_s=timeout_s,
                )
                _append_record(runs_path, record)
                new_records.append(record)
                print(_progress_line(record), file=stream)
                _write_report(out_dir, model)

    return new_records


def _dry_run(
    tasks: list[TaskSpec],
    out_dir: Path,
    build_workspace_fn: Callable[..., Path],
    stream: TextIO,
    harness_python: str | None = None,
) -> int:
    """Build each task's workspace and run its checker, untouched, no harness.

    Cheap validation of fixtures/checkers. Returns 0 if every checker
    passed (or reported a clean CheckResult), 1 otherwise.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    any_failed = False
    for task in tasks:
        workspace_dir = out_dir / task.id / "dry-run" / "workspace"
        mutate = getattr(task, "mutate", None)
        try:
            build_workspace_fn(workspace_dir, mutate=mutate, python=harness_python)
            result = task.check(workspace_dir, [])
            status = "PASS" if result.passed else "FAIL"
            detail = f" - {result.detail}" if result.detail else ""
            if not result.passed:
                any_failed = True
        except Exception as exc:  # noqa: BLE001 - report and keep going
            any_failed = True
            status = "ERROR"
            detail = f" - {type(exc).__name__}: {exc}"
        print(f"{task.id} dry-run {status}{detail}", file=stream)
    return 1 if any_failed else 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evals.driver",
        description="Run the eval task matrix against open-harness and/or Claude Code.",
    )
    parser.add_argument("--harness", default="oh,cc", help="comma list of oh,cc (default oh,cc)")
    parser.add_argument("--model", required=True, help="model id, e.g. claude-sonnet-5")
    parser.add_argument("--tasks", default="all", help="'all' or comma list of task ids")
    parser.add_argument("--runs", type=int, default=5, help="runs per (task, harness)")
    parser.add_argument("--out", default=None, help="output dir (default evals/results/<ts>)")
    parser.add_argument("--harness-python", default=None, help="python to run open-harness with")
    parser.add_argument("--timeout", type=float, default=None, help="per-turn timeout seconds")
    parser.add_argument("--dry-run", action="store_true", help="validate fixtures/checkers only")
    parser.add_argument("--resume", action="store_true", help="skip runs already in runs.jsonl")
    return parser


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    return _build_parser().parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    harnesses = [h.strip() for h in args.harness.split(",") if h.strip()]
    unknown = [h for h in harnesses if h not in VALID_HARNESSES]
    if not harnesses or unknown:
        print(f"error: --harness must be a comma list of oh,cc (got {args.harness!r})",
              file=sys.stderr)
        return 2

    try:
        all_tasks = get_tasks()
    except Exception as exc:  # noqa: BLE001
        print(f"error: could not load evals.tasks.TASKS: {exc}", file=sys.stderr)
        return 2

    try:
        selected_tasks = _select_tasks(all_tasks, args.tasks)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    out_dir = (
        Path(args.out)
        if args.out
        else REPO_ROOT / "evals" / "results" / datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    )

    try:
        build_workspace_fn = get_build_workspace()
    except Exception as exc:  # noqa: BLE001
        print(f"error: could not load evals.fixture.build_workspace: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        return _dry_run(selected_tasks, out_dir, build_workspace_fn, sys.stdout,
                        harness_python=args.harness_python or sys.executable)

    try:
        run_oh_fn = get_run_oh()
        run_cc_fn = get_run_cc()
    except Exception as exc:  # noqa: BLE001
        print(f"error: could not load evals.runners: {exc}", file=sys.stderr)
        return 2

    run_matrix(
        harnesses=harnesses,
        model=args.model,
        tasks=selected_tasks,
        runs=args.runs,
        out_dir=out_dir,
        harness_python=args.harness_python,
        repo_root=REPO_ROOT,
        timeout_s=args.timeout,
        resume=args.resume,
        build_workspace_fn=build_workspace_fn,
        run_oh_fn=run_oh_fn,
        run_cc_fn=run_cc_fn,
        stream=sys.stdout,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
