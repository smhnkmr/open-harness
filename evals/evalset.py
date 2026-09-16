"""Export the task registry as a Google ADK `EvalSet` JSON document.

Spec: SPEC.md section 14.3. The case *format* is borrowed from ADK so the
same cases can be loaded by `adk eval` unchanged; the driver, runners and
checkers stay ours. Fields ADK does not define (task kind, checker and
mutation names, timeout) live under an `open_harness` key on each case,
which ADK ignores.

    python -m evals.evalset            # writes evals/cases/eval_set.json
    python -m evals.evalset --check    # exit 1 if the file is out of date
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Any

from evals.types import TaskSpec

EVAL_SET_ID = "open-harness-core"
CASES_PATH = Path(__file__).resolve().parent / "cases" / "eval_set.json"
# Fixed so regenerating the file does not churn the diff.
_CREATION_TIMESTAMP = 1789516800.0  # 2026-09-16T00:00:00Z


def _invocation_id(task_id: str, turn: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"open-harness/{task_id}/{turn}"))


def case_for(task: TaskSpec) -> dict[str, Any]:
    """One ADK `EvalCase`: a conversation of user turns with no expected
    trajectory (outcome is judged by the task's checker, not by trajectory
    matching, so `final_response` and `intermediate_data` stay null)."""
    conversation = [
        {
            "invocation_id": _invocation_id(task.id, i),
            "user_content": {"role": "user", "parts": [{"text": prompt}]},
            "final_response": None,
            "intermediate_data": None,
            "creation_timestamp": _CREATION_TIMESTAMP,
        }
        for i, prompt in enumerate(task.prompts)
    ]
    mutate = getattr(task, "mutate", None)
    return {
        "eval_id": task.id,
        "conversation": conversation,
        "session_input": {
            "app_name": "open-harness",
            "user_id": "eval",
            "state": {"fixture": "evals/fixture/template"},
        },
        "creation_timestamp": _CREATION_TIMESTAMP,
        "open_harness": {
            "kind": task.kind,
            "description": task.description,
            "timeout_s": task.timeout_s,
            "checker": getattr(task.check, "__name__", "check"),
            "mutate": getattr(mutate, "__name__", None) if mutate else None,
        },
    }


def build_eval_set(tasks: dict[str, TaskSpec]) -> dict[str, Any]:
    return {
        "eval_set_id": EVAL_SET_ID,
        "name": "open-harness core tasks",
        "description": (
            "Ten coding-agent tasks against the ledger fixture, run on open-harness "
            "and Claude Code by evals/driver.py. Outcomes are judged by per-task "
            "checkers (evals/tasks.py); ADK trajectory metrics are not used."
        ),
        "eval_cases": [case_for(tasks[k]) for k in sorted(tasks)],
        "creation_timestamp": _CREATION_TIMESTAMP,
    }


def render(tasks: dict[str, TaskSpec]) -> str:
    return json.dumps(build_eval_set(tasks), indent=2, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.evalset")
    parser.add_argument("--out", type=Path, default=CASES_PATH)
    parser.add_argument("--check", action="store_true", help="verify the file is current")
    args = parser.parse_args(argv)

    from evals.tasks import TASKS

    text = render(TASKS)
    if args.check:
        current = args.out.read_text(encoding="utf-8") if args.out.exists() else ""
        if current != text:
            print(f"{args.out} is out of date; run python -m evals.evalset", file=sys.stderr)
            return 1
        print(f"{args.out} is current")
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"wrote {args.out} ({len(TASKS)} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
