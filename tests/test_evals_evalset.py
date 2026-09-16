"""The task registry is exported in Google ADK EvalSet JSON (SPEC 14.3)."""

from __future__ import annotations

import json

from evals import evalset
from evals.tasks import TASKS


def test_eval_set_has_one_case_per_task_with_prompts_in_order():
    doc = evalset.build_eval_set(TASKS)
    assert doc["eval_set_id"] == evalset.EVAL_SET_ID
    assert [c["eval_id"] for c in doc["eval_cases"]] == sorted(TASKS)
    for case in doc["eval_cases"]:
        task = TASKS[case["eval_id"]]
        prompts = [t["user_content"]["parts"][0]["text"] for t in case["conversation"]]
        assert prompts == task.prompts
        assert all(t["user_content"]["role"] == "user" for t in case["conversation"])
        assert case["open_harness"]["kind"] == task.kind
        assert case["open_harness"]["timeout_s"] == task.timeout_s


def test_committed_eval_set_file_is_current():
    """Regenerate with `python -m evals.evalset` when tasks change."""
    assert evalset.CASES_PATH.exists(), "run python -m evals.evalset"
    assert evalset.CASES_PATH.read_text(encoding="utf-8") == evalset.render(TASKS)


def test_render_is_deterministic_and_valid_json():
    a, b = evalset.render(TASKS), evalset.render(TASKS)
    assert a == b
    json.loads(a)


def test_main_check_reports_stale_file(tmp_path, capsys):
    out = tmp_path / "eval_set.json"
    assert evalset.main(["--out", str(out)]) == 0
    assert evalset.main(["--out", str(out), "--check"]) == 0
    out.write_text("{}", encoding="utf-8")
    assert evalset.main(["--out", str(out), "--check"]) == 1
