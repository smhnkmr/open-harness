"""Unit tests for evals.driver and evals.report.

evals.tasks / evals.fixture / evals.runners are being written concurrently
by other work, so the driver loop tests monkeypatch the driver's lazy
loaders (get_tasks, get_build_workspace, get_run_oh, get_run_cc) with
in-process fakes instead of depending on those modules existing.
"""

from __future__ import annotations

import io
import json
import re
from pathlib import Path

from evals import driver, report
from evals.types import CheckResult, RunRecord, TaskSpec, TokenUsage

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def make_task(task_id: str, *, passed: bool = True, detail: str = "ok", calls: list | None = None):
    def check(workspace: Path, texts: list[str]) -> CheckResult:
        if calls is not None:
            calls.append(task_id)
        return CheckResult(passed, detail)

    task = TaskSpec(id=task_id, kind="qa", prompts=["do the thing"], check=check)
    task.mutate = None  # extra field the real TaskSpec is expected to carry
    return task


def fake_build_workspace(dest: Path, *, mutate=None, python=None) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    if mutate is not None:
        mutate(dest)
    return dest


def make_fake_runner(harness: str, calls: list, raise_for: set | None = None):
    raise_for = raise_for or set()

    def runner(task, workspace, model, run_index, *, run_dir, timeout_s=None, **kwargs):
        calls.append((task.id, harness, run_index))
        run_dir.mkdir(parents=True, exist_ok=True)
        if (task.id, run_index) in raise_for:
            raise RuntimeError("simulated runner crash")
        return RunRecord(
            task_id=task.id,
            harness=harness,
            model=model,
            run_index=run_index,
            passed=False,  # driver overwrites this from the checker
            api_calls=3,
            tool_calls=2,
            wall_s=0.5,
            tokens=TokenUsage(input=100, output=50),
            cost_usd=0.01,
            session_ref=str(run_dir),
        )

    return runner


def make_record(
    task_id: str,
    harness: str,
    run_index: int = 1,
    *,
    passed: bool = True,
    error: str | None = None,
    api_calls: float = 10,
    tool_calls: float = 5,
    cost_usd: float = 0.10,
    reported_cost_usd: float | None = None,
    wall_s: float = 20.0,
    denials: int = 0,
    verifier_failures: int = 0,
    check_detail: str = "",
    session_ref: str = "sess-1",
) -> RunRecord:
    return RunRecord(
        task_id=task_id,
        harness=harness,
        model="claude-sonnet-5",
        run_index=run_index,
        passed=passed,
        check_detail=check_detail,
        api_calls=api_calls,
        tool_calls=tool_calls,
        tokens=TokenUsage(input=1000, output=500),
        cost_usd=cost_usd,
        reported_cost_usd=reported_cost_usd,
        denials=denials,
        verifier_failures=verifier_failures,
        wall_s=wall_s,
        error=error,
        session_ref=session_ref,
    )


# ---------------------------------------------------------------------------
# report.summarize
# ---------------------------------------------------------------------------


def test_summarize_empty_input():
    text = report.summarize([], model="claude-sonnet-5")
    assert "No runs recorded" in text
    assert "claude-sonnet-5" in text


def test_summarize_basic_both_harnesses_and_delta():
    records = [
        make_record("t01", "oh", 1, passed=True, api_calls=10, cost_usd=0.10),
        make_record("t01", "oh", 2, passed=False, check_detail="wrong output", api_calls=12,
                     cost_usd=0.12),
        make_record("t01", "cc", 1, passed=True, api_calls=8, cost_usd=0.20,
                     reported_cost_usd=0.18),
        make_record(
            "t01", "cc", 2, passed=False, error="TimeoutError: turn exceeded 600s",
            session_ref="s2",
        ),
    ]
    text = report.summarize(records, model="claude-sonnet-5")

    assert "claude-sonnet-5" in text
    assert "## Overall" in text
    assert "## Per task" in text
    assert "## Failures" in text
    # overall pass rates: oh 1/2, cc 1/2
    assert "1/2" in text
    # delta column present (both harnesses)
    assert "delta cost (oh-cc)" in text
    assert "delta api (oh-cc)" in text
    # failures section lists both the FAIL and the ERROR run, truncated fields
    assert "t01 oh run2: FAIL - wrong output" in text
    assert "t01 cc run2: ERROR - TimeoutError: turn exceeded 600s" in text
    # errored run excluded from cc's mean cost (only the passed run counts)
    assert "$0.2000" in text


def test_summarize_single_harness_has_no_delta_column():
    records = [make_record("t01", "oh", 1, passed=True)]
    text = report.summarize(records, model="m")
    assert "delta cost" not in text
    assert "oh" in text
    assert "cc" not in text.split("## Per task")[1].split("## Failures")[0]


def test_summarize_truncates_long_detail():
    long_detail = "x" * 300
    records = [make_record("t01", "oh", 1, passed=False, check_detail=long_detail)]
    text = report.summarize(records, model="m")
    assert long_detail not in text
    assert "x" * 197 + "..." in text


def test_load_records_roundtrip(tmp_path: Path):
    path = tmp_path / "runs.jsonl"
    records = [make_record("t01", "oh", 1), make_record("t01", "cc", 1)]
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_json()) + "\n")
    loaded = report.load_records(path)
    assert [r.task_id for r in loaded] == ["t01", "t01"]
    assert [r.harness for r in loaded] == ["oh", "cc"]


def test_load_records_missing_file(tmp_path: Path):
    assert report.load_records(tmp_path / "nope.jsonl") == []


# ---------------------------------------------------------------------------
# driver.main / run_matrix
# ---------------------------------------------------------------------------


def test_driver_interleaved_order_and_outputs(monkeypatch, tmp_path: Path):
    tasks = {"t01": make_task("t01"), "t02": make_task("t02")}
    calls: list[tuple[str, str, int]] = []

    monkeypatch.setattr(driver, "get_tasks", lambda: tasks)
    monkeypatch.setattr(driver, "get_build_workspace", lambda: fake_build_workspace)
    monkeypatch.setattr(driver, "get_run_oh", lambda: make_fake_runner("oh", calls))
    monkeypatch.setattr(driver, "get_run_cc", lambda: make_fake_runner("cc", calls))

    out_dir = tmp_path / "results"
    rc = driver.main(
        [
            "--harness", "oh,cc",
            "--model", "claude-sonnet-5",
            "--tasks", "all",
            "--runs", "2",
            "--out", str(out_dir),
        ]
    )
    assert rc == 0

    expected = [
        ("t01", "oh", 1), ("t01", "cc", 1),
        ("t01", "oh", 2), ("t01", "cc", 2),
        ("t02", "oh", 1), ("t02", "cc", 1),
        ("t02", "oh", 2), ("t02", "cc", 2),
    ]
    assert calls == expected

    runs_path = out_dir / "runs.jsonl"
    assert runs_path.exists()
    lines = [json.loads(line) for line in runs_path.read_text(encoding="utf-8").splitlines()]
    recorded = [(d["task_id"], d["harness"], d["run_index"]) for d in lines]
    assert recorded == expected
    assert all(d["passed"] is True for d in lines)

    assert (out_dir / "report.md").exists()
    assert "# Eval report" in (out_dir / "report.md").read_text(encoding="utf-8")

    for task_id, harness, run_index in expected:
        run_dir = out_dir / task_id / harness / f"run{run_index}"
        assert (run_dir / "workspace").is_dir()


def test_driver_progress_line_format(monkeypatch, tmp_path: Path, capsys):
    tasks = {"t03": make_task("t03")}
    calls: list = []
    monkeypatch.setattr(driver, "get_tasks", lambda: tasks)
    monkeypatch.setattr(driver, "get_build_workspace", lambda: fake_build_workspace)
    monkeypatch.setattr(driver, "get_run_oh", lambda: make_fake_runner("oh", calls))
    monkeypatch.setattr(driver, "get_run_cc", lambda: make_fake_runner("cc", calls))

    out_dir = tmp_path / "results"
    rc = driver.main(
        ["--harness", "oh", "--model", "m", "--tasks", "t03", "--runs", "1", "--out", str(out_dir)]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert re.search(r"^t03 oh run1 PASS \d+\.\d\d\$ \d+s api=\d+ tools=\d+$", out, re.MULTILINE)


def test_driver_resume_skips_existing_triples(monkeypatch, tmp_path: Path):
    tasks = {"t01": make_task("t01"), "t02": make_task("t02")}
    out_dir = tmp_path / "results"

    calls1: list = []
    monkeypatch.setattr(driver, "get_tasks", lambda: tasks)
    monkeypatch.setattr(driver, "get_build_workspace", lambda: fake_build_workspace)
    monkeypatch.setattr(driver, "get_run_oh", lambda: make_fake_runner("oh", calls1))
    monkeypatch.setattr(driver, "get_run_cc", lambda: make_fake_runner("cc", calls1))

    rc = driver.main(
        [
            "--harness", "oh,cc", "--model", "m", "--tasks", "t01",
            "--runs", "1", "--out", str(out_dir),
        ]
    )
    assert rc == 0
    assert len(calls1) == 2  # t01 oh run1, t01 cc run1

    calls2: list = []
    monkeypatch.setattr(driver, "get_run_oh", lambda: make_fake_runner("oh", calls2))
    monkeypatch.setattr(driver, "get_run_cc", lambda: make_fake_runner("cc", calls2))

    rc = driver.main(
        [
            "--harness", "oh,cc", "--model", "m", "--tasks", "t01,t02",
            "--runs", "1", "--out", str(out_dir), "--resume",
        ]
    )
    assert rc == 0
    # only the new task's runs should have been invoked; t01 run1 was skipped
    assert sorted(calls2) == [("t02", "cc", 1), ("t02", "oh", 1)]

    runs_path = out_dir / "runs.jsonl"
    lines = [json.loads(line) for line in runs_path.read_text(encoding="utf-8").splitlines()]
    recorded = {(d["task_id"], d["harness"], d["run_index"]) for d in lines}
    assert recorded == {("t01", "oh", 1), ("t01", "cc", 1), ("t02", "oh", 1), ("t02", "cc", 1)}
    assert len(lines) == 4  # no duplicates


def test_driver_dry_run_calls_no_runner(monkeypatch, tmp_path: Path):
    passing = make_task("t01", passed=True)
    failing = make_task("t02", passed=False, detail="checker says no")
    tasks = {"t01": passing, "t02": failing}

    def boom():
        raise AssertionError("runner should not be invoked during --dry-run")

    monkeypatch.setattr(driver, "get_tasks", lambda: tasks)
    monkeypatch.setattr(driver, "get_build_workspace", lambda: fake_build_workspace)
    monkeypatch.setattr(driver, "get_run_oh", boom)
    monkeypatch.setattr(driver, "get_run_cc", boom)

    out_dir = tmp_path / "results"
    rc = driver.main(
        ["--harness", "oh,cc", "--model", "m", "--tasks", "all", "--dry-run", "--out", str(out_dir)]
    )
    assert rc == 1  # t02's checker fails
    assert not (out_dir / "runs.jsonl").exists()
    assert (out_dir / "t01" / "dry-run" / "workspace").is_dir()
    assert (out_dir / "t02" / "dry-run" / "workspace").is_dir()


def test_driver_runner_exception_does_not_stop_matrix(monkeypatch, tmp_path: Path):
    check_calls: list = []
    tasks = {"t01": make_task("t01", calls=check_calls), "t02": make_task("t02", calls=check_calls)}
    calls: list = []

    monkeypatch.setattr(driver, "get_tasks", lambda: tasks)
    monkeypatch.setattr(driver, "get_build_workspace", lambda: fake_build_workspace)
    monkeypatch.setattr(
        driver, "get_run_oh", lambda: make_fake_runner("oh", calls, raise_for={("t01", 1)})
    )
    monkeypatch.setattr(driver, "get_run_cc", lambda: make_fake_runner("cc", calls))

    out_dir = tmp_path / "results"
    rc = driver.main(
        [
            "--harness", "oh,cc", "--model", "m", "--tasks", "all",
            "--runs", "2", "--out", str(out_dir),
        ]
    )
    assert rc == 0  # matrix completes despite the crash

    lines = [json.loads(line) for line in (out_dir / "runs.jsonl").read_text().splitlines()]
    assert len(lines) == 8  # all 8 combinations recorded

    crashed = [d for d in lines if d["task_id"] == "t01" and d["harness"] == "oh" and
               d["run_index"] == 1]
    assert len(crashed) == 1
    assert crashed[0]["error"] is not None
    assert crashed[0]["passed"] is False
    assert "simulated runner crash" in crashed[0]["error"]

    others_ok = [d for d in lines if not (d["task_id"] == "t01" and d["harness"] == "oh" and
                                           d["run_index"] == 1)]
    assert all(d["error"] is None for d in others_ok)

    # the checker must have been skipped for the crashed run (7 = 8 - 1)
    assert len(check_calls) == 7


def test_run_matrix_resolves_relative_out_dir(tmp_path, monkeypatch):
    """A relative --out must not leak relative workspace paths to runners."""
    from evals import driver
    from evals.types import CheckResult, RunRecord, TaskSpec

    monkeypatch.chdir(tmp_path)
    seen = {}

    def fake_run_oh(task, workspace, model, run_index, **kw):
        seen["workspace"] = Path(workspace)
        return RunRecord(task_id=task.id, harness="oh", model=model, run_index=run_index,
                         passed=False, final_texts=["x"])

    task = TaskSpec(id="t", kind="qa", prompts=["p"], check=lambda w, t: CheckResult(True))
    driver.run_matrix(harnesses=["oh"], model="m", tasks=[task], runs=1, out_dir=Path("rel"),
                      build_workspace_fn=fake_build_workspace, run_oh_fn=fake_run_oh,
                      run_cc_fn=None, stream=io.StringIO())
    assert seen["workspace"].is_absolute()
