"""Tests for the eval fixture template and the ten task checkers.

Keeps runtime lean: one full pytest+ruff pass over the clean template, one
cheap sanity pass over the TASKS table, one build+check per edit task to
confirm its checker fails on an untouched (but possibly mutated) workspace,
and four "golden" hand-edits (t03, t04, t05, t07) to confirm the checkers
actually pass once the task is done correctly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from evals.fixture import build_workspace
from evals.tasks import TASKS, ruff_ok, run_tests

EDIT_TASK_IDS = [task_id for task_id, task in TASKS.items() if task.kind == "edit"]


def test_template_passes_its_own_tests_and_ruff(tmp_path: Path) -> None:
    ws = build_workspace(tmp_path / "clean")
    ok_tests, tests_detail = run_tests(ws)
    assert ok_tests, tests_detail
    ok_ruff, ruff_detail = ruff_ok(ws)
    assert ok_ruff, ruff_detail


def test_ten_tasks_with_expected_ids_and_kinds() -> None:
    assert set(TASKS) == {f"t{i:02d}" for i in range(1, 11)}
    for task_id, task in TASKS.items():
        assert task.id == task_id
        assert task.kind in ("qa", "edit")
        assert task.prompts and all(isinstance(p, str) and p.strip() for p in task.prompts)
        assert callable(task.check)
        # Prompts must read like a real user typed them, not mention the harness.
        for prompt in task.prompts:
            assert "open-harness" not in prompt.lower()
            assert "claude code" not in prompt.lower()


@pytest.mark.parametrize("task_id", EDIT_TASK_IDS)
def test_edit_task_checker_fails_on_untouched_workspace(tmp_path: Path, task_id: str) -> None:
    task = TASKS[task_id]
    ws = build_workspace(tmp_path / task_id, mutate=task.mutate)
    result = task.check(ws, [""] * len(task.prompts))
    assert result.passed is False, f"{task_id} unexpectedly passed: {result.detail}"


def test_t03_golden_add_net_change(tmp_path: Path) -> None:
    task = TASKS["t03"]
    ws = build_workspace(tmp_path / "t03", mutate=task.mutate)
    report = ws / "src" / "ledger" / "report.py"
    text = report.read_text(encoding="utf-8")
    text += (
        "\n\ndef net_change(transactions: list[Transaction]) -> float:\n"
        '    """Return the sum of a list of transactions\' amounts."""\n'
        "    return round(sum(t.amount for t in transactions), 2)\n"
    )
    text = text.replace(
        "from ledger.store import Store\n",
        "from ledger.store import Store\nfrom ledger.models import Transaction\n",
        1,
    )
    report.write_text(text, encoding="utf-8")
    result = task.check(ws, [""])
    assert result.passed, result.detail


def test_t04_golden_fix_sign_bug(tmp_path: Path) -> None:
    task = TASKS["t04"]
    ws = build_workspace(tmp_path / "t04", mutate=task.mutate)
    report = ws / "src" / "ledger" / "report.py"
    text = report.read_text(encoding="utf-8")
    assert "total -= txn.amount" in text, "fixture didn't plant the expected bug"
    report.write_text(text.replace("total -= txn.amount", "total += txn.amount"), encoding="utf-8")
    result = task.check(ws, [""])
    assert result.passed, result.detail


def test_t05_golden_rename_function(tmp_path: Path) -> None:
    task = TASKS["t05"]
    ws = build_workspace(tmp_path / "t05")
    for rel in (
        "src/ledger/report.py",
        "src/ledger/cli.py",
        "tests/test_report.py",
        "tests/test_cli.py",
    ):
        path = ws / rel
        text = path.read_text(encoding="utf-8")
        path.write_text(text.replace("account_balance", "balance_for_account"), encoding="utf-8")
    result = task.check(ws, [""])
    assert result.passed, result.detail


def test_t07_golden_add_overdrawn_tests(tmp_path: Path) -> None:
    task = TASKS["t07"]
    ws = build_workspace(tmp_path / "t07", mutate=task.mutate)
    test_report = ws / "tests" / "test_report.py"
    text = test_report.read_text(encoding="utf-8")
    text += (
        "\n\n"
        "def test_overdrawn_accounts_flags_negative_balance():\n"
        "    from ledger.models import Account, Transaction\n"
        "    from ledger.report import overdrawn_accounts\n"
        "    from ledger.store import Store\n"
        "    store = Store()\n"
        "    store.add_account(Account('a', 'A', 10.0))\n"
        "    store.add_transaction(Transaction('a', -25.0, 'rent'))\n"
        "    assert overdrawn_accounts(store) == ['a']\n"
        "\n\n"
        "def test_overdrawn_accounts_empty_when_all_positive():\n"
        "    from ledger.models import Account\n"
        "    from ledger.report import overdrawn_accounts\n"
        "    from ledger.store import Store\n"
        "    store = Store()\n"
        "    store.add_account(Account('a', 'A', 10.0))\n"
        "    assert overdrawn_accounts(store) == []\n"
    )
    test_report.write_text(text, encoding="utf-8")
    result = task.check(ws, [""])
    assert result.passed, result.detail


def test_build_workspace_writes_identical_instruction_files(tmp_path):
    from evals.fixture import build_workspace

    ws = build_workspace(tmp_path / "ws", python="C:/x/.venv/Scripts/python.exe")
    agents = (ws / "AGENTS.md").read_text(encoding="utf-8")
    assert agents == (ws / "CLAUDE.md").read_text(encoding="utf-8")
    assert "C:/x/.venv/Scripts/python.exe -m pytest -q" in agents
    # Committed, so "files changed vs HEAD" checkers do not count them.
    import subprocess
    out = subprocess.run(["git", "status", "--porcelain"], cwd=ws, capture_output=True,
                         text=True, stdin=subprocess.DEVNULL, check=False).stdout
    assert out.strip() == ""


def test_build_workspace_without_python_writes_no_instruction_files(tmp_path):
    from evals.fixture import build_workspace

    ws = build_workspace(tmp_path / "ws")
    assert not (ws / "AGENTS.md").exists()
    assert not (ws / "CLAUDE.md").exists()
