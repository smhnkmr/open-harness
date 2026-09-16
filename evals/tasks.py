"""Ten eval tasks run against the `ledger` fixture (evals/fixture).

Spec: SPEC.md section 18.2, roadmap step 1. Each `TaskSpec` pairs a set of
user-facing prompts with a `check` function that inspects the workspace
(and the harness's final answers) after the run. Checkers never raise: any
internal failure is caught and reported as a failed `CheckResult` so a
broken checker looks like "task failed", not a driver crash.

Task ids are stable (`t01`..`t10`) so run records can be joined across
harnesses and over time.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

from evals.types import CheckResult, TaskSpec

# The interpreter used for every subprocess this module runs (pytest, ruff,
# the ledger CLI, one-off import checks). Overridable so CI or a different
# machine can point this at a different venv without editing the file.
PYTHON = os.environ.get(
    "EVALS_PYTHON",
    "C:/Drive/play/harness/open-harness/.venv/Scripts/python.exe",
)

_SUBPROCESS_TIMEOUT_S = 60


# --------------------------------------------------------------------------- subprocess helpers


def _with_pythonpath(workspace: Path) -> dict[str, str]:
    env = dict(os.environ)
    src = str(workspace / "src")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{src}{os.pathsep}{existing}" if existing else src
    return env


def _run(
    argv: list[str],
    workspace: Path,
    *,
    env: dict[str, str] | None = None,
    timeout: int = _SUBPROCESS_TIMEOUT_S,
) -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            argv,
            cwd=workspace,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            check=False,
        )
        return proc.returncode == 0, proc.stdout + proc.stderr
    except subprocess.TimeoutExpired as exc:
        return False, f"timed out after {timeout}s: {exc}"
    except Exception as exc:  # noqa: BLE001 - checkers must never raise
        return False, f"subprocess error: {exc!r}"


def run_tests(workspace: Path, timeout: int = _SUBPROCESS_TIMEOUT_S) -> tuple[bool, str]:
    """Run the workspace's own pytest suite. Returns (all passed, output)."""
    return _run([PYTHON, "-m", "pytest", "-q"], workspace, timeout=timeout)


def ruff_ok(workspace: Path, timeout: int = _SUBPROCESS_TIMEOUT_S) -> tuple[bool, str]:
    """Run `ruff check .` in the workspace. Returns (clean, output)."""
    return _run([PYTHON, "-m", "ruff", "check", "."], workspace, timeout=timeout)


def _run_python(workspace: Path, code: str, timeout: int = 20) -> tuple[bool, str]:
    """Run `code` with `python -c`, with `src/` on PYTHONPATH."""
    return _run([PYTHON, "-c", code], workspace, env=_with_pythonpath(workspace), timeout=timeout)


def _run_cli(workspace: Path, args: list[str], timeout: int = 20) -> tuple[bool, str]:
    """Run `python -m ledger.cli <args>`, with `src/` on PYTHONPATH."""
    return _run(
        [PYTHON, "-m", "ledger.cli", *args],
        workspace,
        env=_with_pythonpath(workspace),
        timeout=timeout,
    )


def _changed_files_under_src(workspace: Path) -> list[str]:
    """Files under src/ that differ from the fixture commit (tracked or not)."""
    ok, out = _run(["git", "status", "--porcelain"], workspace, timeout=15)
    if not ok:
        return []
    files = []
    for line in out.splitlines():
        path = line[3:].strip()
        # Handle "old -> new" rename entries by taking the new path.
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        if path.replace("\\", "/").startswith("src/"):
            files.append(path)
    return files


def _count_tests_referencing(workspace: Path, symbol: str) -> int:
    """Count test functions under tests/ whose body mentions `symbol`.

    Splits each file on function boundaries; naive (doesn't parse the AST),
    but good enough for this fixture's flat, unnested test functions.
    """
    count = 0
    tests_dir = workspace / "tests"
    if not tests_dir.is_dir():
        return 0
    for path in tests_dir.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for chunk in re.split(r"\n(?=def )", text):
            if chunk.lstrip().startswith("def test_") and symbol in chunk:
                count += 1
    return count


def _read(workspace: Path, rel: str) -> str:
    return (workspace / rel).read_text(encoding="utf-8")


# --------------------------------------------------------------------------- mutations


def _mutate_t04_sign_bug(dest: Path) -> None:
    """Flip `+=` to `-=` in account_balance: a one-character sign bug."""
    path = dest / "src" / "ledger" / "report.py"
    text = path.read_text(encoding="utf-8")
    old = "        total += txn.amount\n"
    new = "        total -= txn.amount\n"
    if old not in text:
        raise RuntimeError("t04 mutate: expected line not found in report.py")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def _mutate_t07_drop_tests(dest: Path) -> None:
    """Remove the tests covering `overdrawn_accounts`, leaving the function."""
    path = dest / "tests" / "test_report.py"
    text = path.read_text(encoding="utf-8")
    kept = []
    removed = 0
    for chunk in re.split(r"\n(?=def )", text):
        if chunk.lstrip().startswith("def test_") and "overdrawn_accounts" in chunk:
            removed += 1
            continue
        kept.append(chunk)
    if not removed:
        raise RuntimeError("t07 mutate: no overdrawn_accounts tests found to remove")
    path.write_text("\n".join(kept), encoding="utf-8")


# --------------------------------------------------------------------------- checkers


def _last_text(final_texts: list[str]) -> str:
    return final_texts[-1] if final_texts else ""


def check_t01(workspace: Path, final_texts: list[str]) -> CheckResult:
    try:
        text = _last_text(final_texts).lower()
        modules = ["models", "store", "report", "cli"]
        hits = [m for m in modules if m in text]
        mentions_cli = "cli" in text or "argparse" in text
        passed = len(hits) >= 3 and mentions_cli
        return CheckResult(passed, f"module mentions={hits} cli/argparse={mentions_cli}")
    except Exception as exc:  # noqa: BLE001 - checkers must never raise
        return CheckResult(False, f"checker error: {exc!r}")


def check_t02(workspace: Path, final_texts: list[str]) -> CheckResult:
    try:
        norm = _last_text(final_texts).lower().replace("`", "").replace("*", "")
        has_file = "report.py" in norm
        has_func = "account_balance" in norm
        return CheckResult(has_file and has_func, f"report.py={has_file} account_balance={has_func}")
    except Exception as exc:  # noqa: BLE001 - checkers must never raise
        return CheckResult(False, f"checker error: {exc!r}")


def check_t03(workspace: Path, final_texts: list[str]) -> CheckResult:
    try:
        ok_tests, tests_out = run_tests(workspace)
        script = (
            "from ledger.report import net_change\n"
            "from ledger.models import Transaction\n"
            "cases = [\n"
            "    ([], 0.0),\n"
            "    ([Transaction('a', 10.0, 'x')], 10.0),\n"
            "    ([Transaction('a', 10.5, 'x'), Transaction('a', -3.25, 'y')], 7.25),\n"
            "]\n"
            "for txns, expected in cases:\n"
            "    got = net_change(txns)\n"
            "    assert abs(got - expected) < 1e-6, (txns, got, expected)\n"
            "print('NET_CHANGE_OK')\n"
        )
        ok_func, func_out = _run_python(workspace, script)
        func_ok = ok_func and "NET_CHANGE_OK" in func_out
        passed = ok_tests and func_ok
        detail = (
            f"tests_ok={ok_tests} net_change_ok={func_ok} "
            f"func_out={func_out[-300:]!r} tests_out={tests_out[-200:]!r}"
        )
        return CheckResult(passed, detail)
    except Exception as exc:  # noqa: BLE001 - checkers must never raise
        return CheckResult(False, f"checker error: {exc!r}")


def check_t04(workspace: Path, final_texts: list[str]) -> CheckResult:
    try:
        report_src = _read(workspace, "src/ledger/report.py")
        bug_gone = "total -= txn.amount" not in report_src
        ok_tests, tests_out = run_tests(workspace)
        passed = bug_gone and ok_tests
        return CheckResult(passed, f"bug_gone={bug_gone} tests_ok={ok_tests} {tests_out[-400:]}")
    except Exception as exc:  # noqa: BLE001 - checkers must never raise
        return CheckResult(False, f"checker error: {exc!r}")


_T05_OLD_NAME = "account_balance"
_T05_NEW_NAME = "balance_for_account"


def check_t05(workspace: Path, final_texts: list[str]) -> CheckResult:
    try:
        files = list((workspace / "src").rglob("*.py")) + list((workspace / "tests").rglob("*.py"))
        texts = [f.read_text(encoding="utf-8") for f in files]
        old_present = any(_T05_OLD_NAME in t for t in texts)
        new_present = any(_T05_NEW_NAME in t for t in texts)
        ok_tests, tests_out = run_tests(workspace)
        passed = (not old_present) and new_present and ok_tests
        detail = f"old_present={old_present} new_present={new_present} tests_ok={ok_tests} {tests_out[-400:]}"
        return CheckResult(passed, detail)
    except Exception as exc:  # noqa: BLE001 - checkers must never raise
        return CheckResult(False, f"checker error: {exc!r}")


def check_t06(workspace: Path, final_texts: list[str]) -> CheckResult:
    try:
        with tempfile.TemporaryDirectory() as td:
            store_path = Path(td) / "store.json"
            data = {
                "accounts": [
                    {"id": "chk", "name": "Checking", "opening_balance": 100.0},
                    {"id": "sav", "name": "Savings", "opening_balance": 50.0},
                ],
                "transactions": [
                    {"account_id": "chk", "amount": -20.0, "description": "coffee"},
                    {"account_id": "sav", "amount": 5.0, "description": "interest"},
                ],
            }
            store_path.write_text(json.dumps(data), encoding="utf-8")
            ok, out = _run_cli(workspace, ["--file", str(store_path), "summary"])
            expected_bits = ["Checking", "80.00", "Savings", "55.00"]
            passed = ok and all(bit in out for bit in expected_bits)
            return CheckResult(passed, f"rc_ok={ok} stdout={out[:400]!r}")
    except Exception as exc:  # noqa: BLE001 - checkers must never raise
        return CheckResult(False, f"checker error: {exc!r}")


def check_t07(workspace: Path, final_texts: list[str]) -> CheckResult:
    try:
        new_tests = _count_tests_referencing(workspace, "overdrawn_accounts")
        ok_tests, tests_out = run_tests(workspace)
        passed = new_tests >= 2 and ok_tests
        return CheckResult(passed, f"new_tests={new_tests} tests_ok={ok_tests} {tests_out[-400:]}")
    except Exception as exc:  # noqa: BLE001 - checkers must never raise
        return CheckResult(False, f"checker error: {exc!r}")


def check_t08(workspace: Path, final_texts: list[str]) -> CheckResult:
    try:
        script = (
            "from ledger.report import total_balance\n"
            "from ledger.models import Account\n"
            "from ledger.store import Store\n"
            "store = Store()\n"
            "store.add_account(Account('a', 'A', 10.0))\n"
            "store.add_account(Account('b', 'B', -3.0))\n"
            "assert total_balance(store) == 7.0, total_balance(store)\n"
            "print('TOTAL_BALANCE_OK')\n"
        )
        ok_func, func_out = _run_python(workspace, script)
        feature_ok = ok_func and "TOTAL_BALANCE_OK" in func_out
        new_tests = _count_tests_referencing(workspace, "total_balance")
        ok_tests, tests_out = run_tests(workspace)
        passed = feature_ok and new_tests >= 1 and ok_tests
        detail = (
            f"feature_ok={feature_ok} new_tests={new_tests} tests_ok={ok_tests} "
            f"func_out={func_out[-300:]!r} tests_out={tests_out[-200:]!r}"
        )
        return CheckResult(passed, detail)
    except Exception as exc:  # noqa: BLE001 - checkers must never raise
        return CheckResult(False, f"checker error: {exc!r}")


def check_t09(workspace: Path, final_texts: list[str]) -> CheckResult:
    try:
        norm = _last_text(final_texts).lower().replace("`", "")
        has_file = "store.py" in norm
        has_func = "get_account" in norm
        return CheckResult(has_file and has_func, f"store.py={has_file} get_account={has_func}")
    except Exception as exc:  # noqa: BLE001 - checkers must never raise
        return CheckResult(False, f"checker error: {exc!r}")


def check_t10(workspace: Path, final_texts: list[str]) -> CheckResult:
    try:
        changed = _changed_files_under_src(workspace)
        ok_tests, tests_out = run_tests(workspace)
        ok_ruff, ruff_out = ruff_ok(workspace)
        passed = len(changed) >= 1 and ok_tests and ok_ruff
        detail = (
            f"files_changed={len(changed)} changed={changed} "
            f"tests_ok={ok_tests} ruff_ok={ok_ruff} "
            f"tests_out={tests_out[-200:]!r} ruff_out={ruff_out[-300:]!r}"
        )
        return CheckResult(passed, detail)
    except Exception as exc:  # noqa: BLE001 - checkers must never raise
        return CheckResult(False, f"checker error: {exc!r}")


# --------------------------------------------------------------------------- tasks

TASKS: dict[str, TaskSpec] = {
    "t01": TaskSpec(
        id="t01",
        kind="qa",
        description="Repo overview: name the modules and note it's a CLI app.",
        prompts=[
            (
                "I just cloned this repo and haven't looked at the code yet. "
                "Can you give me a quick overview of what it does and how the "
                "code is organized before I start poking around?"
            ),
        ],
        check=check_t01,
    ),
    "t02": TaskSpec(
        id="t02",
        kind="qa",
        description="Locate where balances are computed (report.py:account_balance).",
        prompts=[
            (
                "Where in this codebase is the logic that turns an account's "
                "opening balance and its transactions into a current balance? "
                "I need the exact file and function name so I can look at it."
            ),
        ],
        check=check_t02,
    ),
    "t03": TaskSpec(
        id="t03",
        kind="edit",
        description="Add report.net_change(transactions) -> float.",
        prompts=[
            (
                "Please add a small helper function called `net_change` to "
                "src/ledger/report.py. It should take a list of Transaction "
                "objects (the same type used elsewhere in this package) and "
                "return the sum of their `amount` fields as a float, rounded to "
                "2 decimal places, the same way the other functions in that "
                "file round their results. It doesn't need to touch the store "
                "or any account - just add up the amounts it's given. Make sure "
                "the existing tests still pass."
            ),
        ],
        check=check_t03,
    ),
    "t04": TaskSpec(
        id="t04",
        kind="edit",
        description="Fix a planted sign bug in account_balance (mutate flips += to -=).",
        prompts=[
            (
                "Something's wrong with the balance calculation - I added an "
                "opening balance of 100 to an account and then a transaction "
                "for -20, and the reported balance came out as 120 instead of "
                "80. Can you find the bug and fix it? Please make sure the test "
                "suite passes afterwards."
            ),
        ],
        check=check_t04,
        mutate=_mutate_t04_sign_bug,
    ),
    "t05": TaskSpec(
        id="t05",
        kind="edit",
        description="Rename account_balance -> balance_for_account everywhere.",
        prompts=[
            (
                "I'd like to rename the `account_balance` function to "
                "`balance_for_account` - I think the new name reads better at "
                "the call site. Please rename it everywhere it's defined, "
                "called, and referenced (including the tests), and make sure "
                "nothing still refers to the old name. The test suite should "
                "still pass afterwards."
            ),
        ],
        check=check_t05,
    ),
    "t06": TaskSpec(
        id="t06",
        kind="edit",
        description="Add a `summary` CLI subcommand that prints format_summary().",
        prompts=[
            (
                "Can you add a new `summary` subcommand to the CLI? It should "
                "take no extra arguments beyond the usual `--file`, and it "
                "should just print the balance summary for every account in "
                "the ledger, using the existing `format_summary` function from "
                "report.py - one line per account, in the same format that "
                "function already produces."
            ),
        ],
        check=check_t06,
    ),
    "t07": TaskSpec(
        id="t07",
        kind="edit",
        description="Add tests for overdrawn_accounts (mutate removes existing ones).",
        prompts=[
            (
                "I noticed the `overdrawn_accounts` function in report.py "
                "doesn't have any test coverage right now. Could you add a "
                "couple of tests for it? Cover at least the case where an "
                "account is overdrawn and the case where none are."
            ),
        ],
        check=check_t07,
        mutate=_mutate_t07_drop_tests,
    ),
    "t08": TaskSpec(
        id="t08",
        kind="edit",
        description="Two turns: add total_balance(store), then test it.",
        prompts=[
            (
                "Please add a `total_balance` function to report.py that takes "
                "a Store and returns the sum of every account's current "
                "balance (i.e. the sum of what `account_balance` would return "
                "for each account), rounded to 2 decimal places the same way "
                "the rest of that file does."
            ),
            (
                "Thanks - now please add tests for the `total_balance` function "
                "you just added. Cover a normal case with a couple of accounts, "
                "not just a trivial empty one."
            ),
        ],
        check=check_t08,
    ),
    "t09": TaskSpec(
        id="t09",
        kind="qa",
        description="Diagnose a KeyError traceback from an unknown account id.",
        prompts=[
            (
                "I ran this and got a traceback I don't understand:\n\n"
                "$ python -m ledger.cli --file mine.json balance chk\n"
                "Traceback (most recent call last):\n"
                '  File "src/ledger/cli.py", line 78, in main\n'
                "    handlers[args.command](store, args)\n"
                '  File "src/ledger/cli.py", line 61, in _cmd_balance\n'
                "    print(f\"{account_balance(store, args.account_id):.2f}\")\n"
                '  File "src/ledger/report.py", line 19, in account_balance\n'
                "    account = store.get_account(account_id)\n"
                '  File "src/ledger/store.py", line 36, in get_account\n'
                '    raise KeyError(f"unknown account: {account_id}") from None\n'
                "KeyError: 'unknown account: chk'\n\n"
                "What's actually going wrong here, and which file and function "
                "is responsible for the error being raised?"
            ),
        ],
        check=check_t09,
    ),
    "t10": TaskSpec(
        id="t10",
        kind="edit",
        description="Underspecified: 'make the report output nicer' (scope discipline).",
        prompts=[
            (
                "The report output from `format_summary` looks pretty rough "
                "right now - could you make it nicer to look at?"
            ),
        ],
        check=check_t10,
    ),
}
