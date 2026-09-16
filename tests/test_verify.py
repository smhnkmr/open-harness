"""Tests for open_harness.verify.gate.

Spec: open-harness-spec.md section 7.
"""

from __future__ import annotations

import sys
from pathlib import Path

from open_harness.backend.base import Backend, ExecResult, FileOpResult
from open_harness.backend.local import LocalBackend
from open_harness.config import VerifyConfig
from open_harness.verify.gate import GateResult, run_gate, run_lint


class FakeBackend(Backend):
    """A Backend whose `execute` returns pre-scripted results (or raises),
    so lint/gate behavior can be tested without shelling out."""

    def __init__(self, result: ExecResult | None = None, raises: Exception | None = None) -> None:
        self.root = Path(".")
        self.result = result or ExecResult(stdout="", stderr="", exit_code=0)
        self.raises = raises
        self.calls: list[str] = []

    def execute(self, cmd: str, *, timeout: float, env: dict[str, str] | None = None,
                cwd: Path | None = None) -> ExecResult:
        self.calls.append(cmd)
        if self.raises is not None:
            raise self.raises
        return self.result

    def upload(self, files: dict[str, bytes]) -> list[FileOpResult]:
        return []

    def download(self, paths: list[str]) -> dict[str, bytes | FileOpResult]:
        return {}


# ------------------------------------------------------------------- run_lint


def test_run_lint_not_configured_returns_empty() -> None:
    backend = FakeBackend()
    assert run_lint(backend, "a.py", VerifyConfig(lint=None), "/cwd") == ""
    assert backend.calls == []  # never even invoked


def test_run_lint_clean_pass_returns_empty() -> None:
    backend = FakeBackend(ExecResult(stdout="", stderr="", exit_code=0))
    out = run_lint(backend, "a.py", VerifyConfig(lint="ruff check {file}"), "/cwd")
    assert out == ""


def test_run_lint_substitutes_file_placeholder() -> None:
    backend = FakeBackend(ExecResult(stdout="", stderr="", exit_code=0))
    run_lint(backend, "src/a.py", VerifyConfig(lint="ruff check {file} --fix"), "/cwd")
    assert backend.calls == ["ruff check src/a.py --fix"]


def test_run_lint_failure_returns_prefixed_tail() -> None:
    lines = [f"error {i}" for i in range(100)]
    backend = FakeBackend(ExecResult(stdout="", stderr="\n".join(lines), exit_code=1))
    out = run_lint(backend, "a.py", VerifyConfig(lint="ruff check {file}"), "/cwd")
    assert out.startswith("LINT ERRORS:\n")
    body = out[len("LINT ERRORS:\n"):]
    body_lines = body.splitlines()
    assert len(body_lines) == 60
    assert body_lines[-1] == "error 99"
    assert "error 0" not in body_lines  # only the tail is kept


def test_run_lint_prefers_stderr_falls_back_to_stdout() -> None:
    backend = FakeBackend(ExecResult(stdout="stdout errors", stderr="", exit_code=1))
    out = run_lint(backend, "a.py", VerifyConfig(lint="lint {file}"), "/cwd")
    assert "stdout errors" in out


def test_run_lint_never_raises_on_backend_exception() -> None:
    backend = FakeBackend(raises=RuntimeError("no such command"))
    out = run_lint(backend, "a.py", VerifyConfig(lint="lint {file}"), "/cwd")
    assert out.startswith("LINT ERRORS:\n")
    assert "no such command" in out


def test_run_lint_real_backend_pass_and_fail(tmp_path: Path) -> None:
    backend = LocalBackend(root=tmp_path)
    py = f'"{sys.executable}"'
    ok_cfg = VerifyConfig(lint=f'{py} -c "exit(0)"')
    assert run_lint(backend, "a.py", ok_cfg, tmp_path) == ""

    bad_cfg = VerifyConfig(lint=f'{py} -c "import sys; sys.stderr.write(\'boom\\n\'); sys.exit(1)"')
    out = run_lint(backend, "a.py", bad_cfg, tmp_path)
    assert out.startswith("LINT ERRORS:\n")
    assert "boom" in out


# ------------------------------------------------------------------- run_gate


def test_run_gate_no_test_configured() -> None:
    backend = FakeBackend()
    result = run_gate(backend, VerifyConfig(test=None), "/cwd")
    assert result.ok is True
    assert result.failures == "no test command configured"
    assert backend.calls == []


def test_run_gate_pass() -> None:
    backend = FakeBackend(ExecResult(stdout="", stderr="", exit_code=0))
    result = run_gate(backend, VerifyConfig(test="pytest -q"), "/cwd")
    assert result == GateResult(ok=True, failures="")


def test_run_gate_failure_reports_tail() -> None:
    backend = FakeBackend(ExecResult(stdout="", stderr="AssertionError: boom", exit_code=1))
    result = run_gate(backend, VerifyConfig(test="pytest -q"), "/cwd")
    assert result.ok is False
    assert "AssertionError: boom" in result.failures


def test_run_gate_timeout_reported_as_failure() -> None:
    backend = FakeBackend(ExecResult(stdout="partial", stderr="", exit_code=0, timed_out=True))
    result = run_gate(backend, VerifyConfig(test="pytest -q"), "/cwd")
    assert result.ok is False
    assert "timed out" in result.failures.lower()


def test_run_gate_never_raises_on_backend_exception() -> None:
    backend = FakeBackend(raises=OSError("no shell"))
    result = run_gate(backend, VerifyConfig(test="pytest -q"), "/cwd")
    assert result.ok is False
    assert "no shell" in result.failures
