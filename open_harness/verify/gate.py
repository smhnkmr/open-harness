"""Verification gates: per-edit lint, pre-done test gate.

Spec: open-harness-spec.md section 7.

Both functions are backend-driven (spec 6.5: `Backend.execute`) and never
raise -- a command that cannot even be launched is reported as a failure,
not an exception, since verification must never crash the turn loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from open_harness.backend.base import Backend
from open_harness.config import VerifyConfig

_TAIL_LINES = 60
_LINT_TIMEOUT_SECONDS = 120.0
_TEST_TIMEOUT_SECONDS = 600.0


@dataclass
class GateResult:
    ok: bool
    failures: str = ""


def run_lint(backend: Backend, file: str, cfg: VerifyConfig, cwd: str | Path) -> str:
    """Run the configured linter on `file`. Returns "" on a clean (exit 0,
    not timed out) run, else "LINT ERRORS:\\n" followed by the last 60 lines
    of stderr (or stdout if stderr is empty)."""
    if not cfg.lint:
        return ""
    # The command runs through a shell (bash on Windows too); a backslash path
    # would be eaten as escapes. Use forward slashes and quote it.
    safe_file = str(file).replace("\\", "/")
    cmd = cfg.lint.replace("{file}", f'"{safe_file}"')
    try:
        result = backend.execute(cmd, timeout=_LINT_TIMEOUT_SECONDS, cwd=_as_path(cwd))
    except Exception as exc:  # noqa: BLE001 - verification must never raise
        return f"LINT ERRORS:\n(failed to run lint command: {exc})"
    if result.exit_code == 0 and not result.timed_out:
        return ""
    return "LINT ERRORS:\n" + _tail(result.stderr, result.stdout, result.timed_out)


def run_gate(backend: Backend, cfg: VerifyConfig, cwd: str | Path) -> GateResult:
    """Run the configured test command before a turn may end. `ok=True` with
    a note when no test command is configured; never raises."""
    if not cfg.test:
        return GateResult(ok=True, failures="no test command configured")
    try:
        result = backend.execute(cfg.test, timeout=_TEST_TIMEOUT_SECONDS, cwd=_as_path(cwd))
    except Exception as exc:  # noqa: BLE001 - verification must never raise
        return GateResult(ok=False, failures=f"failed to run test command: {exc}")
    if result.exit_code == 0 and not result.timed_out:
        return GateResult(ok=True, failures="")
    return GateResult(ok=False, failures=_tail(result.stderr, result.stdout, result.timed_out))


def _as_path(cwd: str | Path) -> Path | None:
    if cwd is None:
        return None
    return cwd if isinstance(cwd, Path) else Path(cwd)


def _tail(stderr: str, stdout: str, timed_out: bool) -> str:
    text = (stderr or "").strip() or (stdout or "").strip()
    if timed_out:
        text = (text + "\n" if text else "") + "(command timed out)"
    lines = text.splitlines()
    return "\n".join(lines[-_TAIL_LINES:])
