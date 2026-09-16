"""Drivers that invoke each harness's CLI, one subprocess per task turn, and
turn the raw transcripts into a `RunRecord`.

Neither `run_oh` nor `run_cc` calls `task.check` -- the eval driver applies
the checker itself once the run is over, using `RunRecord.final_texts`
(`passed` is always left `False` here).

`ParsedTurn` is the common shape both `evals.parse_cc.parse_cc_stream` and
`evals.parse_oh.parse_oh_log` return for a single turn; `run_oh`/`run_cc` sum
its fields across turns into the run's `RunRecord`.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from evals.pricing import cost_usd, cost_usd_by_model
from evals.types import RunRecord, TokenUsage

if TYPE_CHECKING:
    from evals.types import TaskSpec


@dataclass
class ParsedTurn:
    """Metrics extracted from a single turn's transcript (one CLI call)."""

    api_calls: int = 0
    tool_calls: int = 0
    tokens: TokenUsage = field(default_factory=TokenUsage)
    usage_by_model: dict[str, TokenUsage] = field(default_factory=dict)
    reported_cost_usd: float | None = None
    denials: int = 0
    verifier_failures: int = 0
    turns: int = 0
    final_text: str = ""
    is_error: bool = False
    error_text: str = ""


# Imported after ParsedTurn is defined: both modules import `ParsedTurn` back
# from this module (lazily, inside their functions), so this module must
# finish defining it before triggering their import.
from evals.parse_cc import parse_cc_stream
from evals.parse_oh import parse_oh_log

# Subdirectory of `run_dir` that holds the open-harness session log for a run
# (tests rely on this exact name to locate/seed the session file).
OH_SESSIONS_DIRNAME = "oh_sessions"

_OH_CONFIG_TEMPLATE = """\
max_turns = 60
session_root = "__SESSION_ROOT__"

[providers.anthropic]
adapter = "anthropic"

[roles]
main = "anthropic:__MODEL__"
compactor = "anthropic:claude-haiku-4-5"

[policy]
mode = "accept_edits"
allow = ["read", "grep", "glob", "shell(python*)", "shell(pytest*)", "shell(ruff*)", \
"shell(__HARNESS_PYTHON__*)", "shell(\\"__HARNESS_PYTHON__\\"*)", \
"shell(git status*)", "shell(git diff*)", "shell(git log*)", \
"shell(ls*)", "shell(cat*)", "shell(find*)", "shell(wc*)"]

[verify]
lint = "__HARNESS_PYTHON__ -m ruff check {file}"
test = "__HARNESS_PYTHON__ -m pytest -q -x"
"""


# Mirrors the shell allow rules in _OH_CONFIG_TEMPLATE so both harnesses can
# run the same commands without a human. Anything else is denied on both.
CC_ALLOWED_TOOLS = (
    "Bash(python *),Bash(python.exe *),Bash(pytest *),Bash(ruff *),Bash(git status *),"
    "Bash(git diff *),Bash(git log *),Bash(ls *),Bash(cat *),Bash(find *),Bash(wc *),"
    "Bash(cd *),Bash(export *)"
)


def cc_allowed_tools(harness_python: str | None) -> str:
    """CC_ALLOWED_TOOLS plus the project interpreter by absolute path (bare
    and double-quoted, as the model writes it), mirroring the
    `shell(<harness_python>*)` rule open-harness gets."""
    if not harness_python:
        return CC_ALLOWED_TOOLS
    py = _posix(harness_python)
    return f'{CC_ALLOWED_TOOLS},Bash({py} *),Bash("{py}" *)'


def _posix(path: Path | str) -> str:
    return str(path).replace("\\", "/")


def _write_oh_config(run_dir: Path, *, model: str, harness_python: str, session_root: Path) -> Path:
    content = (
        _OH_CONFIG_TEMPLATE.replace("__SESSION_ROOT__", _posix(session_root))
        .replace("__MODEL__", model)
        .replace("__HARNESS_PYTHON__", _posix(harness_python))
    )
    config_path = run_dir / "oh_config.toml"
    config_path.write_text(content, encoding="utf-8")
    return config_path


def _merge_usage(record: RunRecord, parsed: ParsedTurn) -> None:
    record.api_calls += parsed.api_calls
    record.tool_calls += parsed.tool_calls
    record.tokens.input += parsed.tokens.input
    record.tokens.output += parsed.tokens.output
    record.tokens.cache_read += parsed.tokens.cache_read
    record.tokens.cache_write += parsed.tokens.cache_write
    for model_name, usage in parsed.usage_by_model.items():
        slot = record.usage_by_model.setdefault(
            model_name, {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
        )
        slot["input"] += usage.input
        slot["output"] += usage.output
        slot["cache_read"] += usage.cache_read
        slot["cache_write"] += usage.cache_write
    record.denials += parsed.denials
    record.verifier_failures += parsed.verifier_failures
    record.turns += parsed.turns


def _finish_cost(record: RunRecord, model: str) -> None:
    if record.usage_by_model:
        record.cost_usd = cost_usd_by_model(record.usage_by_model)
    else:
        record.cost_usd = cost_usd(model, record.tokens)


def _pad_final_texts(final_texts: list[str], n_prompts: int) -> None:
    while len(final_texts) < n_prompts:
        final_texts.append("")


def run_oh(
    task: TaskSpec,
    workspace: Path,
    model: str,
    run_index: int,
    *,
    run_dir: Path,
    harness_python: str,
    repo_root: Path,
    timeout_s: int | None = None,
    runner=subprocess.run,
) -> RunRecord:
    """Run `task` against open-harness in one-shot (`-p`) mode, one subprocess
    per prompt in `task.prompts`, resuming the same session for later turns.

    `runner` defaults to `subprocess.run`; tests inject a fake to avoid
    spawning real processes / making model calls.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(repo_root)
    workspace = Path(workspace)

    session_root = run_dir / OH_SESSIONS_DIRNAME
    config_path = _write_oh_config(
        run_dir, model=model, harness_python=harness_python, session_root=session_root
    )

    env_src = repo_root / ".env"
    if env_src.exists():
        shutil.copyfile(env_src, run_dir / ".env")  # never read/print its contents

    record = RunRecord(task_id=task.id, harness="oh", model=model, run_index=run_index, passed=False)
    session_log_path: Path | None = None

    for i, prompt in enumerate(task.prompts):
        session_id = session_log_path.stem if session_log_path is not None else None
        argv = [
            str(harness_python), "-m", "open_harness",
            "--config", str(config_path),
            "--cwd", str(workspace),
            "-p", prompt,
            "--output-format", "stream-json",
            "--client", "stdio",
        ]
        if session_id is not None:
            argv += ["--resume", session_id]

        start = time.monotonic()
        try:
            result = runner(
                argv,
                cwd=str(repo_root),  # neutral cwd: never the workspace
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_s or task.timeout_s,
            )
        except subprocess.TimeoutExpired:
            record.wall_s += time.monotonic() - start
            record.error = f"turn {i + 1} timed out after {timeout_s or task.timeout_s}s"
            record.final_texts.append("")
            break
        except Exception as exc:  # noqa: BLE001 - surface any launch failure as a run error
            record.wall_s += time.monotonic() - start
            record.error = f"turn {i + 1} crashed launching open-harness: {exc}"
            record.final_texts.append("")
            break
        record.wall_s += time.monotonic() - start

        (run_dir / f"turn{i + 1}.stdout.jsonl").write_text(result.stdout or "", encoding="utf-8")
        (run_dir / f"turn{i + 1}.stderr.txt").write_text(result.stderr or "", encoding="utf-8")

        if result.returncode != 0:
            record.error = f"turn {i + 1} exited {result.returncode}: {(result.stderr or '')[:500]}"
            record.final_texts.append("")
            break

        session_files = sorted(session_root.glob("*.jsonl")) if session_root.exists() else []
        if not session_files:
            record.error = f"turn {i + 1}: no session log found under {session_root}"
            record.final_texts.append("")
            break
        session_log_path = session_files[0]
        record.session_ref = str(session_log_path)

        parsed = parse_oh_log(session_log_path, turn_index=i)
        _merge_usage(record, parsed)
        record.final_texts.append(parsed.final_text)
        if parsed.is_error and record.error is None:
            record.error = parsed.error_text

    _pad_final_texts(record.final_texts, len(task.prompts))
    _finish_cost(record, model)
    return record


def run_cc(
    task: TaskSpec,
    workspace: Path,
    model: str,
    run_index: int,
    *,
    run_dir: Path,
    timeout_s: int | None = None,
    harness_python: str | None = None,
    runner=subprocess.run,
) -> RunRecord:
    """Run `task` against the Claude Code CLI, one subprocess per prompt in
    `task.prompts`, resuming the same `--session-id` for later turns.

    `runner` defaults to `subprocess.run`; tests inject a fake to avoid
    spawning real processes / making model calls.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    workspace = Path(workspace)

    session_id = str(uuid.uuid4())
    env = os.environ.copy()
    env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"

    record = RunRecord(task_id=task.id, harness="cc", model=model, run_index=run_index, passed=False)
    record.session_ref = session_id
    reported_cost_total = 0.0
    have_reported_cost = False

    for i, prompt in enumerate(task.prompts):
        argv = [
            "claude", "-p", prompt,
            "--output-format", "stream-json",
            "--verbose",
            "--model", model,
            "--permission-mode", "acceptEdits",
            "--allowedTools", cc_allowed_tools(harness_python),
            "--strict-mcp-config",
            "--mcp-config", json.dumps({"mcpServers": {}}),
        ]
        argv += ["--session-id", session_id] if i == 0 else ["--resume", session_id]

        start = time.monotonic()
        try:
            result = runner(
                argv,
                cwd=str(workspace),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_s or task.timeout_s,
                env=env,
            )
        except subprocess.TimeoutExpired:
            record.wall_s += time.monotonic() - start
            record.error = f"turn {i + 1} timed out after {timeout_s or task.timeout_s}s"
            record.final_texts.append("")
            break
        except Exception as exc:  # noqa: BLE001 - surface any launch failure as a run error
            record.wall_s += time.monotonic() - start
            record.error = f"turn {i + 1} crashed launching claude: {exc}"
            record.final_texts.append("")
            break
        record.wall_s += time.monotonic() - start

        (run_dir / f"turn{i + 1}.stdout.jsonl").write_text(result.stdout or "", encoding="utf-8")
        (run_dir / f"turn{i + 1}.stderr.txt").write_text(result.stderr or "", encoding="utf-8")

        if result.returncode != 0:
            record.error = f"turn {i + 1} exited {result.returncode}: {(result.stderr or '')[:500]}"
            record.final_texts.append("")
            break

        parsed = parse_cc_stream((result.stdout or "").splitlines())
        _merge_usage(record, parsed)
        record.final_texts.append(parsed.final_text)
        if parsed.reported_cost_usd is not None:
            reported_cost_total += parsed.reported_cost_usd
            have_reported_cost = True
        if parsed.is_error and record.error is None:
            record.error = parsed.error_text

    _pad_final_texts(record.final_texts, len(task.prompts))
    record.reported_cost_usd = reported_cost_total if have_reported_cost else None
    _finish_cost(record, model)
    return record
