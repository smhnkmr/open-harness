"""Tests for evals.parse_cc, evals.parse_oh and evals.runners.

No live subprocesses or model calls: `run_oh`/`run_cc` are exercised with an
injected fake `runner` that returns canned output (and, for open-harness,
writes the session log a real invocation would have written), matching the
`runner=subprocess.run` injection point in their signatures.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from evals.parse_cc import parse_cc_stream
from evals.parse_oh import parse_oh_log
from evals.pricing import cost_usd, cost_usd_by_model
from evals.runners import OH_SESSIONS_DIRNAME, run_cc, run_oh
from evals.types import CheckResult, TaskSpec

FIXTURES = Path(__file__).parent / "fixtures" / "evals"

CC_LINES = (FIXTURES / "cc_turn.jsonl").read_text(encoding="utf-8").splitlines()
OH_LINES = (FIXTURES / "oh_session.jsonl").read_text(encoding="utf-8").splitlines()

# The fixture holds two turns (a fresh session, then one `--resume`); the
# first segment is lines 1-29 (session_start .. the first turn_end), the
# second is the remaining 22 lines (the second user_message .. its turn_end).
OH_TURN1_LINES = OH_LINES[:29]
OH_TURN2_LINES = OH_LINES[29:]

MODEL = "claude-sonnet-5"


def _task(n_prompts: int = 2, timeout_s: int = 60) -> TaskSpec:
    prompts = [f"prompt {i + 1}" for i in range(n_prompts)]
    return TaskSpec(
        id="demo-task",
        kind="qa",
        prompts=prompts,
        check=lambda workspace, texts: CheckResult(True),
        timeout_s=timeout_s,
    )


# --------------------------------------------------------------------------- parse_cc_stream


def test_parse_cc_stream_exact_counts() -> None:
    parsed = parse_cc_stream(CC_LINES)

    assert parsed.api_calls == 3  # distinct message ids: msg_A, msg_B, msg_C
    assert parsed.tool_calls == 2  # Read (msg_A), Bash (msg_B)
    assert parsed.tokens.input == 60  # 50 (sonnet) + 10 (haiku)
    assert parsed.tokens.output == 125  # 120 + 5
    assert parsed.tokens.cache_read == 200  # 200 + 0
    assert parsed.tokens.cache_write == 30  # 30 + 0
    assert parsed.reported_cost_usd == pytest.approx(0.0234)
    assert parsed.denials == 1
    assert parsed.verifier_failures == 0
    assert parsed.turns == 5
    assert parsed.final_text == "Done. Summary: explored the repo and ran the test suite."
    assert parsed.is_error is False
    assert parsed.error_text == ""


def test_parse_cc_stream_tolerates_malformed_lines() -> None:
    # The fixture already embeds a malformed JSON line and a blank line;
    # also throw in extra garbage to be sure nothing raises.
    lines = list(CC_LINES) + ["not json at all", "", "   ", "42", "[1, 2]"]
    parsed = parse_cc_stream(lines)
    assert parsed.api_calls == 3
    assert parsed.tool_calls == 2


def test_parse_cc_stream_no_result_record_is_error() -> None:
    lines = [line for line in CC_LINES if '"type": "result"' not in line]
    parsed = parse_cc_stream(lines)
    assert parsed.is_error is True
    assert parsed.error_text
    assert parsed.reported_cost_usd is None
    assert parsed.turns == 0


# --------------------------------------------------------------------------- parse_oh_log


def test_parse_oh_log_turn0_exact_counts(tmp_path: Path) -> None:
    log_path = tmp_path / "session.jsonl"
    log_path.write_text("\n".join(OH_LINES) + "\n", encoding="utf-8")

    parsed = parse_oh_log(log_path, turn_index=0)
    assert parsed.api_calls == 3
    assert parsed.tool_calls == 2
    assert parsed.tokens.input == 240
    assert parsed.tokens.output == 180
    assert parsed.tokens.cache_read == 20
    assert parsed.tokens.cache_write == 10
    assert parsed.denials == 1
    assert parsed.verifier_failures == 0
    assert parsed.turns == 3
    assert parsed.final_text == (
        "Implemented Session.set_role and the /model command; existing tests pass."
    )
    assert parsed.reported_cost_usd is None
    assert parsed.is_error is False


def test_parse_oh_log_turn1_exact_counts(tmp_path: Path) -> None:
    log_path = tmp_path / "session.jsonl"
    log_path.write_text("\n".join(OH_LINES) + "\n", encoding="utf-8")

    parsed = parse_oh_log(log_path, turn_index=1)
    assert parsed.api_calls == 3
    assert parsed.tool_calls == 1
    assert parsed.tokens.input == 75
    assert parsed.tokens.output == 490
    assert parsed.tokens.cache_read == 1460
    assert parsed.tokens.cache_write == 0
    assert parsed.denials == 0
    assert parsed.verifier_failures == 1
    assert parsed.turns == 3
    assert parsed.final_text == (
        "Fixed the lint issue; edge cases covered for invalid spec and unknown provider."
    )
    assert parsed.is_error is False


def test_parse_oh_log_turn_index_none_uses_last_segment(tmp_path: Path) -> None:
    log_path = tmp_path / "session.jsonl"
    log_path.write_text("\n".join(OH_LINES) + "\n", encoding="utf-8")

    default = parse_oh_log(log_path)
    explicit_last = parse_oh_log(log_path, turn_index=1)
    assert default == explicit_last


def test_parse_oh_log_per_turn_does_not_double_count(tmp_path: Path) -> None:
    """Re-parsing the (appended) whole-file log for turn 2 must not add
    turn 1's records again -- that's the whole point of segmenting on
    user_message/turn_end rather than reading the file cumulatively."""
    log_path = tmp_path / "session.jsonl"

    # Simulate only turn 1 having run so far.
    log_path.write_text("\n".join(OH_TURN1_LINES) + "\n", encoding="utf-8")
    turn0_only = parse_oh_log(log_path, turn_index=0)
    assert turn0_only.api_calls == 3
    assert turn0_only.tool_calls == 2

    # Now simulate the resume: the log gains turn 2's records (append-only).
    with log_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(OH_TURN2_LINES) + "\n")

    turn0_again = parse_oh_log(log_path, turn_index=0)
    turn1 = parse_oh_log(log_path, turn_index=1)

    # turn 1's numbers are unaffected by the file growing...
    assert turn0_again == turn0_only
    # ...and turn 2's numbers are turn 2's alone, not turn1+turn2.
    assert turn1.api_calls == 3
    assert turn1.tool_calls == 1
    assert turn1.tokens.input == 75


def test_parse_oh_log_missing_turn_index(tmp_path: Path) -> None:
    log_path = tmp_path / "session.jsonl"
    log_path.write_text("\n".join(OH_TURN1_LINES) + "\n", encoding="utf-8")
    parsed = parse_oh_log(log_path, turn_index=5)
    assert parsed.is_error is True
    assert parsed.error_text


def test_parse_oh_log_tolerates_malformed_lines(tmp_path: Path) -> None:
    log_path = tmp_path / "session.jsonl"
    lines = list(OH_TURN1_LINES)
    lines.insert(3, "{broken json")
    lines.insert(5, "")
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    parsed = parse_oh_log(log_path, turn_index=0)
    assert parsed.api_calls == 3
    assert parsed.tool_calls == 2


# --------------------------------------------------------------------------- run_oh


def _make_fake_oh_runner(session_root: Path, calls: list) -> callable:
    first_session_id = "11111111-1111-1111-1111-111111111111"

    def fake(argv, **kwargs):
        calls.append((list(argv), kwargs))
        if "--resume" in argv:
            session_id = argv[argv.index("--resume") + 1]
            path = session_root / f"{session_id}.jsonl"
            with path.open("a", encoding="utf-8") as f:
                f.write("\n".join(OH_TURN2_LINES) + "\n")
        else:
            session_root.mkdir(parents=True, exist_ok=True)
            path = session_root / f"{first_session_id}.jsonl"
            path.write_text("\n".join(OH_TURN1_LINES) + "\n", encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    return fake


def test_run_oh_happy_path(tmp_path: Path) -> None:
    task = _task(n_prompts=2)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    run_dir = tmp_path / "run"
    harness_python = "C:/fake/python.exe"

    calls: list = []
    session_root = run_dir / OH_SESSIONS_DIRNAME
    fake_runner = _make_fake_oh_runner(session_root, calls)

    record = run_oh(
        task, workspace, MODEL, 0,
        run_dir=run_dir, harness_python=harness_python, repo_root=repo_root,
        runner=fake_runner,
    )

    assert len(calls) == 2
    argv1, kwargs1 = calls[0]
    argv2, kwargs2 = calls[1]

    # Turn 1: no --resume, cwd is the neutral repo root (not the workspace).
    assert "--resume" not in argv1
    assert argv1[0] == harness_python
    assert "-p" in argv1 and argv1[argv1.index("-p") + 1] == "prompt 1"
    assert "--cwd" in argv1 and argv1[argv1.index("--cwd") + 1] == str(workspace)
    assert kwargs1["cwd"] == str(repo_root)
    assert kwargs1["stdin"] == subprocess.DEVNULL

    # Turn 2: resumes the session id discovered from turn 1's session_root.
    assert "--resume" in argv2
    session_id = argv2[argv2.index("--resume") + 1]
    assert session_id == "11111111-1111-1111-1111-111111111111"
    assert kwargs2["cwd"] == str(repo_root)

    # Config file lives in run_dir, never in the workspace.
    config_path = run_dir / "oh_config.toml"
    assert config_path.exists()
    assert not (workspace / "oh_config.toml").exists()
    config_text = config_path.read_text(encoding="utf-8")
    assert f'main = "anthropic:{MODEL}"' in config_text
    assert "C:/fake/python.exe" in config_text  # forward slashes, not backslashes
    assert "\\" not in config_text.split("session_root = ")[1].splitlines()[0]

    assert record.task_id == "demo-task"
    assert record.harness == "oh"
    assert record.passed is False
    assert record.error is None
    assert record.api_calls == 6
    assert record.tool_calls == 3
    assert record.tokens.input == 315
    assert record.tokens.output == 670
    assert record.tokens.cache_read == 1480
    assert record.tokens.cache_write == 10
    assert record.denials == 1
    assert record.verifier_failures == 1
    assert record.turns == 6
    assert record.reported_cost_usd is None
    assert record.usage_by_model
    assert record.cost_usd == pytest.approx(cost_usd_by_model(record.usage_by_model))
    assert record.final_texts == [
        "Implemented Session.set_role and the /model command; existing tests pass.",
        "Fixed the lint issue; edge cases covered for invalid spec and unknown provider.",
    ]
    assert record.session_ref == str(session_root / f"{session_id}.jsonl")


def test_run_oh_copies_env_file(tmp_path: Path) -> None:
    task = _task(n_prompts=1)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / ".env").write_text("FAKE_KEY=not-a-real-secret\n", encoding="utf-8")
    run_dir = tmp_path / "run"

    calls: list = []
    session_root = run_dir / OH_SESSIONS_DIRNAME
    fake_runner = _make_fake_oh_runner(session_root, calls)

    run_oh(
        task, workspace, MODEL, 0,
        run_dir=run_dir, harness_python="python", repo_root=repo_root,
        runner=fake_runner,
    )

    copied = run_dir / ".env"
    assert copied.exists()
    assert copied.read_text(encoding="utf-8") == "FAKE_KEY=not-a-real-secret\n"


def test_run_oh_timeout_stops_further_turns(tmp_path: Path) -> None:
    task = _task(n_prompts=2)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    run_dir = tmp_path / "run"

    calls: list = []

    def timing_out(argv, **kwargs):
        calls.append((list(argv), kwargs))
        raise subprocess.TimeoutExpired(cmd=argv, timeout=kwargs.get("timeout", 0))

    record = run_oh(
        task, workspace, MODEL, 0,
        run_dir=run_dir, harness_python="python", repo_root=repo_root,
        timeout_s=5, runner=timing_out,
    )

    assert len(calls) == 1  # second turn never attempted
    assert record.error is not None
    assert "timed out" in record.error
    assert record.final_texts == ["", ""]
    assert record.passed is False


# --------------------------------------------------------------------------- run_cc


def _make_fake_cc_runner(calls: list) -> callable:
    def fake(argv, **kwargs):
        calls.append((list(argv), kwargs))
        return subprocess.CompletedProcess(argv, 0, stdout="\n".join(CC_LINES) + "\n", stderr="")

    return fake


def test_run_cc_happy_path(tmp_path: Path) -> None:
    task = _task(n_prompts=2)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    run_dir = tmp_path / "run"

    calls: list = []
    record = run_cc(task, workspace, MODEL, 0, run_dir=run_dir, runner=_make_fake_cc_runner(calls))

    assert len(calls) == 2
    argv1, kwargs1 = calls[0]
    argv2, kwargs2 = calls[1]

    assert argv1[0] == "claude"
    assert "-p" in argv1 and argv1[argv1.index("-p") + 1] == "prompt 1"
    assert "--session-id" in argv1
    assert "--resume" not in argv1
    session_id = argv1[argv1.index("--session-id") + 1]

    assert "--resume" in argv2
    assert argv2[argv2.index("--resume") + 1] == session_id
    assert "--session-id" not in argv2

    for kwargs in (kwargs1, kwargs2):
        assert kwargs["cwd"] == str(workspace)
        assert kwargs["stdin"] == subprocess.DEVNULL
        assert kwargs["env"]["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"

    assert record.passed is False
    assert record.error is None
    assert record.session_ref == session_id
    assert record.api_calls == 6
    assert record.tool_calls == 4
    assert record.tokens.input == 120
    assert record.tokens.output == 250
    assert record.tokens.cache_read == 400
    assert record.tokens.cache_write == 60
    assert record.denials == 2
    assert record.verifier_failures == 0
    assert record.turns == 10
    assert record.reported_cost_usd == pytest.approx(0.0468)
    assert record.usage_by_model
    assert record.cost_usd == pytest.approx(cost_usd_by_model(record.usage_by_model))
    assert record.final_texts == [
        "Done. Summary: explored the repo and ran the test suite.",
        "Done. Summary: explored the repo and ran the test suite.",
    ]

    # Raw transcripts are saved per turn under run_dir.
    assert (run_dir / "turn1.stdout.jsonl").exists()
    assert (run_dir / "turn2.stdout.jsonl").exists()
    assert (run_dir / "turn1.stderr.txt").exists()


def test_run_cc_timeout_stops_further_turns(tmp_path: Path) -> None:
    task = _task(n_prompts=2)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    run_dir = tmp_path / "run"

    calls: list = []

    def timing_out(argv, **kwargs):
        calls.append((list(argv), kwargs))
        raise subprocess.TimeoutExpired(cmd=argv, timeout=kwargs.get("timeout", 0))

    record = run_cc(task, workspace, MODEL, 0, run_dir=run_dir, timeout_s=5, runner=timing_out)

    assert len(calls) == 1
    assert record.error is not None
    assert "timed out" in record.error
    assert record.final_texts == ["", ""]
    assert record.passed is False


def test_run_cc_passes_allowed_tools_mirroring_oh_rules(tmp_path):
    from evals.runners import CC_ALLOWED_TOOLS, run_cc
    from evals.types import CheckResult, TaskSpec

    seen = {}

    def fake_runner(argv, **kwargs):
        seen["argv"] = argv
        raise subprocess.TimeoutExpired(argv, 1)

    task = TaskSpec(id="t", kind="qa", prompts=["hi"], check=lambda w, t: CheckResult(True))
    run_cc(task, tmp_path / "ws", "claude-sonnet-5", 1, run_dir=tmp_path / "run", runner=fake_runner)
    argv = seen["argv"]
    assert argv[argv.index("--allowedTools") + 1] == CC_ALLOWED_TOOLS

    run_cc(task, tmp_path / "ws", "claude-sonnet-5", 1, run_dir=tmp_path / "run2",
           runner=fake_runner, harness_python=r"C:\x\.venv\Scripts\python.exe")
    allowed = seen["argv"][seen["argv"].index("--allowedTools") + 1]
    assert "Bash(C:/x/.venv/Scripts/python.exe *)" in allowed
    assert 'Bash("C:/x/.venv/Scripts/python.exe" *)' in allowed
    for prefix in ("Bash(python *)", "Bash(pytest *)", "Bash(ruff *)", "Bash(find *)"):
        assert prefix in CC_ALLOWED_TOOLS


def test_cc_cost_prices_each_model_at_its_own_rate():
    """Haiku side calls must not be priced at the Sonnet rate."""
    import json

    from evals.parse_cc import parse_cc_stream
    from evals.pricing import cost_usd_by_model
    from evals.types import TokenUsage

    result = {
        "type": "result", "total_cost_usd": 0.1, "num_turns": 1, "result": "ok",
        "modelUsage": {
            "claude-sonnet-5": {"inputTokens": 0, "outputTokens": 1000,
                                "cacheReadInputTokens": 0, "cacheCreationInputTokens": 0},
            "claude-haiku-4-5": {"inputTokens": 0, "outputTokens": 1000,
                                 "cacheReadInputTokens": 0, "cacheCreationInputTokens": 0},
        },
    }
    parsed = parse_cc_stream([json.dumps(result)])
    assert set(parsed.usage_by_model) == {"claude-sonnet-5", "claude-haiku-4-5"}
    per_model = cost_usd_by_model(parsed.usage_by_model)
    sonnet_only = cost_usd("claude-sonnet-5", TokenUsage(output=2000))
    assert per_model == pytest.approx(0.015 + 0.005)
    assert per_model < sonnet_only


def test_oh_instructions_message_belongs_to_first_turn(tmp_path):
    """AGENTS.md is pushed as a harness-tagged user message right before the
    prompt; the parser must not treat it as an empty turn of its own."""
    import json

    from evals.parse_oh import parse_oh_log

    recs = [
        {"kind": "session_start"},
        {"kind": "user_message", "message": {"role": "user", "blocks": [], "meta": {"harness": True}}},
        {"kind": "user_message", "message": {"role": "user", "blocks": [], "meta": {}}},
        {"kind": "usage", "spec": "anthropic:claude-sonnet-5", "input": 1, "output": 10,
         "cache_read": 0, "cache_write": 0},
        {"kind": "turn_end", "reason": "completed", "turns": 1},
    ]
    p = tmp_path / "s.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in recs), encoding="utf-8")
    parsed = parse_oh_log(p, turn_index=0)
    assert not parsed.is_error, parsed.error_text
    assert parsed.api_calls == 1
    assert parse_oh_log(p, turn_index=1).is_error


def test_oh_usage_split_by_model(tmp_path):
    import json

    from evals.parse_oh import parse_oh_log

    recs = [
        {"kind": "user_message", "message": {"role": "user", "blocks": []}},
        {"kind": "usage", "spec": "anthropic:claude-sonnet-5", "input": 1, "output": 10,
         "cache_read": 0, "cache_write": 0},
        {"kind": "usage", "spec": "anthropic:claude-haiku-4-5", "input": 2, "output": 20,
         "cache_read": 0, "cache_write": 0},
        {"kind": "turn_end", "reason": "completed", "turns": 2},
    ]
    p = tmp_path / "s.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in recs), encoding="utf-8")
    parsed = parse_oh_log(p)
    assert parsed.usage_by_model["claude-sonnet-5"].output == 10
    assert parsed.usage_by_model["claude-haiku-4-5"].output == 20
    assert parsed.tokens.output == 30
