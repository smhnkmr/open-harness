"""CLI smoke: drive `main()` end to end over the stdio protocol with a
scripted adapter. Proves config loading, session creation, the loop, tools,
policy, the approval round trip, and JSON output all wire together."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest

from open_harness.clients import stdio as cli
from open_harness.kernel import roles as roles_mod
from open_harness.model.types import Stop, TextDelta, ToolCallFragment, Usage
from tests.test_e2e_fake import ScriptedAdapter, tool_call

CONFIG = """
[providers.fake]
adapter = "scripted"

[roles]
main = "fake:m"

[policy]
mode = "default"
allow = ["read", "glob"]

max_turns = 8
session_root = "{root}"
"""


@pytest.fixture
def project(tmp_path: Path, monkeypatch):
    cwd = tmp_path / "proj"
    cwd.mkdir()
    (cwd / "hello.txt").write_text("hello world\n", encoding="utf-8")
    cfg = tmp_path / "open-harness.toml"
    cfg.write_text(CONFIG.format(root=(tmp_path / "sessions").as_posix()), encoding="utf-8")
    return cwd, cfg


def install_scripted(monkeypatch, scripts):
    adapter = ScriptedAdapter(scripts)
    monkeypatch.setattr(roles_mod, "build_adapter", lambda provider: adapter)
    return adapter


def test_prompt_mode_prints_final_text(project, monkeypatch, capsys):
    cwd, cfg = project
    install_scripted(monkeypatch, [
        tool_call(0, "c1", "read", {"file_path": "hello.txt"}) + [Usage(3, 3), Stop("tool_use")],
        [TextDelta("It says hello world."), Usage(3, 3), Stop("end_turn")],
    ])
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    rc = cli.main(["--config", str(cfg), "--cwd", str(cwd), "-p", "what is in hello.txt?"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "It says hello world." in out


def test_stream_json_approval_round_trip(project, monkeypatch, capsys):
    cwd, cfg = project
    install_scripted(monkeypatch, [
        tool_call(0, "c1", "shell", {"command": "mkdir made_by_agent"}) + [Stop("tool_use")],
        [TextDelta("ran it"), Stop("end_turn")],
    ])
    # scripted stdin: one turn, then an approval for whatever request arrives, then shutdown.
    # The request id is unknown ahead of time, so the client must accept an approve op
    # carrying request_id "*" or the client resolves by order; we send both forms.
    ops = [
        {"op": "turn_input", "text": "make a directory"},
        {"op": "approve", "request_id": "*", "behavior": "allow"},
        {"op": "shutdown"},
    ]
    monkeypatch.setattr(sys, "stdin", io.StringIO("\n".join(json.dumps(o) for o in ops) + "\n"))
    rc = cli.main(["--config", str(cfg), "--cwd", str(cwd), "--output-format", "stream-json"])
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]
    kinds = [rec.get("kind") for rec in lines]
    assert rc == 0
    assert "approval_request" in kinds, kinds
    assert "result" in kinds, kinds
    result = [rec for rec in lines if rec.get("kind") == "result"][-1]
    assert result["reason"] == "completed"
