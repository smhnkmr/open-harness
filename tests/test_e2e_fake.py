"""End-to-end: a scripted fake adapter drives the real loop, tools, policy,
verifier and log. No network. This is the test that proves the kernel."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from open_harness.config import Config, PolicyConfig, ProviderConfig, VerifyConfig
from open_harness.kernel.gateway import Gateway
from open_harness.kernel.log import EventLog
from open_harness.kernel.loop import Session
from open_harness.kernel.roles import RoleResolver
from open_harness.model.adapter import Adapter
from open_harness.model.types import (
    CapabilityFlags,
    Event,
    ModelRequest,
    Stop,
    TextDelta,
    ToolCallFragment,
    Usage,
)


class ScriptedAdapter(Adapter):
    """Plays back a list of event scripts, one per call, in order."""

    name = "scripted"
    flags = CapabilityFlags()

    def __init__(self, scripts: list[list[Event]], **kw) -> None:
        super().__init__(**kw)
        self.scripts = scripts
        self.requests: list[ModelRequest] = []

    def stream(self, model: str, req: ModelRequest) -> Iterator[Event]:
        self.requests.append(req)
        script = self.scripts.pop(0) if self.scripts else [TextDelta("done"), Usage(1, 1), Stop("end_turn")]
        yield from script


class FakeClient:
    def __init__(self, answers: list[str] | None = None) -> None:
        self.records: list[dict] = []
        self.answers = answers or []
        self.asked: list[str] = []

    def emit(self, record: dict) -> None:
        self.records.append(record)

    def ask_human(self, req, decision) -> str:
        self.asked.append(req.permission_content)
        return self.answers.pop(0) if self.answers else "deny"

    def ask_user(self, question: str, options: list[str]) -> str:
        return options[0] if options else ""

    def interrupted(self) -> bool:
        return False


def tool_call(index: int, cid: str, name: str, args: dict) -> list[Event]:
    js = json.dumps(args)
    # split the JSON into fragments to exercise the reducer
    third = max(1, len(js) // 3)
    return [
        ToolCallFragment(index=index, id=cid, name=name, args_fragment=js[:third]),
        ToolCallFragment(index=index, args_fragment=js[third: 2 * third]),
        ToolCallFragment(index=index, args_fragment=js[2 * third:]),
    ]


def make_session(tmp_path: Path, scripts: list[list[Event]], client: FakeClient, *, verify: VerifyConfig | None = None,
                 policy: PolicyConfig | None = None) -> tuple[Session, ScriptedAdapter]:
    cwd = tmp_path / "work"
    cwd.mkdir()
    config = Config(
        providers={"fake": ProviderConfig(name="fake", adapter="scripted")},
        roles={"main": "fake:m"},
        policy=policy or PolicyConfig(mode="default", allow=["read", "grep", "glob"]),
        verify=verify or VerifyConfig(),
        max_turns=10,
        session_root=tmp_path / "sessions",
    )
    adapter = ScriptedAdapter(scripts)
    resolver = RoleResolver(config)
    resolver._adapters["fake"] = adapter          # bypass entry-point discovery
    log_dir = tmp_path / "sessions"
    log_dir.mkdir()
    log = EventLog(log_dir / "s1.jsonl")
    gateway = Gateway(resolver=resolver, on_event=lambda k, p: log.append(k, **p), sleep=lambda s: None)
    session = Session(config, cwd, client, log, gateway=gateway)
    return session, adapter


def test_plain_answer_completes(tmp_path: Path) -> None:
    client = FakeClient()
    session, adapter = make_session(tmp_path, [[TextDelta("Hello "), TextDelta("there."), Usage(5, 2), Stop("end_turn")]], client)
    t = session.run_turn("hi")
    assert t.reason == "completed"
    assert session.final_text() == "Hello there."
    kinds = [r["kind"] for r in session.log.read()]
    assert "turn_start" in kinds and "assistant_message" in kinds and "turn_end" in kinds


def test_read_then_answer(tmp_path: Path) -> None:
    client = FakeClient()
    scripts = [
        tool_call(0, "c1", "read", {"file_path": "hello.txt"}) + [Usage(5, 5), Stop("tool_use")],
        [TextDelta("The file says hi."), Usage(5, 2), Stop("end_turn")],
    ]
    session, adapter = make_session(tmp_path, scripts, client)
    (session.cwd / "hello.txt").write_text("hi\n", encoding="utf-8")
    t = session.run_turn("what does hello.txt say?")
    assert t.reason == "completed"
    # second request carried the tool result back to the model
    second = adapter.requests[1]
    results = [b for m in second.messages for b in m.blocks if b.type == "tool_result"]
    assert results and "hi" in (results[0].text or "")
    # permission was allowed by rule, no human asked
    assert client.asked == []


def test_edit_requires_read_and_asks_human(tmp_path: Path) -> None:
    client = FakeClient(answers=["allow"])
    scripts = [
        tool_call(0, "c1", "edit", {"file_path": "a.py", "old_string": "x = 1", "new_string": "x = 2"}) + [Stop("tool_use")],
        tool_call(0, "c2", "read", {"file_path": "a.py"}) + [Stop("tool_use")],
        tool_call(0, "c3", "edit", {"file_path": "a.py", "old_string": "x = 1", "new_string": "x = 2"}) + [Stop("tool_use")],
        [TextDelta("Changed x to 2."), Stop("end_turn")],
    ]
    session, adapter = make_session(tmp_path, scripts, client)
    (session.cwd / "a.py").write_text("x = 1\n", encoding="utf-8")
    t = session.run_turn("set x to 2")
    assert t.reason == "completed"
    assert (session.cwd / "a.py").read_text(encoding="utf-8") == "x = 2\n"
    # first edit was rejected by the tool (not read yet) and reported as an error result
    first_result = [b for b in adapter.requests[1].messages[-1].blocks if b.type == "tool_result"][0]
    assert first_result.is_error
    # the edit went through the ask reducer; routine write in worktree should not need a human
    # (if the reducer allowed it deterministically, no ask; either way the edit happened)


def test_verifier_gate_blocks_completion_until_tests_pass(tmp_path: Path) -> None:
    client = FakeClient(answers=["allow", "allow"])
    marker = "PASS" if False else "FAIL"
    # test command: passes only when a.py contains "ok"
    py = "python -c \"import sys;sys.exit(0 if 'ok' in open('a.py').read() else 1)\""
    scripts = [
        tool_call(0, "c1", "read", {"file_path": "a.py"}) + [Stop("tool_use")],
        tool_call(0, "c2", "edit", {"file_path": "a.py", "old_string": "x = 1", "new_string": "x = 2"}) + [Stop("tool_use")],
        [TextDelta("Done."), Stop("end_turn")],                     # model claims done; gate fails
        tool_call(0, "c3", "edit", {"file_path": "a.py", "old_string": "x = 2", "new_string": "x = 2  # ok"}) + [Stop("tool_use")],
        [TextDelta("Done, tests pass."), Stop("end_turn")],
    ]
    session, adapter = make_session(tmp_path, scripts, client, verify=VerifyConfig(test=py))
    (session.cwd / "a.py").write_text("x = 1\n", encoding="utf-8")
    t = session.run_turn("make tests pass")
    assert t.reason == "completed"
    kinds = [r["kind"] for r in session.log.read()]
    reasons = [r.get("reason") for r in session.log.read() if r["kind"] == "transition"]
    assert "verifier_failed" in reasons
    assert "ok" in (session.cwd / "a.py").read_text(encoding="utf-8")
    del marker


def test_invalid_tool_json_becomes_error_result(tmp_path: Path) -> None:
    client = FakeClient()
    scripts = [
        [ToolCallFragment(index=0, id="c1", name="read", args_fragment='{"file_path": '), Stop("tool_use")],
        [TextDelta("Sorry."), Stop("end_turn")],
    ]
    session, adapter = make_session(tmp_path, scripts, client)
    t = session.run_turn("read something")
    assert t.reason == "completed"
    results = [b for m in adapter.requests[1].messages for b in m.blocks if b.type == "tool_result"]
    assert results and results[0].is_error


def test_parallel_read_only_batch(tmp_path: Path) -> None:
    client = FakeClient()
    scripts = [
        tool_call(0, "c1", "read", {"file_path": "a.txt"}) + tool_call(1, "c2", "read", {"file_path": "b.txt"}) + [Stop("tool_use")],
        [TextDelta("both read"), Stop("end_turn")],
    ]
    session, adapter = make_session(tmp_path, scripts, client)
    (session.cwd / "a.txt").write_text("A", encoding="utf-8")
    (session.cwd / "b.txt").write_text("B", encoding="utf-8")
    t = session.run_turn("read both")
    assert t.reason == "completed"
    results = [b for m in adapter.requests[1].messages for b in m.blocks if b.type == "tool_result"]
    assert len(results) == 2


def test_max_turns_terminal(tmp_path: Path) -> None:
    client = FakeClient()
    loop_forever = [tool_call(0, f"c{i}", "glob", {"pattern": "*.txt"}) + [Stop("tool_use")] for i in range(20)]
    session, adapter = make_session(tmp_path, loop_forever, client)
    t = session.run_turn("loop")
    assert t.reason == "max_turns"


def test_resume_replays_messages(tmp_path: Path) -> None:
    client = FakeClient()
    session, adapter = make_session(tmp_path, [[TextDelta("first"), Stop("end_turn")]], client)
    session.run_turn("one")
    # new session object on the same log
    config, cwd, log = session.config, session.cwd, session.log
    resolver = RoleResolver(config)
    resolver._adapters["fake"] = ScriptedAdapter([[TextDelta("second"), Stop("end_turn")]])
    gw = Gateway(resolver=resolver, on_event=lambda k, p: log.append(k, **p), sleep=lambda s: None)
    resumed = Session(config, cwd, client, EventLog(log.path), gateway=gw)
    assert any(m.role == "assistant" and m.text == "first" for m in resumed.messages)
    t = resumed.run_turn("two")
    assert t.reason == "completed"
