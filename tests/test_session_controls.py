"""Session-level controls used by every client's slash commands: add_rule,
set_mode, usage_totals, and resume re-applying rule_added/mode_changed from
the log. These already exist in kernel/loop.py, so this file is green now --
no terminal client required. Kept separate from tests/test_terminal.py (which
importorskips the not-yet-implemented terminal client) so it always runs."""

from __future__ import annotations

from pathlib import Path

from open_harness.kernel.gateway import Gateway
from open_harness.kernel.log import EventLog
from open_harness.kernel.loop import Session
from open_harness.kernel.roles import RoleResolver
from open_harness.model.types import Stop, TextDelta, Usage
from tests.test_e2e_fake import FakeClient, ScriptedAdapter, make_session


def test_add_rule_appends_to_rules_and_logs(tmp_path: Path) -> None:
    client = FakeClient()
    session, _ = make_session(tmp_path, [], client)
    before = len(session.rules)

    rule = session.add_rule("shell(mkdir *)", "allow")

    assert rule.tool == "shell"
    assert rule.content == "mkdir *"
    assert rule.behavior == "allow"
    assert rule.source == "session"
    assert len(session.rules) == before + 1
    assert session.rules[-1] is rule

    recs = [r for r in session.log.read() if r["kind"] == "rule_added"]
    assert len(recs) == 1
    assert recs[0]["rule"] == "shell(mkdir *)"
    assert recs[0]["behavior"] == "allow"


def test_add_rule_defaults_behavior_to_allow(tmp_path: Path) -> None:
    client = FakeClient()
    session, _ = make_session(tmp_path, [], client)

    rule = session.add_rule("read(secrets/**)")

    assert rule.behavior == "allow"


def test_set_mode_updates_session_and_logs(tmp_path: Path) -> None:
    client = FakeClient()
    session, _ = make_session(tmp_path, [], client)
    assert session.mode == "default"

    session.set_mode("plan")

    assert session.mode == "plan"
    recs = [r for r in session.log.read() if r["kind"] == "mode_changed"]
    assert len(recs) == 1
    assert recs[0]["mode"] == "plan"


def test_usage_totals_sums_across_turns(tmp_path: Path) -> None:
    client = FakeClient()
    scripts = [
        [TextDelta("hi"), Usage(5, 2), Stop("end_turn")],
        [TextDelta("again"), Usage(3, 1), Stop("end_turn")],
    ]
    session, _ = make_session(tmp_path, scripts, client)

    session.run_turn("first")
    session.run_turn("second")
    totals = session.usage_totals()

    assert totals == {"input": 8, "output": 3, "cache_read": 0, "cache_write": 0}


def test_usage_totals_empty_log_is_all_zero(tmp_path: Path) -> None:
    client = FakeClient()
    session, _ = make_session(tmp_path, [], client)

    assert session.usage_totals() == {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}


def test_resume_reapplies_rule_added_and_mode_changed(tmp_path: Path) -> None:
    client = FakeClient()
    session, _ = make_session(tmp_path, [], client)

    session.add_rule("shell(mkdir *)", "allow")
    session.set_mode("accept_edits")

    config, cwd, log = session.config, session.cwd, session.log
    resolver = RoleResolver(config)
    resolver._adapters["fake"] = ScriptedAdapter([])
    gateway = Gateway(resolver=resolver, on_event=lambda k, p: log.append(k, **p), sleep=lambda s: None)

    resumed = Session(config, cwd, client, EventLog(log.path), gateway=gateway)

    assert resumed.mode == "accept_edits"
    session_rules = [r for r in resumed.rules if r.source == "session"]
    assert len(session_rules) == 1
    assert session_rules[0].tool == "shell"
    assert session_rules[0].content == "mkdir *"
    assert session_rules[0].behavior == "allow"
