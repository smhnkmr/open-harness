"""Tests for open_harness.policy (rules, safety, engine, reducer).

Spec: open-harness-spec.md sections 8.2 to 8.5.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from open_harness.policy.engine import decide
from open_harness.policy.reducer import FIXED_REPO_COMMANDS, reduce_ask
from open_harness.policy.rules import matches, parse_rule, split_compound
from open_harness.policy.safety import immune_check
from open_harness.policy.types import Decision, PolicyContext, Rule, ToolCallRequest


def req(
    tool_name: str,
    permission_content: str = "",
    *,
    args: dict | None = None,
    is_read_only: bool = False,
    is_destructive: bool = False,
    paths: list[str] | None = None,
) -> ToolCallRequest:
    return ToolCallRequest(
        tool_name=tool_name,
        args=args or {},
        permission_content=permission_content,
        is_read_only=is_read_only,
        is_destructive=is_destructive,
        paths=paths or [],
    )


def ctx(
    *,
    mode: str = "default",
    cwd: str,
    rules: list[Rule] | None = None,
    additional_dirs: list[str] | None = None,
    bypass_available: bool = False,
) -> PolicyContext:
    return PolicyContext(
        mode=mode,
        cwd=cwd,
        additional_dirs=additional_dirs or [],
        rules=rules or [],
        bypass_available=bypass_available,
    )


# --------------------------------------------------------------------- rules


def test_parse_whole_tool_rule() -> None:
    r = parse_rule("read", "allow", "user")
    assert r.tool == "read"
    assert r.content is None
    assert r.whole_tool


def test_parse_rule_with_content() -> None:
    r = parse_rule("shell(git *)", "allow", "project")
    assert r.tool == "shell"
    assert r.content == "git *"
    assert not r.whole_tool


def test_parse_rule_escaped_parens() -> None:
    r = parse_rule(r"shell(echo \(hi\))", "allow", "user")
    assert r.tool == "shell"
    assert r.content == "echo (hi)"


def test_parse_rule_escaped_parens_path() -> None:
    r = parse_rule(r"read(/weird \(dir\)/**)", "deny", "user")
    assert r.content == "/weird (dir)/**"


def test_split_compound_basic() -> None:
    assert split_compound("ls && git push") == ["ls", "git push"]
    assert split_compound("a; b | c || d") == ["a", "b", "c", "d"]


def test_split_compound_respects_quotes() -> None:
    segs = split_compound('echo "a && b" && ls')
    assert segs == ['echo "a && b"', "ls"]


def test_shell_prefix_rule_matches() -> None:
    rule = parse_rule("shell(git *)", "allow", "user")
    assert matches(rule, req("shell", "git status"), "/cwd")
    assert matches(rule, req("shell", "git push origin main"), "/cwd")
    assert not matches(rule, req("shell", "npm install"), "/cwd")


def test_shell_colon_star_prefix_rule() -> None:
    rule = parse_rule("shell(npm install:*)", "allow", "user")
    assert matches(rule, req("shell", "npm install foo"), "/cwd")
    assert not matches(rule, req("shell", "npm run build"), "/cwd")


def test_shell_exact_rule() -> None:
    rule = parse_rule("shell(pwd)", "allow", "user")
    assert matches(rule, req("shell", "pwd"), "/cwd")
    assert not matches(rule, req("shell", "pwd -L"), "/cwd")


def test_compound_deny_rule_triggers_on_any_segment() -> None:
    deny = parse_rule("shell(git push*)", "deny", "user")
    assert matches(deny, req("shell", "ls && git push"), "/cwd")


def test_compound_allow_does_not_cover_whole_command_via_single_rule() -> None:
    # matches() on a single allow rule against a compound command only
    # reports "any segment", not "every segment" -- see shell_fully_allowed
    # for the aggregate semantics used by the engine at step 2b.
    from open_harness.policy.rules import shell_fully_allowed

    allow_ls = parse_rule("shell(ls*)", "allow", "user")
    r = req("shell", "ls && git push")
    assert not shell_fully_allowed(r, "/cwd", [allow_ls])


def test_path_rule_home() -> None:
    rule = parse_rule("read(~/.zshrc)", "allow", "user")
    home = str(Path.home() / ".zshrc")
    assert matches(rule, req("read", paths=[home]), "/cwd")
    assert not matches(rule, req("read", paths=[str(Path.home() / ".bashrc")]), "/cwd")


def test_path_rule_leading_slash_relative_to_cwd(tmp_path: Path) -> None:
    rule = parse_rule("edit(/src/**)", "allow", "user")
    cwd = str(tmp_path)
    inside = str(tmp_path / "src" / "sub" / "foo.py")
    outside = str(tmp_path / "other" / "foo.py")
    assert matches(rule, req("edit", paths=[inside]), cwd)
    assert not matches(rule, req("edit", paths=[outside]), cwd)


def test_path_rule_double_slash_root(tmp_path: Path) -> None:
    rule = parse_rule("read(//etc/**)", "deny", "user")
    r = req("read", paths=["/etc/passwd"])
    assert matches(rule, r, str(tmp_path))


def test_path_rule_unrooted_matches_anywhere(tmp_path: Path) -> None:
    rule = parse_rule("read(*.env)", "deny", "user")
    nested = str(tmp_path / "a" / "b" / ".env")
    assert matches(rule, req("read", paths=[nested]), str(tmp_path))


def test_fetch_domain_suffix_match() -> None:
    rule = parse_rule("fetch(domain:example.com)", "allow", "user")
    assert matches(rule, req("fetch", args={"url": "https://api.example.com/x"}), "/cwd")
    assert matches(rule, req("fetch", args={"url": "https://example.com"}), "/cwd")
    assert not matches(rule, req("fetch", args={"url": "https://evil.com"}), "/cwd")


def test_mcp_server_prefix_and_exact() -> None:
    server_rule = parse_rule("mcp__github", "allow", "user")
    tool_rule = parse_rule("mcp__github__list_prs", "allow", "user")
    assert matches(server_rule, req("mcp__github__list_prs"), "/cwd")
    assert matches(server_rule, req("mcp__github"), "/cwd")
    assert not matches(server_rule, req("mcp__gitlab__list_prs"), "/cwd")
    assert matches(tool_rule, req("mcp__github__list_prs"), "/cwd")
    assert not matches(tool_rule, req("mcp__github__merge_pr"), "/cwd")


# -------------------------------------------------------------------- safety


def test_immune_env_file_is_ask_in_bypass(tmp_path: Path) -> None:
    r = req("read", paths=[str(tmp_path / ".env")])
    c = ctx(mode="bypass", cwd=str(tmp_path))
    d = decide(r, c)
    assert d.behavior == "ask"
    assert d.immune


def test_immune_dotgit_directory() -> None:
    d = immune_check(req("read", paths=["/repo/.git/config"]), "/repo")
    assert d is not None
    assert d.behavior == "ask" and d.immune


def test_immune_symlink_escaping_cwd(tmp_path: Path) -> None:
    work = tmp_path / "work"
    outside = tmp_path / "outside"
    work.mkdir()
    outside.mkdir()
    (outside / "secret.txt").write_text("s", encoding="utf-8")
    link = work / "link.txt"
    try:
        os.symlink(outside / "secret.txt", link)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation not permitted in this environment")
    d = immune_check(req("read", paths=[str(link)]), str(work))
    assert d is not None and d.immune


def test_immune_relative_traversal_escapes_cwd_without_symlinks(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    escaping = str(work / ".." / "elsewhere.txt")
    d = immune_check(req("read", paths=[escaping]), str(work))
    assert d is not None and d.immune


def test_immune_path_within_additional_dirs_not_flagged(tmp_path: Path) -> None:
    work = tmp_path / "work"
    extra = tmp_path / "extra"
    work.mkdir()
    extra.mkdir()
    target = str(extra / "file.txt")
    d = immune_check(req("read", paths=[target]), str(work), [str(extra)])
    assert d is None


def test_immune_windows_ads_pattern() -> None:
    d = immune_check(req("read", paths=["C:\\work\\file.txt:secret"]), "C:\\work")
    assert d is not None and d.immune


def test_immune_windows_device_name() -> None:
    d = immune_check(req("write", paths=["C:\\work\\CON"]), "C:\\work")
    assert d is not None and d.immune


def test_immune_shell_rm_rf() -> None:
    d = immune_check(req("shell", "rm -rf /tmp/build"), "/cwd")
    assert d is not None and d.immune


def test_immune_shell_command_substitution() -> None:
    d = immune_check(req("shell", "echo $(whoami)"), "/cwd")
    assert d is not None and d.immune


def test_non_dangerous_path_is_not_immune(tmp_path: Path) -> None:
    d = immune_check(req("read", paths=[str(tmp_path / "readme.md")]), str(tmp_path))
    assert d is None


# -------------------------------------------------------------------- engine


def test_deny_beats_allow(tmp_path: Path) -> None:
    rules = [
        parse_rule("shell(git *)", "allow", "user"),
        parse_rule("shell(git push*)", "deny", "user"),
    ]
    d = decide(req("shell", "git push origin main"), ctx(cwd=str(tmp_path), rules=rules))
    assert d.behavior == "deny"
    assert d.step == "1c"


def test_content_ask_rule_is_immune_even_in_bypass(tmp_path: Path) -> None:
    rules = [parse_rule("shell(git push*)", "ask", "user")]
    d = decide(req("shell", "git push origin main"), ctx(mode="bypass", cwd=str(tmp_path), rules=rules))
    assert d.behavior == "ask"
    assert d.immune
    assert d.step == "1f"


def test_plan_mode_denies_edit(tmp_path: Path) -> None:
    d = decide(
        req("edit", paths=[str(tmp_path / "a.py")]),
        ctx(mode="plan", cwd=str(tmp_path)),
    )
    assert d.behavior == "deny"
    assert d.reason == "plan mode"


def test_plan_mode_allows_read_only(tmp_path: Path) -> None:
    d = decide(
        req("read", paths=[str(tmp_path / "a.py")], is_read_only=True),
        ctx(mode="plan", cwd=str(tmp_path)),
    )
    assert d.behavior == "allow"


def test_accept_edits_allows_edit_in_cwd(tmp_path: Path) -> None:
    target = str(tmp_path / "a.py")
    d = decide(req("edit", paths=[target]), ctx(mode="accept_edits", cwd=str(tmp_path)))
    assert d.behavior == "allow"
    assert d.step == "2a'"


def test_accept_edits_does_not_allow_edit_outside_cwd(tmp_path: Path) -> None:
    outside = str(tmp_path.parent / "elsewhere.py")
    d = decide(req("edit", paths=[outside]), ctx(mode="accept_edits", cwd=str(tmp_path)))
    assert d.behavior != "allow" or d.step != "2a'"


def test_passthrough_suggests_rule(tmp_path: Path) -> None:
    d = decide(req("shell", "docker ps -a"), ctx(cwd=str(tmp_path)))
    assert d.behavior == "ask"
    assert d.step == "3"
    assert d.suggested_rule == "shell(docker ps *)"


def test_dont_ask_converts_ask_to_deny(tmp_path: Path) -> None:
    d = decide(req("shell", "docker ps -a"), ctx(mode="dont_ask", cwd=str(tmp_path)))
    assert d.behavior == "deny"


def test_dont_ask_never_silently_allows_immune(tmp_path: Path) -> None:
    d = decide(req("read", paths=[str(tmp_path / ".env")]), ctx(mode="dont_ask", cwd=str(tmp_path)))
    assert d.behavior == "deny"
    assert d.immune


def test_bypass_allows_passthrough(tmp_path: Path) -> None:
    d = decide(req("shell", "docker ps -a"), ctx(mode="bypass", cwd=str(tmp_path)))
    assert d.behavior == "allow"
    assert d.step == "2a"


# ------------------------------------------------------------------- reducer


def test_reducer_routine_write_allowed_without_human(tmp_path: Path) -> None:
    r = req("edit", paths=[str(tmp_path / "a.py")])
    c = ctx(cwd=str(tmp_path))
    decision = Decision(behavior="ask", reason="passthrough", step="3")
    called = {"human": False}

    def human(_req, _decision):
        called["human"] = True
        return "deny"

    out = reduce_ask(r, decision, c, human=human)
    assert out.behavior == "allow"
    assert not called["human"]


def test_reducer_fixed_repo_command_allowed(tmp_path: Path) -> None:
    r = req("shell", "git status --short")
    c = ctx(cwd=str(tmp_path))
    decision = Decision(behavior="ask", reason="passthrough", step="3")
    out = reduce_ask(r, decision, c, human=lambda *_: "deny")
    assert out.behavior == "allow"


def test_reducer_unknown_command_goes_to_human(tmp_path: Path) -> None:
    r = req("shell", "docker ps -a")
    c = ctx(cwd=str(tmp_path))
    decision = Decision(behavior="ask", reason="passthrough", step="3")
    seen = {}

    def human(req_, decision_):
        seen["called"] = True
        return "deny"

    out = reduce_ask(r, decision, c, human=human)
    assert seen.get("called")
    assert out.behavior == "deny"


def test_reducer_human_allow_always_returns_suggested_rule(tmp_path: Path) -> None:
    r = req("shell", "docker ps -a")
    c = ctx(cwd=str(tmp_path))
    decision = Decision(behavior="ask", reason="passthrough", suggested_rule="shell(docker ps *)", step="3")
    out = reduce_ask(r, decision, c, human=lambda *_: "allow_always")
    assert out.behavior == "allow"
    assert out.suggested_rule == "shell(docker ps *)"


def test_reducer_classifier_exception_falls_to_human(tmp_path: Path) -> None:
    r = req("shell", "docker ps -a")
    c = ctx(cwd=str(tmp_path))
    decision = Decision(behavior="ask", reason="passthrough", step="3")

    def bad_classifier(_req, _decision):
        raise RuntimeError("boom")

    out = reduce_ask(r, decision, c, classifier=bad_classifier, human=lambda *_: "allow")
    assert out.behavior == "allow"
    assert out.reason == "approved by user"


def test_reducer_classifier_result_used_when_it_returns_decision(tmp_path: Path) -> None:
    r = req("shell", "docker ps -a")
    c = ctx(cwd=str(tmp_path))
    decision = Decision(behavior="ask", reason="passthrough", step="3")

    def classifier(_req, _decision):
        return Decision(behavior="deny", reason="classified as risky", step="classifier")

    out = reduce_ask(r, decision, c, classifier=classifier, human=lambda *_: "allow")
    assert out.behavior == "deny"
    assert out.reason == "classified as risky"


def test_reducer_immune_ask_never_auto_allowed_at_stage1(tmp_path: Path) -> None:
    r = req("edit", paths=[str(tmp_path / "a.py")])
    c = ctx(cwd=str(tmp_path))
    decision = Decision(behavior="ask", reason="dangerous", immune=True, step="1g")
    out = reduce_ask(r, decision, c, human=lambda *_: "deny")
    assert out.behavior == "deny"


def test_fixed_repo_commands_set_contains_git_status() -> None:
    assert "git status" in FIXED_REPO_COMMANDS


def test_shell_command_naming_program_outside_cwd_is_not_immune(tmp_path: Path) -> None:
    """A shell command may name an interpreter outside the project (spec 8.5 applies
    containment to file tools only). Regression for the live run that denied
    `<venv>/python.exe -m pytest` as a protected path."""
    from open_harness.policy.safety import immune_check

    exe = "C:/somewhere/else/.venv/Scripts/python.exe" if str(tmp_path).startswith("C:") else "/usr/local/bin/python3"
    req = ToolCallRequest(tool_name="shell", args={"command": f"{exe} -m pytest -q"},
                          permission_content=f"{exe} -m pytest -q", is_read_only=False, is_destructive=False)
    assert immune_check(req, str(tmp_path)) is None


def test_read_outside_cwd_is_still_immune(tmp_path: Path) -> None:
    from open_harness.policy.safety import immune_check

    outside = str(tmp_path.parent / "elsewhere.txt")
    req = ToolCallRequest(tool_name="read", args={"file_path": outside}, permission_content=outside,
                          is_read_only=True, is_destructive=False, paths=[outside])
    d = immune_check(req, str(tmp_path / "proj"))
    assert d is not None and d.immune


def test_compound_allowed_when_extra_segments_are_read_only_filters() -> None:
    """`<allowed cmd> 2>&1 | tail -40` needs no rule for `tail` (regression from the
    live comparison run where Claude Code allowed the same shape)."""
    from open_harness.policy.rules import shell_fully_allowed

    rules = [parse_rule("shell(python*)", "allow", "user")]
    ok = ToolCallRequest(tool_name="shell", args={}, permission_content="python -m pytest -q 2>&1 | tail -40",
                         is_read_only=False, is_destructive=False)
    bad = ToolCallRequest(tool_name="shell", args={}, permission_content="python -m pytest -q && rm -rf build",
                          is_read_only=False, is_destructive=False)
    assert shell_fully_allowed(ok, "C:/tmp", rules)
    assert not shell_fully_allowed(bad, "C:/tmp", rules)


def test_pytest_node_id_is_not_an_alternate_data_stream() -> None:
    from open_harness.policy.safety import immune_check

    cmd = "python -m pytest tests/test_terminal.py::test_handle_slash_version -v"
    req = ToolCallRequest(tool_name="shell", args={"command": cmd}, permission_content=cmd,
                          is_read_only=False, is_destructive=False)
    assert immune_check(req, "C:/tmp") is None


def test_dot_relative_path_is_not_a_trailing_dot_pattern() -> None:
    from open_harness.policy.safety import immune_check

    p = "./open_harness/kernel/loop.py"
    req = ToolCallRequest(tool_name="read", args={"file_path": p}, permission_content=p,
                          is_read_only=True, is_destructive=False, paths=[p])
    assert immune_check(req, os.getcwd()) is None


def test_read_only_shell_may_name_protected_directory():
    """A read-only `find` that prunes .git cannot modify it; the safety check
    must not send it to a human. Found by the eval driver (t01, open-harness
    denied `find . -path ./.git -prune ... | head`). Writes still ask."""
    from open_harness.policy.safety import immune_check
    from open_harness.policy.types import ToolCallRequest

    cmd = "find . -path ./.venv -prune -o -path ./.git -prune -o -type f -print | head -200"
    ro = ToolCallRequest(tool_name="shell", args={"command": cmd}, permission_content=cmd,
                         is_read_only=True, is_destructive=False, paths=[])
    assert immune_check(ro, "/proj") is None

    write = "rm -rf .git"
    rw = ToolCallRequest(tool_name="shell", args={"command": write}, permission_content=write,
                         is_read_only=False, is_destructive=True, paths=[])
    assert immune_check(rw, "/proj") is not None

    # Secrets stay protected even for read-only commands.
    leak = "cat .env"
    ro_leak = ToolCallRequest(tool_name="shell", args={"command": leak}, permission_content=leak,
                              is_read_only=True, is_destructive=False, paths=[])
    assert immune_check(ro_leak, "/proj") is not None
