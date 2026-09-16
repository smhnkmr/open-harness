"""The policy decision pipeline.

Spec: open-harness-spec.md section 8.2.

The task brief adapts the spec's 8.2 order for this prototype, since
`ToolCallRequest` (unlike a real `Tool`) carries no `check_permissions` or
tool-implementation-denial hook:

    1a whole-tool deny rule                             -> deny
    1b whole-tool ask rule                               -> ask
    1c content-specific deny rule                        -> deny
    1d reserved (tool_name == "__denied__" placeholder)  -> deny
    1e tool requires user interaction (ask_user)          -> ask   IMMUNE
    1f content-specific ask rule                          -> ask   IMMUNE
    1g immune_check (dotfiles, .git, secrets, escape,
       Windows tricks, dangerous shell patterns)          -> ask   IMMUNE
    2a mode is bypass (or plan with bypass_available)     -> allow
    2a' mode accept_edits, tool in (edit, write), all
        paths under cwd                                   -> allow
    2a'' read-only request outside plan-restricted mode   -> allow
         (in plan mode: edit/write/shell that is not
         read-only is denied with reason "plan mode")
    2b whole-tool allow rule or content allow rule        -> allow
    3  passthrough                                        -> ask, with a
                                                              suggested rule

`dont_ask` converts any resulting `ask` into a `deny`, applied once after the
whole pipeline above has produced its verdict (this happens even for IMMUNE
asks: `dont_ask` never turns an immune ask into a silent `allow`, it can only
make it stricter).

Ambiguity noted (not resolved by changing the contract): steps 1a-1c run
unconditionally before the mode-based steps (2a etc.), so an explicitly
configured deny/ask rule is never bypassed by `mode == "bypass"` -- only the
"no explicit rule matched" passthrough path is affected by bypass mode. This
reading follows the literal step order given, and is consistent with 8.2's
"a deny anywhere wins".
"""

from __future__ import annotations

from open_harness.policy.rules import matches, path_under, shell_fully_allowed, suggest_rule
from open_harness.policy.safety import immune_check
from open_harness.policy.types import Decision, PolicyContext, Rule, ToolCallRequest


def decide(req: ToolCallRequest, ctx: PolicyContext) -> Decision:
    decision = _pipeline(req, ctx)
    if decision.behavior == "ask" and ctx.mode == "dont_ask":
        return Decision(
            behavior="deny",
            reason=f"dont_ask: converted ask to deny ({decision.reason})",
            immune=decision.immune,
            suggested_rule=decision.suggested_rule,
            step=decision.step,
        )
    return decision


def _pipeline(req: ToolCallRequest, ctx: PolicyContext) -> Decision:
    rules = ctx.rules
    cwd = ctx.cwd

    def whole(behavior: str) -> Rule | None:
        for r in rules:
            if r.behavior == behavior and r.whole_tool and matches(r, req, cwd):
                return r
        return None

    def content(behavior: str) -> Rule | None:
        for r in rules:
            if r.behavior == behavior and not r.whole_tool and matches(r, req, cwd):
                return r
        return None

    # 1a whole-tool deny rule
    r = whole("deny")
    if r is not None:
        return Decision(behavior="deny", reason=f"denied by rule '{_fmt(r)}'", step="1a")

    # 1b whole-tool ask rule
    r = whole("ask")
    if r is not None:
        return Decision(behavior="ask", reason=f"ask rule '{_fmt(r)}'", step="1b")

    # 1c content-specific deny rule
    r = content("deny")
    if r is not None:
        return Decision(behavior="deny", reason=f"denied by rule '{_fmt(r)}'", step="1c")

    # 1d reserved placeholder for tool-implementation denial (spec 8.2 1d).
    # ToolCallRequest carries no tool-implementation hook in this prototype,
    # so this is only reachable via the documented sentinel tool name.
    if req.tool_name == "__denied__":
        return Decision(behavior="deny", reason="tool implementation denied", step="1d")

    # 1e tool requires user interaction
    if req.tool_name == "ask_user":
        return Decision(
            behavior="ask", reason="tool requires user interaction", immune=True, step="1e"
        )

    # 1f content-specific ask rule
    r = content("ask")
    if r is not None:
        return Decision(behavior="ask", reason=f"ask rule '{_fmt(r)}'", immune=True, step="1f")

    # 1g safety check: dotfiles, .git, .claude, secrets, path escape, Windows tricks
    immune = immune_check(req, cwd, ctx.additional_dirs)
    if immune is not None:
        immune.step = "1g"
        return immune

    # 2a mode bypass (or plan with bypass available)
    if ctx.mode == "bypass":
        return Decision(behavior="allow", reason="bypass mode", step="2a")
    if ctx.mode == "plan" and ctx.bypass_available:
        return Decision(behavior="allow", reason="plan mode with bypass available", step="2a")

    # 2a' accept_edits: routine writes within the worktree
    if ctx.mode == "accept_edits" and req.tool_name in ("edit", "write") and _all_paths_under(
        req, cwd
    ):
        return Decision(behavior="allow", reason="accept_edits: write within worktree", step="2a'")

    # 2a'' read-only passthrough / plan-mode restriction
    if ctx.mode == "plan":
        if req.is_read_only:
            return Decision(behavior="allow", reason="read-only in plan mode", step="2a''")
        if req.tool_name in ("edit", "write", "shell"):
            return Decision(behavior="deny", reason="plan mode", step="2a''")
    elif req.is_read_only:
        return Decision(behavior="allow", reason="read-only request", step="2a''")

    # 2b whole-tool allow rule or content allow rule
    if req.tool_name == "shell":
        allow_rules = [r for r in rules if r.behavior == "allow" and r.tool == "shell"]
        if allow_rules and shell_fully_allowed(req, cwd, allow_rules):
            return Decision(behavior="allow", reason="allow rule covers shell command", step="2b")
    else:
        r = whole("allow") or content("allow")
        if r is not None:
            return Decision(behavior="allow", reason=f"allow rule '{_fmt(r)}'", step="2b")

    # 3 passthrough
    return Decision(
        behavior="ask",
        reason="passthrough: no rule matched",
        suggested_rule=suggest_rule(req),
        step="3",
    )


def _all_paths_under(req: ToolCallRequest, cwd: str) -> bool:
    paths = req.paths or ([req.permission_content] if req.permission_content else [])
    if not paths:
        return True
    return all(path_under(p, cwd) for p in paths)


def _fmt(rule: Rule) -> str:
    return rule.tool if rule.whole_tool else f"{rule.tool}({rule.content})"
