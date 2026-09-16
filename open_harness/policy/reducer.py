"""The ask reducer (spec 8.4). Stage 1 (deterministic policy) and stage 3
(human) only in this prototype; stage 2 (classifier role) is wired as an
optional hook a caller may supply.

    1 deterministic policy   routine writes in the worktree, fixed-repo
                              commands                       -> allow
    2 classifier role        typed decision, run with a 20s timeout in a
                              thread; None or an exception falls through
                              (never auto-approve on classifier failure)
    3 human                  never silently approves; `allow_always` also
                              returns a rule to persist

Stage 1 only fires for non-immune asks. An immune ask (spec 8.2: dotfiles,
path escapes, secrets, dangerous shell patterns, ...) is exactly the class of
thing that must never be silently approved even when it happens to also look
like "a write under cwd" or "a fixed-repo command" -- so both stage-1 checks
below require `not decision.immune`. This is slightly stricter than the
literal task brief, which only spells out the `not immune` guard for the
write case; extending it to the shell case is a deliberate, documented
choice, not a contract change (nothing here changes `Decision`, `Behavior`
or any frozen type).
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from open_harness.policy.rules import path_under, split_compound, suggest_rule
from open_harness.policy.types import Behavior, Decision, PolicyContext, ToolCallRequest

Classifier = Callable[[ToolCallRequest, Decision], "Decision | None"]
Human = Callable[[ToolCallRequest, Decision], Behavior]

CLASSIFIER_TIMEOUT_SECONDS = 20.0

FIXED_REPO_COMMANDS = (
    "git status",
    "git diff",
    "git log",
    "git show",
    "ls",
    "cat",
    "pwd",
    "npm test",
    "pytest",
    "ruff",
    "mypy",
    "make test",
)


def reduce_ask(
    req: ToolCallRequest,
    decision: Decision,
    ctx: PolicyContext,
    *,
    classifier: Classifier | None = None,
    human: Human,
) -> Decision:
    if decision.behavior != "ask":
        return decision

    # Stage 1: deterministic policy, no model call.
    if not decision.immune:
        if req.tool_name in ("edit", "write") and _all_paths_under(req, ctx.cwd):
            return Decision(
                behavior="allow", reason="routine write in worktree", step="reduce:1"
            )
        if req.tool_name == "shell":
            segments = split_compound(req.permission_content)
            if segments and all(_is_fixed_repo_command(s) for s in segments):
                return Decision(behavior="allow", reason="fixed repo command", step="reduce:1")

    # Stage 2: classifier role, best-effort with a hard timeout.
    if classifier is not None:
        result = _run_classifier(classifier, req, decision)
        if result is not None:
            return result

    # Stage 3: human. Never silently approves.
    verdict = human(req, decision)
    if verdict == "allow":
        return Decision(behavior="allow", reason="approved by user", step="reduce:3")
    if verdict == "allow_always":
        suggested = decision.suggested_rule or suggest_rule(req)
        return Decision(
            behavior="allow",
            reason="approved by user (always)",
            suggested_rule=suggested,
            step="reduce:3",
        )
    if isinstance(verdict, str) and verdict.startswith("deny:") and verdict[5:].strip():
        # "deny:<message>": the message goes back to the model as the error result.
        return Decision(behavior="deny", reason=f"denied by user: {verdict[5:].strip()}", step="reduce:3")
    return Decision(behavior="deny", reason="denied by user", step="reduce:3")


def _all_paths_under(req: ToolCallRequest, cwd: str) -> bool:
    paths = req.paths or ([req.permission_content] if req.permission_content else [])
    if not paths:
        return True
    return all(path_under(p, cwd) for p in paths)


def _is_fixed_repo_command(segment: str) -> bool:
    seg = segment.strip()
    for cmd in FIXED_REPO_COMMANDS:
        if seg == cmd or seg.startswith(cmd + " "):
            return True
    return False


def _run_classifier(
    classifier: Classifier, req: ToolCallRequest, decision: Decision
) -> Decision | None:
    outcome: dict[str, Any] = {}

    def target() -> None:
        try:
            outcome["value"] = classifier(req, decision)
        except Exception as exc:  # noqa: BLE001 - classifier failure must never propagate
            outcome["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(CLASSIFIER_TIMEOUT_SECONDS)
    if thread.is_alive():
        return None  # timed out -> fall through to human
    if "error" in outcome:
        return None  # exception -> fall through to human
    return outcome.get("value")
