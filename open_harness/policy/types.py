"""Policy types shared by the engine, the ask reducer and the kernel.

Spec: open-harness-spec.md sections 8.2 to 8.4.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Mode = Literal["default", "accept_edits", "plan", "bypass", "dont_ask"]
Behavior = Literal["allow", "deny", "ask"]
Source = Literal["user", "project", "local", "flag", "managed", "cli", "session"]


@dataclass(frozen=True)
class Rule:
    """`ToolName` or `ToolName(content)`. Content is a prefix/glob for shell,
    gitignore-style for paths, `domain:x` for fetch, `mcp__server__tool` for MCP."""

    tool: str
    content: str | None
    behavior: Behavior
    source: Source

    @property
    def whole_tool(self) -> bool:
        return self.content is None


@dataclass
class Decision:
    behavior: Behavior
    reason: str
    immune: bool = False                     # survived bypass mode
    suggested_rule: str | None = None        # shown with an ask so the user can settle a class
    step: str = ""                           # which pipeline step decided (1a .. 3)


@dataclass
class ToolCallRequest:
    tool_name: str
    args: dict[str, Any]
    permission_content: str
    is_read_only: bool
    is_destructive: bool
    paths: list[str] = field(default_factory=list)   # any filesystem paths the call touches


@dataclass
class PolicyContext:
    mode: Mode
    cwd: str
    additional_dirs: list[str]
    rules: list[Rule]
    bypass_available: bool = False
