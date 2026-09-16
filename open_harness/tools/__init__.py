"""Core tool set. Spec: open-harness-spec.md section 6.2.

`default_registry` registers read, edit, write, shell, grep, glob, ask_user
in that fixed order -- tool order is pinned (spec P1), the registry does not
sort.
"""

from __future__ import annotations

from typing import Any

from open_harness.tools.ask_user import AskUserTool
from open_harness.tools.base import ToolRegistry
from open_harness.tools.edit import EditTool
from open_harness.tools.glob import GlobTool
from open_harness.tools.grep import GrepTool
from open_harness.tools.read import ReadTool
from open_harness.tools.shell import ShellTool
from open_harness.tools.write import WriteTool

__all__ = [
    "AskUserTool",
    "EditTool",
    "GlobTool",
    "GrepTool",
    "ReadTool",
    "ShellTool",
    "WriteTool",
    "default_registry",
]


def default_registry(ctx_hint: Any = None) -> ToolRegistry:
    """Build the core tool set in the pinned order: read, edit, write,
    shell, grep, glob, ask_user. `ctx_hint` is accepted for forward
    compatibility (e.g. future backend-aware tool construction) but unused
    by any tool in this prototype."""
    registry = ToolRegistry()
    registry.register(ReadTool())
    registry.register(EditTool())
    registry.register(WriteTool())
    registry.register(ShellTool())
    registry.register(GrepTool())
    registry.register(GlobTool())
    registry.register(AskUserTool())
    return registry
