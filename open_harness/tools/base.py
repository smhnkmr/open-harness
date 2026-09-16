"""Tool contract and result shape.

Spec: open-harness-spec.md sections 6.1 and 6.4.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from open_harness.backend.base import Backend
from open_harness.model.types import Block, ToolSpec

DEFAULT_MAX_RESULT_CHARS = 50_000
PREVIEW_BYTES = 2_000
AGGREGATE_MAX_CHARS = 200_000


@dataclass
class ToolContext:
    """What a tool sees at call time. Built by the kernel per turn."""

    cwd: Path
    session_dir: Path                     # where tool-results/ and the log live
    backend: Backend
    read_state: dict[str, ReadRecord]   # path -> record of last read (for edit enforcement)
    is_main_thread: bool = True
    abort: Callable[[], bool] = lambda: False
    ask_user: Callable[[str, list[str]], str] | None = None   # ask_user tool uses this


@dataclass
class ReadRecord:
    mtime_ns: int
    size: int
    content_hash: str
    full_read: bool


@dataclass
class ToolResult:
    """Returned by Tool.call. `content` is text or blocks; `data` is the raw
    structured payload kept in the log; `persisted_path` is set by the kernel
    when content was written to disk."""

    content: str | list[Block]
    is_error: bool = False
    data: Any = None
    persisted_path: str | None = None


@dataclass
class ValidationResult:
    ok: bool
    message: str = ""


class Tool(ABC):
    name: str = "abstract"
    description: str = ""
    search_hint: str = ""
    defer: bool = False
    max_result_chars: int | float = DEFAULT_MAX_RESULT_CHARS

    @property
    @abstractmethod
    def input_schema(self) -> dict[str, Any]:
        """OpenAI-shaped JSON schema for the tool's parameters."""

    def prompt(self) -> str:
        """Long-form documentation, injected only when the tool is loaded."""
        return self.description

    def is_read_only(self, args: dict[str, Any]) -> bool:
        return False

    def is_concurrency_safe(self, args: dict[str, Any]) -> bool:
        return self.is_read_only(args)

    def is_destructive(self, args: dict[str, Any]) -> bool:
        return False

    def validate(self, args: dict[str, Any], ctx: ToolContext) -> ValidationResult:
        return ValidationResult(ok=True)

    def permission_content(self, args: dict[str, Any]) -> str:
        """The string that permission rules match against, e.g. the shell
        command, or the file path. Default: JSON of args."""
        import json
        return json.dumps(args, sort_keys=True)

    @abstractmethod
    def call(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult: ...

    def spec(self) -> ToolSpec:
        return ToolSpec(name=self.name, description=self.description,
                        json_schema=self.input_schema, defer=self.defer)


@dataclass
class ToolRegistry:
    tools: dict[str, Tool] = field(default_factory=dict)

    def register(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self.tools.get(name)

    def specs(self, *, include_deferred: bool = False) -> list[ToolSpec]:
        # Order is pinned by registration order. Do not sort. (spec P1)
        return [t.spec() for t in self.tools.values() if include_deferred or not t.defer]
