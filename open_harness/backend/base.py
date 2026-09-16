"""Backend contract: three primitives. All file tools derive from these.

Spec: open-harness-spec.md section 6.5.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExecResult:
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool = False


@dataclass
class FileOpResult:
    path: str
    ok: bool
    error: str | None = None


class Backend(ABC):
    """Where commands run and files live. The prototype ships LocalBackend only.

    `root` is the working directory. Backends do not enforce permissions; the
    policy engine does, before a tool is called. Backends MUST NOT be reachable
    by the model except through tools.
    """

    root: Path

    @abstractmethod
    def execute(self, cmd: str, *, timeout: float, env: dict[str, str] | None = None,
                cwd: Path | None = None) -> ExecResult: ...

    @abstractmethod
    def upload(self, files: dict[str, bytes]) -> list[FileOpResult]:
        """Write files. Keys are paths relative to root or absolute under root."""

    @abstractmethod
    def download(self, paths: list[str]) -> dict[str, bytes | FileOpResult]:
        """Read files. Missing or unreadable paths return a FileOpResult with error."""
