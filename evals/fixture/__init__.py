"""Builds disposable, git-tracked copies of the eval template workspace.

Both harnesses under test are run against a workspace produced by
`build_workspace`, so they see byte-identical starting points (including an
identical git history: a single "fixture" commit at HEAD), which is what
lets task checkers compare "what changed" against `git`.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

TEMPLATE_DIR = Path(__file__).parent / "template"


def template_dir() -> Path:
    """Return the path to the synthetic project template."""
    return TEMPLATE_DIR


INSTRUCTIONS = """# Project notes

- Run the tests with `{python} -m pytest -q` and lint with `{python} -m ruff check .`.
  That interpreter has pytest and ruff installed; the system `python` may not.
- Source lives under `src/ledger`, tests under `tests`.
"""


def build_workspace(
    dest: Path,
    *,
    mutate: Callable[[Path], None] | None = None,
    python: str | None = None,
) -> Path:
    """Copy the template into `dest`, optionally mutate it, then commit it.

    `dest`'s parent directories are created if needed. `dest` itself must
    not already exist and be non-empty. If `mutate` is given, it is called
    with `dest` after the copy and before the commit, so its edits (e.g.
    planting a bug, or removing some tests) become part of the fixture
    commit both harnesses start from. If `python` is given, an AGENTS.md and
    an identical CLAUDE.md naming that interpreter are committed too.
    """
    dest = Path(dest)
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError(f"destination is non-empty: {dest}")
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copytree(TEMPLATE_DIR, dest, dirs_exist_ok=True)

    if python is not None:
        # Both harnesses read project instructions (AGENTS.md for open-harness,
        # CLAUDE.md for Claude Code); identical content keeps the start fair.
        text = INSTRUCTIONS.format(python=str(python).replace("\\", "/"))
        (dest / "AGENTS.md").write_text(text, encoding="utf-8")
        (dest / "CLAUDE.md").write_text(text, encoding="utf-8")

    if mutate is not None:
        mutate(dest)

    _git(dest, "init", "-q")
    _git(dest, "add", "-A")
    _git(
        dest,
        "-c",
        "user.name=eval",
        "-c",
        "user.email=eval@example.com",
        "commit",
        "-q",
        "-m",
        "fixture",
    )
    return dest


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        stdin=subprocess.DEVNULL,
        capture_output=True,
    )
