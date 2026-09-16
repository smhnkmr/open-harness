"""Internal helpers shared by the file tools. Not part of the public
contract in open_harness/tools/base.py -- just implementation plumbing.
"""

from __future__ import annotations

from pathlib import Path

from open_harness.tools.base import ToolContext

# Error codes surfaced (embedded in the message) by write/edit when the
# read-before-write invariant is violated. See SPEC.md 6.3.
NOT_READ = "NOT_READ"
CHANGED_ON_DISK = "CHANGED_ON_DISK"


def resolve_path(raw: str, cwd: Path) -> Path:
    """Resolve a possibly-relative path against cwd. Does not require the
    path to exist."""
    p = Path(raw)
    resolved = p if p.is_absolute() else (cwd / p)
    try:
        return resolved.resolve()
    except OSError:
        return resolved


def check_read_before_write(resolved: Path, ctx: ToolContext) -> str | None:
    """Enforce: an existing file must have been read this session, with an
    unchanged mtime/size, before it may be written or edited. Returns an
    error code, or None if the write/edit may proceed (including the case
    where the file does not exist yet)."""
    if not resolved.exists():
        return None
    record = ctx.read_state.get(str(resolved))
    if record is None:
        return NOT_READ
    try:
        stat = resolved.stat()
    except OSError:
        return NOT_READ
    if stat.st_mtime_ns != record.mtime_ns or stat.st_size != record.size:
        return CHANGED_ON_DISK
    return None
