"""Bypass-immune safety checks.

Spec: open-harness-spec.md sections 8.2 (step 1g) and 8.5.

`immune_check` never depends on rules or mode: it is the one thing that runs
"in every mode" (spec 8.2). It flags:
  - dangerous directories/files (.git, .env, id_rsa, ...) anywhere in a
    touched path,
  - a path that resolves (following symlinks, dangling links resolved to
    their deepest existing ancestor) outside cwd and any additional_dirs,
  - Windows path tricks (ADS, 8.3 short names, device names, long-path /
    device prefixes, trailing dots/spaces, UNC), checked on every platform
    since the model's text is platform-independent,
  - for shell requests, both path-like tokens extracted heuristically from
    the command, and a set of command-injection / destructive-shell
    patterns in the command text itself.

`immune_check`'s signature in the task brief is `(req, cwd) -> Decision |
None`; `additional_dirs` is added as an optional keyword-only-by-convention
parameter (positionally after cwd, defaulting to None/empty) since the
containment check described in the brief needs it and PolicyContext carries
it. This is noted rather than silently deviating from the given signature:
callers that only pass `req, cwd` still work exactly as specified.
"""

from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path

from open_harness.policy.types import Decision, ToolCallRequest

DANGEROUS_DIRS = {
    ".git", ".claude", ".open-harness", ".vscode", ".idea", ".ssh", ".aws", ".gnupg",
}

DANGEROUS_FILES = {
    ".env", ".env.*", ".gitconfig", ".bashrc", ".zshrc", ".profile",
    ".npmrc", ".pypirc", "id_rsa*", "*.pem", "*.key", "open-harness.toml",
}

_DEVICE_NAMES = re.compile(
    r"^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(\..*)?$", re.IGNORECASE
)

_MAX_SYMLINK_HOPS = 40

_SHELL_DANGEROUS_SUBSTRINGS = {
    "$(": "command substitution $(...)",
    "`": "backtick command substitution",
    "${": "parameter expansion ${...}",
    "<(": "process substitution <(...)",
    ">(": "process substitution >(...)",
}


def immune_check(
    req: ToolCallRequest, cwd: str, additional_dirs: list[str] | None = None
) -> Decision | None:
    additional_dirs = additional_dirs or []
    candidates: list[str] = list(req.paths)
    if req.tool_name == "shell" and req.permission_content:
        candidates.extend(_extract_shell_paths(req.permission_content))

    seen: set[str] = set()
    for raw in candidates:
        if raw in seen:
            continue
        seen.add(raw)

        # A read-only shell command that merely names a protected directory
        # (`find . -path ./.git -prune`, `ls .git`) cannot modify it; only the
        # protected-file check (secrets) still applies to it. Writes and file
        # tools keep the full check.
        read_only_shell = req.tool_name == "shell" and req.is_read_only
        reason = _dangerous_name_reason(raw, check_dirs=not read_only_shell)
        if reason:
            return Decision(
                behavior="ask",
                reason=f"touches a protected path: {reason} ({raw})",
                immune=True,
            )

        win_reason = _windows_suspicious(raw)
        if win_reason:
            return Decision(
                behavior="ask",
                reason=f"suspicious Windows path pattern: {win_reason} ({raw})",
                immune=True,
            )

        # Containment applies to file tools only. A shell command legitimately
        # names programs outside the project (interpreters, venvs, anything on
        # PATH), so for shell we check dangerous names and patterns above but
        # not containment. Where a shell command writes is governed by rules.
        if req.tool_name != "shell" and _escapes_roots(raw, cwd, additional_dirs):
            return Decision(
                behavior="ask",
                reason=f"path resolves outside the working directory: {raw}",
                immune=True,
            )

    if req.tool_name == "shell" and req.permission_content:
        shell_reason = _shell_suspicious_reason(req.permission_content)
        if shell_reason:
            return Decision(
                behavior="ask",
                reason=f"suspicious shell pattern: {shell_reason}",
                immune=True,
            )

    return None


# --- dangerous names ---------------------------------------------------


def _dangerous_name_reason(raw: str, *, check_dirs: bool = True) -> str | None:
    parts = [p for p in re.split(r"[\\/]+", raw) if p]
    for part in parts:
        if check_dirs and part.lower() in DANGEROUS_DIRS:
            return f"protected directory '{part}'"
    basename = parts[-1] if parts else raw
    for pattern in DANGEROUS_FILES:
        if fnmatch.fnmatchcase(basename.lower(), pattern.lower()):
            return f"protected file '{basename}'"
    return None


# --- Windows path tricks (checked on every platform) --------------------


def _windows_suspicious(raw: str) -> str | None:
    s = raw
    low = s.lower()

    if low.startswith(("\\\\?\\", "\\\\.\\")):
        return "long-path or device prefix (\\\\?\\ or \\\\.\\)"

    if low.startswith("\\\\") or (low.startswith("//") and low.count("/") >= 3):
        return "UNC path"

    drive_colon = len(s) >= 2 and s[1] == ":" and s[0].isalpha()
    colon_positions = [i for i, c in enumerate(s) if c == ":"]
    extra_colons = [i for i in colon_positions if not (drive_colon and i == 1)]
    # `::` is never an NTFS stream separator; it is a pytest node id
    # (tests/x.py::test_y) or a C++/Rust scope. Only a single colon inside a
    # filename component denotes an alternate data stream.
    if extra_colons and "::" not in s:
        return "alternate data stream (':' in path)"

    if re.search(r"~\d", s):
        return "8.3 short filename"

    parts = [p for p in re.split(r"[\\/]+", s) if p and p not in (".", "..")]
    for part in parts:
        if _DEVICE_NAMES.match(part):
            return f"reserved device name '{part}'"
        if part[-1] in (" ", "."):
            return f"trailing dot or space in '{part}'"

    if "..." in s:
        return "3+ consecutive dots"

    return None


# --- containment ----------------------------------------------------------


def _escapes_roots(raw: str, cwd: str, additional_dirs: list[str]) -> bool:
    target = Path(raw)
    if not target.is_absolute():
        target = Path(cwd) / target
    resolved = _resolve_symlink_chain(target)

    roots = [Path(cwd)] + [Path(d) for d in additional_dirs]
    resolved_posix = resolved.as_posix().lower()
    for root in roots:
        try:
            root_resolved = root.resolve(strict=False)
        except OSError:
            root_resolved = root
        root_posix = root_resolved.as_posix().lower()
        if resolved_posix == root_posix or resolved_posix.startswith(root_posix + "/"):
            return False
    return True


def _resolve_symlink_chain(path: Path, max_hops: int = _MAX_SYMLINK_HOPS) -> Path:
    """Follow a symlink chain up to `max_hops`. A dangling target resolves to
    its deepest existing ancestor."""
    current = path
    hops = 0
    while hops < max_hops:
        try:
            if current.is_symlink():
                target = Path(os.readlink(current))
                current = target if target.is_absolute() else (current.parent / target)
                hops += 1
                continue
        except OSError:
            pass
        break

    node = current
    try:
        if node.exists():
            return node.resolve(strict=False)
    except OSError:
        pass
    while not node.exists() and node.parent != node:
        node = node.parent
    try:
        return node.resolve(strict=False)
    except OSError:
        return node


# --- shell heuristics -------------------------------------------------------


def _extract_shell_paths(cmd: str) -> list[str]:
    tokens = re.findall(r"[^\s]+", cmd)
    paths = []
    for t in tokens:
        stripped = t.strip("'\"")
        if not stripped:
            continue
        if "/" in stripped or "\\" in stripped or stripped.startswith("~"):
            paths.append(stripped)
        elif stripped.startswith(".") and stripped not in (".", ".."):
            # Bare dotfiles (`cat .env`, `ls .git`) are paths too; without
            # this a secret read through the shell never reached the check.
            paths.append(stripped)
        elif any(fnmatch.fnmatchcase(stripped.lower(), p.lower()) for p in DANGEROUS_FILES):
            paths.append(stripped)
    return paths


def _shell_suspicious_reason(cmd: str) -> str | None:
    for token, reason in _SHELL_DANGEROUS_SUBSTRINGS.items():
        if token in cmd:
            return reason
    low = cmd.lower()
    if re.search(r"\beval\b", low):
        return "eval"
    if re.search(r"\bsudo\b", low):
        return "sudo"
    if re.search(r"\bchmod\b", low):
        return "chmod"
    if _has_rm_rf(low):
        return "rm -rf"
    if re.search(r"\b(curl|wget)\b[^|]*\|\s*(sudo\s+)?(sh|bash|zsh)\b", low):
        return "curl|sh pipe-to-shell"
    return None


def _has_rm_rf(low_cmd: str) -> bool:
    """Detect `rm` invoked with both recursive and force, in any flag
    arrangement: `-rf`, `-fr`, `-r -f`, `-R --force`, etc."""
    tokens = re.findall(r"\S+", low_cmd)
    for i, tok in enumerate(tokens):
        base = tok.rsplit("/", 1)[-1]
        if base != "rm":
            continue
        rest = tokens[i + 1 :]
        combined = any(
            re.fullmatch(r"-[a-z]*r[a-z]*f[a-z]*", t) or re.fullmatch(r"-[a-z]*f[a-z]*r[a-z]*", t)
            for t in rest
        )
        if combined:
            return True
        has_r = any(t in ("-r", "-R", "--recursive") or re.fullmatch(r"-[a-z]*r[a-z]*", t) for t in rest)
        has_f = any(t in ("-f", "--force") or re.fullmatch(r"-[a-z]*f[a-z]*", t) for t in rest)
        if has_r and has_f:
            return True
    return False
