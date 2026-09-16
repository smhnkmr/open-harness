"""Rule parsing and matching.

Spec: open-harness-spec.md section 8.3.

Grammar: `ToolName` (whole-tool rule) or `ToolName(content)` (content rule).
Parens inside content are escaped as `\\(` `\\)`.

Content syntax by tool:
  shell            prefix match if it ends with `*` or `:*` (stripped before
                    comparing); otherwise exact match; an embedded `*` that is
                    not a trailing marker is a glob matched with fnmatch
                    against the whole command string.
  read/edit/write/glob/grep
                    gitignore-style path pattern. `~/x` anchors at the user's
                    home directory. A leading single `/` anchors at cwd --
                    this is a prototype simplification of the real settings-
                    file-relative semantics described in spec 8.3, noted here
                    rather than in the settings file itself since there is no
                    settings-file path concept in this prototype. A leading
                    `//` anchors at the filesystem root. Anything else is
                    unrooted and matches the pattern anywhere in the path.
  fetch             `domain:example.com` matches the request's host or any
                    subdomain of it.
  mcp__server__tool / mcp__server
                    whole-tool rules (no parens) that match an exact MCP tool
                    name or any tool on that server.

Compound shell commands (segments split on `&&`, `||`, `;`, `|` outside
quotes) are handled at two levels: `matches()` on a single rule reports
whether that rule's content matches *any* segment (this is what deny/ask
checks need: one bad segment is enough to trigger). `shell_fully_allowed()`
reports whether *every* segment is covered by *some* rule in a pool of allow
rules (segments may be covered by different rules) -- this is what an allow
determination for a compound command needs, and it cannot be expressed by a
single call to `matches()` since it spans the whole rule pool.
"""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path
from urllib.parse import urlparse

from open_harness.policy.types import Behavior, Rule, Source, ToolCallRequest

PATH_TOOLS = {"read", "edit", "write", "glob", "grep"}

_COMPOUND_OPS_2 = ("&&", "||")
_COMPOUND_OPS_1 = (";", "|")


def parse_rule(text: str, behavior: Behavior, source: Source) -> Rule:
    """Parse `ToolName` or `ToolName(content)`, honouring `\\(` `\\)` escapes
    inside content."""
    text = text.strip()
    n = len(text)
    i = 0
    paren_idx: int | None = None
    while i < n:
        c = text[i]
        if c == "\\" and i + 1 < n and text[i + 1] in "()":
            i += 2
            continue
        if c == "(":
            paren_idx = i
            break
        i += 1
    if paren_idx is None:
        return Rule(tool=text, content=None, behavior=behavior, source=source)
    if not text.endswith(")"):
        raise ValueError(f"malformed rule (unclosed parenthesis): {text!r}")
    tool = text[:paren_idx]
    raw_content = text[paren_idx + 1 : -1]
    content = raw_content.replace("\\(", "(").replace("\\)", ")")
    return Rule(tool=tool, content=content, behavior=behavior, source=source)


def split_compound(cmd: str) -> list[str]:
    """Split a shell command into segments on `&&`, `||`, `;`, `|`, ignoring
    those operators when they appear inside single or double quotes. A
    simple tokenizer, not a full shell grammar."""
    segments: list[str] = []
    current: list[str] = []
    quote: str | None = None
    i = 0
    n = len(cmd)
    while i < n:
        c = cmd[i]
        if quote:
            current.append(c)
            if c == quote:
                quote = None
            i += 1
            continue
        if c in ("'", '"'):
            quote = c
            current.append(c)
            i += 1
            continue
        if cmd[i : i + 2] in _COMPOUND_OPS_2:
            segments.append("".join(current).strip())
            current = []
            i += 2
            continue
        if c in _COMPOUND_OPS_1:
            segments.append("".join(current).strip())
            current = []
            i += 1
            continue
        current.append(c)
        i += 1
    tail = "".join(current).strip()
    if tail:
        segments.append(tail)
    return [s for s in segments if s]


def matches(rule: Rule, req: ToolCallRequest, cwd: str) -> bool:
    """Does this single rule match this request? For a compound shell
    command this is true if the rule's content matches *any* segment -- the
    right semantics for a deny/ask rule ("any bad segment triggers it"), and
    also usable (via `shell_fully_allowed`) as a building block for the
    "every segment covered" allow semantics."""
    if rule.tool.startswith("mcp__") and rule.whole_tool:
        return _mcp_matches(rule.tool, req.tool_name)
    if rule.tool != req.tool_name:
        return False
    if rule.whole_tool:
        return True
    content = rule.content or ""
    if rule.tool == "shell":
        segments = split_compound(req.permission_content)
        if len(segments) <= 1:
            return _shell_content_matches(content, req.permission_content.strip())
        return any(_shell_content_matches(content, seg) for seg in segments)
    if rule.tool == "fetch":
        return _fetch_matches(content, req)
    if rule.tool in PATH_TOOLS:
        candidates = req.paths or ([req.permission_content] if req.permission_content else [])
        return any(_path_content_matches(content, p, cwd) for p in candidates)
    return content == req.permission_content


def shell_fully_allowed(req: ToolCallRequest, cwd: str, allow_rules: list[Rule]) -> bool:
    """True if every segment of a (possibly compound) shell command is
    covered by at least one rule in `allow_rules` (which may each cover
    different segments). `allow_rules` should already be filtered to
    behavior == "allow" and tool == "shell"."""
    segments = split_compound(req.permission_content) or [req.permission_content.strip()]
    for seg in segments:
        seg_req = ToolCallRequest(
            tool_name=req.tool_name,
            args=req.args,
            permission_content=seg,
            is_read_only=req.is_read_only,
            is_destructive=req.is_destructive,
            paths=req.paths,
        )
        if any(matches(r, seg_req, cwd) for r in allow_rules):
            continue
        if _segment_is_read_only(seg):
            # A known read-only filter such as `| tail -40` or `| grep x` never
            # needs its own rule; only the segments that can change state do.
            continue
        return False
    return True


def _segment_is_read_only(segment: str) -> bool:
    try:
        from open_harness.tools.shell import _segment_is_read_only as check
    except ImportError:  # pragma: no cover - tools package always present
        return False
    return check(segment)


def _strip_env_prefix(segment: str) -> str:
    try:
        from open_harness.tools.shell import strip_env_prefix
    except ImportError:  # pragma: no cover - tools package always present
        return segment
    return strip_env_prefix(segment)


def path_under(path: str, root: str) -> bool:
    """Lexical containment check (no symlink resolution): is `path` at or
    below `root`? Case-insensitive, since this prototype targets Windows and
    POSIX both and treats case-sensitivity as out of scope."""
    p = Path(path)
    if not p.is_absolute():
        p = Path(root) / p
    root_p = Path(root)
    if not root_p.is_absolute():
        root_p = Path.cwd() / root_p
    p_norm = Path(os.path.normpath(str(p))).as_posix().lower()
    root_norm = Path(os.path.normpath(str(root_p))).as_posix().lower()
    return p_norm == root_norm or p_norm.startswith(root_norm + "/")


def suggest_rule(req: ToolCallRequest) -> str:
    """Build a suggested rule string for a passthrough ask (spec 8.2 step 3)."""
    if req.tool_name == "shell":
        tokens = req.permission_content.split()
        prefix = " ".join(tokens[:2]) if tokens else ""
        return f"shell({prefix} *)".replace("  ", " ")
    if req.tool_name in PATH_TOOLS:
        path = req.paths[0] if req.paths else req.permission_content
        return f"{req.tool_name}({path})"
    return f"{req.tool_name}({req.permission_content})"


# --- shell content matching -------------------------------------------------


def _shell_content_matches(content: str, command: str) -> bool:
    command = _strip_env_prefix(command)
    if content.endswith(":*"):
        prefix = content[:-2]
        return command.startswith(prefix)
    if content.endswith("*"):
        prefix = content[:-1]
        return command.startswith(prefix)
    if "*" in content:
        return fnmatch.fnmatchcase(command, content)
    return command == content


# --- fetch matching -----------------------------------------------------


def _fetch_matches(content: str, req: ToolCallRequest) -> bool:
    pattern = content.removeprefix("domain:").strip().lower()
    host = _extract_host(req)
    if not host:
        return False
    host = host.lower()
    return host == pattern or host.endswith("." + pattern)


def _extract_host(req: ToolCallRequest) -> str | None:
    candidate = req.args.get("url") or req.args.get("domain") or req.permission_content
    if not candidate:
        return None
    if "://" in candidate:
        try:
            return urlparse(candidate).hostname
        except ValueError:
            return None
    return candidate.split("/")[0]


# --- mcp matching --------------------------------------------------------


def _mcp_matches(rule_tool: str, tool_name: str) -> bool:
    if rule_tool == tool_name:
        return True
    parts = rule_tool.split("__")
    if len(parts) == 2:  # mcp__server -> server prefix
        server = parts[1]
        return tool_name.startswith(f"mcp__{server}__")
    return False


# --- path matching ---------------------------------------------------------


def _to_posix_abs(path_str: str, cwd: str) -> str:
    p = Path(path_str)
    if not p.is_absolute():
        p = Path(cwd) / p
    return Path(os.path.normpath(str(p))).as_posix()


def _rule_path_pattern(content: str, cwd: str) -> tuple[str, bool]:
    """Returns (pattern, anchored). Anchored patterns are matched against the
    full posix-normalised absolute path; unanchored ones may match anywhere
    in it."""
    if content.startswith("~/") or content == "~":
        rest = content[2:] if content.startswith("~/") else ""
        pattern = Path(os.path.normpath(str(Path.home() / rest))).as_posix()
        return pattern, True
    if content.startswith("//"):
        rest = content[2:].lstrip("/")
        # Filesystem root. POSIX has one root; Windows has one per drive, so
        # the drive letter is left as a single-char wildcard ('?' in fnmatch)
        # rather than picking cwd's drive -- "//" means "any drive's root".
        pattern = ("?:/" + rest) if os.name == "nt" else ("/" + rest)
        return pattern, True
    if content.startswith("/"):
        pattern = Path(os.path.normpath(str(Path(cwd) / content[1:]))).as_posix()
        return pattern, True
    return content, False


def _path_content_matches(content: str, path_str: str, cwd: str) -> bool:
    abs_path = _to_posix_abs(path_str, cwd)
    pattern, anchored = _rule_path_pattern(content, cwd)
    if anchored:
        if fnmatch.fnmatchcase(abs_path, pattern):
            return True
        if pattern.endswith("/**"):
            prefix = pattern[:-3]
            if fnmatch.fnmatchcase(abs_path, prefix) or fnmatch.fnmatchcase(abs_path, prefix + "/*"):
                return True
        return False
    # unrooted: match anywhere -- basename, or any suffix of the path.
    name = Path(abs_path).name
    if fnmatch.fnmatchcase(name, content):
        return True
    if fnmatch.fnmatchcase(abs_path, "*" + content):
        return True
    return "**" in content and fnmatch.fnmatchcase(abs_path, "*/" + content)
