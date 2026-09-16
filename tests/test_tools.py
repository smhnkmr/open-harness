from __future__ import annotations

import shutil
import time
from pathlib import Path

import pytest

from open_harness.backend.local import LocalBackend
from open_harness.tools.base import ToolContext, ToolResult
from open_harness.tools.edit import EditTool
from open_harness.tools.glob import GlobTool
from open_harness.tools.grep import GrepTool
from open_harness.tools.read import ReadTool
from open_harness.tools.results import (
    AGGREGATE_MAX_CHARS,
    enforce_aggregate_budget,
    persist_if_large,
)
from open_harness.tools.shell import ShellTool
from open_harness.tools.write import WriteTool


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    cwd = tmp_path / "work"
    cwd.mkdir()
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    backend = LocalBackend(root=tmp_path)
    return ToolContext(cwd=cwd, session_dir=session_dir, backend=backend, read_state={})


# --------------------------------------------------------------------- read


def test_read_numbers_lines_cat_n_style(ctx: ToolContext) -> None:
    f = ctx.cwd / "hello.txt"
    f.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    tool = ReadTool()

    result = tool.call({"file_path": "hello.txt"}, ctx)

    assert not result.is_error
    lines = result.content.splitlines()
    assert lines[0] == "     1\talpha"
    assert lines[1] == "     2\tbeta"
    assert lines[2] == "     3\tgamma"
    assert str(f.resolve()) in ctx.read_state


def test_read_records_read_state(ctx: ToolContext) -> None:
    f = ctx.cwd / "hello.txt"
    f.write_text("one\ntwo\n", encoding="utf-8")
    tool = ReadTool()
    tool.call({"file_path": "hello.txt"}, ctx)

    record = ctx.read_state[str(f.resolve())]
    assert record.full_read is True
    assert record.size == f.stat().st_size


def test_read_unchanged_stub_on_repeat(ctx: ToolContext) -> None:
    f = ctx.cwd / "hello.txt"
    f.write_text("alpha\nbeta\n", encoding="utf-8")
    tool = ReadTool()

    first = tool.call({"file_path": "hello.txt"}, ctx)
    assert "alpha" in first.content

    second = tool.call({"file_path": "hello.txt"}, ctx)
    assert second.content == "(file unchanged since last read)"


def test_read_offset_limit(ctx: ToolContext) -> None:
    f = ctx.cwd / "hello.txt"
    f.write_text("\n".join(f"line{i}" for i in range(1, 11)) + "\n", encoding="utf-8")
    tool = ReadTool()

    result = tool.call({"file_path": "hello.txt", "offset": 3, "limit": 2}, ctx)
    lines = result.content.splitlines()
    assert lines[0].endswith("line3")
    assert lines[1].endswith("line4")
    assert len(lines) == 2


def test_read_errors_on_missing_file(ctx: ToolContext) -> None:
    tool = ReadTool()
    result = tool.call({"file_path": "nope.txt"}, ctx)
    assert result.is_error


def test_read_rejects_oversized_full_read(ctx: ToolContext) -> None:
    f = ctx.cwd / "big.txt"
    f.write_bytes(b"x" * (ReadTool.MAX_FILE_BYTES + 1))
    tool = ReadTool()
    result = tool.call({"file_path": "big.txt"}, ctx)
    assert result.is_error
    assert "limit" in result.content.lower()


# --------------------------------------------------------------------- edit


def test_edit_requires_read_first(ctx: ToolContext) -> None:
    f = ctx.cwd / "f.txt"
    f.write_text("hello world\n", encoding="utf-8")
    tool = EditTool()

    result = tool.call({"file_path": "f.txt", "old_string": "hello", "new_string": "goodbye"}, ctx)
    assert result.is_error
    assert "NOT_READ" in result.content


def test_edit_succeeds_after_read(ctx: ToolContext) -> None:
    f = ctx.cwd / "f.txt"
    f.write_text("hello world\n", encoding="utf-8")
    ReadTool().call({"file_path": "f.txt"}, ctx)

    result = EditTool().call({"file_path": "f.txt", "old_string": "hello", "new_string": "goodbye"}, ctx)
    assert not result.is_error
    assert f.read_text(encoding="utf-8") == "goodbye world\n"


def test_edit_detects_change_on_disk(ctx: ToolContext) -> None:
    f = ctx.cwd / "f.txt"
    f.write_text("hello world\n", encoding="utf-8")
    ReadTool().call({"file_path": "f.txt"}, ctx)

    # Simulate an external modification: change content and bump mtime.
    time.sleep(0.01)
    f.write_text("hello mutated world\n", encoding="utf-8")

    result = EditTool().call({"file_path": "f.txt", "old_string": "hello", "new_string": "goodbye"}, ctx)
    assert result.is_error
    assert "CHANGED_ON_DISK" in result.content


def test_edit_rejects_multiple_matches_without_replace_all(ctx: ToolContext) -> None:
    f = ctx.cwd / "f.txt"
    f.write_text("dup dup dup\n", encoding="utf-8")
    ReadTool().call({"file_path": "f.txt"}, ctx)

    result = EditTool().call({"file_path": "f.txt", "old_string": "dup", "new_string": "one"}, ctx)
    assert result.is_error
    assert "MULTIPLE_MATCHES" in result.content


def test_edit_replace_all(ctx: ToolContext) -> None:
    f = ctx.cwd / "f.txt"
    f.write_text("dup dup dup\n", encoding="utf-8")
    ReadTool().call({"file_path": "f.txt"}, ctx)

    result = EditTool().call(
        {"file_path": "f.txt", "old_string": "dup", "new_string": "one", "replace_all": True}, ctx
    )
    assert not result.is_error
    assert f.read_text(encoding="utf-8") == "one one one\n"


def test_edit_no_match(ctx: ToolContext) -> None:
    f = ctx.cwd / "f.txt"
    f.write_text("hello world\n", encoding="utf-8")
    ReadTool().call({"file_path": "f.txt"}, ctx)

    result = EditTool().call({"file_path": "f.txt", "old_string": "missing", "new_string": "x"}, ctx)
    assert result.is_error
    assert "NO_MATCH" in result.content


def test_edit_preserves_crlf(ctx: ToolContext) -> None:
    f = ctx.cwd / "crlf.txt"
    f.write_bytes(b"line one\r\nline two\r\nline three\r\n")
    ReadTool().call({"file_path": "crlf.txt"}, ctx)

    result = EditTool().call({"file_path": "crlf.txt", "old_string": "line two", "new_string": "LINE TWO"}, ctx)
    assert not result.is_error
    raw = f.read_bytes()
    assert b"\r\n" in raw
    assert b"\n\n" not in raw  # no accidental double-newline from bad normalisation
    assert raw == b"line one\r\nLINE TWO\r\nline three\r\n"


# -------------------------------------------------------------------- write


def test_write_new_file_does_not_require_read(ctx: ToolContext) -> None:
    result = WriteTool().call({"file_path": "new.txt", "content": "hi"}, ctx)
    assert not result.is_error
    assert (ctx.cwd / "new.txt").read_text(encoding="utf-8") == "hi"


def test_write_requires_read_for_existing_file(ctx: ToolContext) -> None:
    f = ctx.cwd / "f.txt"
    f.write_text("original", encoding="utf-8")

    result = WriteTool().call({"file_path": "f.txt", "content": "overwritten"}, ctx)
    assert result.is_error
    assert "read the file first" in result.content.lower()
    assert f.read_text(encoding="utf-8") == "original"


def test_write_succeeds_after_read(ctx: ToolContext) -> None:
    f = ctx.cwd / "f.txt"
    f.write_text("original", encoding="utf-8")
    ReadTool().call({"file_path": "f.txt"}, ctx)

    result = WriteTool().call({"file_path": "f.txt", "content": "overwritten"}, ctx)
    assert not result.is_error
    assert f.read_text(encoding="utf-8") == "overwritten"


def test_write_changed_on_disk(ctx: ToolContext) -> None:
    f = ctx.cwd / "f.txt"
    f.write_text("original", encoding="utf-8")
    ReadTool().call({"file_path": "f.txt"}, ctx)
    time.sleep(0.01)
    f.write_text("mutated externally", encoding="utf-8")

    result = WriteTool().call({"file_path": "f.txt", "content": "overwritten"}, ctx)
    assert result.is_error
    assert "changed on disk" in result.content.lower()


# -------------------------------------------------------------------- shell


def test_shell_echo(ctx: ToolContext) -> None:
    tool = ShellTool()
    result = tool.call({"command": "echo hi"}, ctx)
    assert not result.is_error
    assert "hi" in result.content


def test_shell_times_out(ctx: ToolContext) -> None:
    if ctx.backend.shell not in ("bash", "sh"):
        pytest.skip("sleep assumed posix-shell here")
    tool = ShellTool()
    start = time.time()
    result = tool.call({"command": "sleep 5", "timeout_ms": 500}, ctx)
    elapsed = time.time() - start
    assert result.data["timed_out"] is True
    assert elapsed < 4
    assert "timed out" in result.content.lower()


def test_shell_caps_output(ctx: ToolContext) -> None:
    if ctx.backend.shell not in ("bash", "sh"):
        pytest.skip("head/tr assumed posix-shell here")
    tool = ShellTool()
    result = tool.call({"command": "head -c 50000 /dev/zero | tr '\\0' 'x'"}, ctx)
    assert len(result.content) < 50000
    assert "chars omitted" in result.content


def test_shell_read_only_heuristic() -> None:
    from open_harness.tools.shell import command_is_read_only

    assert command_is_read_only("ls -la") is True
    assert command_is_read_only("git status") is True
    assert command_is_read_only("git log --oneline") is True
    assert command_is_read_only("cat foo.txt | grep bar") is True
    assert command_is_read_only("rm -rf /") is False
    assert command_is_read_only("git commit -m x") is False
    assert command_is_read_only("git push") is False
    assert command_is_read_only("echo hi > out.txt") is False
    assert command_is_read_only("ls && rm foo") is False


# --------------------------------------------------------------------- grep


def test_grep_python_fallback_finds_pattern(ctx: ToolContext) -> None:
    (ctx.cwd / "a.txt").write_text("needle here\nhaystack\n", encoding="utf-8")
    (ctx.cwd / "b.txt").write_text("nothing to see\n", encoding="utf-8")

    tool = GrepTool(force_python_fallback=True)
    result = tool.call({"pattern": "needle", "output_mode": "files_with_matches"}, ctx)

    assert not result.is_error
    assert "a.txt" in result.content
    assert "b.txt" not in result.content


def test_grep_python_fallback_content_mode(ctx: ToolContext) -> None:
    (ctx.cwd / "a.txt").write_text("needle here\nhaystack\n", encoding="utf-8")
    tool = GrepTool(force_python_fallback=True)
    result = tool.call({"pattern": "needle", "output_mode": "content"}, ctx)
    assert "needle here" in result.content


@pytest.mark.skipif(shutil.which("rg") is None, reason="ripgrep not on PATH")
def test_grep_with_ripgrep(ctx: ToolContext) -> None:
    (ctx.cwd / "a.txt").write_text("needle here\n", encoding="utf-8")
    tool = GrepTool(force_python_fallback=False)
    result = tool.call({"pattern": "needle", "output_mode": "files_with_matches"}, ctx)
    assert "a.txt" in result.content


def test_grep_no_matches(ctx: ToolContext) -> None:
    (ctx.cwd / "a.txt").write_text("nothing\n", encoding="utf-8")
    tool = GrepTool(force_python_fallback=True)
    result = tool.call({"pattern": "zzz_not_present"}, ctx)
    assert result.content == "(grep completed with no output)"


# --------------------------------------------------------------------- glob


def test_glob_caps_and_sorts(ctx: ToolContext) -> None:
    for i in range(5):
        p = ctx.cwd / f"file{i}.txt"
        p.write_text("x", encoding="utf-8")
        # ensure distinguishable mtimes on filesystems with coarse resolution
        t = time.time() + i
        import os

        os.utime(p, (t, t))

    tool = GlobTool()
    result = tool.call({"pattern": "*.txt"}, ctx)
    names = result.content.splitlines()
    assert names[0] == "file4.txt"  # most recently modified first
    assert names[-1] == "file0.txt"


def test_glob_cap_at_max_results(ctx: ToolContext, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("open_harness.tools.glob.MAX_RESULTS", 3)
    for i in range(6):
        (ctx.cwd / f"f{i}.txt").write_text("x", encoding="utf-8")

    tool = GlobTool()
    result = tool.call({"pattern": "*.txt"}, ctx)
    lines = [ln for ln in result.content.splitlines() if ln.endswith(".txt")]
    assert len(lines) == 3
    assert result.data["truncated"] is True


def test_glob_no_matches(ctx: ToolContext) -> None:
    tool = GlobTool()
    result = tool.call({"pattern": "*.nonexistent"}, ctx)
    assert result.content == "(glob completed with no output)"


# ------------------------------------------------------------------ results


def test_persist_if_large_writes_file_and_previews(ctx: ToolContext) -> None:
    big = "line\n" * 20000  # well over DEFAULT_MAX_RESULT_CHARS
    result = ToolResult(content=big)

    class SmallCapTool:
        name = "dummy"
        max_result_chars = 100

    persisted = persist_if_large(result, SmallCapTool(), "call_123", ctx.session_dir)

    assert persisted.persisted_path is not None
    persisted_file = Path(persisted.persisted_path)
    assert persisted_file.exists()
    assert persisted_file.read_text(encoding="utf-8") == big
    assert "persisted-output" in persisted.content
    assert persisted_file.name in persisted.content or str(persisted_file) in persisted.content


def test_persist_if_large_leaves_small_content(ctx: ToolContext) -> None:
    class SmallCapTool:
        name = "dummy"
        max_result_chars = 100

    result = ToolResult(content="short")
    out = persist_if_large(result, SmallCapTool(), "call_1", ctx.session_dir)
    assert out.content == "short"
    assert out.persisted_path is None


def test_persist_if_large_empty_content_becomes_no_output_message(ctx: ToolContext) -> None:
    class DummyTool:
        name = "grep"
        max_result_chars = 100

    result = ToolResult(content="")
    out = persist_if_large(result, DummyTool(), "call_1", ctx.session_dir)
    assert out.content == "(grep completed with no output)"


def test_persist_if_large_never_persists_image_blocks(ctx: ToolContext) -> None:
    from open_harness.model.types import Block

    class DummyTool:
        name = "read"
        max_result_chars = 100

    blocks = [Block(type="image", media_type="image/png", data="a" * 1000)]
    result = ToolResult(content=blocks)
    out = persist_if_large(result, DummyTool(), "call_1", ctx.session_dir)
    assert out.content is blocks
    assert out.persisted_path is None


def test_persist_write_once_keeps_first_write(ctx: ToolContext) -> None:
    class DummyTool:
        name = "dummy"
        max_result_chars = 10

    r1 = persist_if_large(ToolResult(content="a" * 1000), DummyTool(), "same_id", ctx.session_dir)
    r2 = persist_if_large(ToolResult(content="b" * 1000), DummyTool(), "same_id", ctx.session_dir)

    assert r1.persisted_path == r2.persisted_path
    assert Path(r1.persisted_path).read_text(encoding="utf-8") == "a" * 1000


def test_enforce_aggregate_budget_persists_largest_first(ctx: ToolContext) -> None:
    class DummyTool:
        name = "dummy"
        max_result_chars = float("inf")  # individually under any per-result cap

    big1 = "a" * 150_000
    big2 = "b" * 100_000
    small = "c" * 10

    results = [
        (DummyTool(), "call_1", ToolResult(content=big1)),
        (DummyTool(), "call_2", ToolResult(content=big2)),
        (DummyTool(), "call_3", ToolResult(content=small)),
    ]

    out = enforce_aggregate_budget(results, ctx.session_dir)
    total = sum(len(r.content) for r in out)
    assert total <= AGGREGATE_MAX_CHARS
    # The largest one should have been persisted (forced) to bring total down.
    assert out[0].persisted_path is not None
    # The small one should be untouched.
    assert out[2].content == small
    assert out[2].persisted_path is None


def test_stderr_redirect_does_not_make_command_non_read_only():
    from open_harness.tools.shell import command_is_read_only

    assert command_is_read_only("head -5 a.txt b.txt 2>&1")
    assert command_is_read_only("ls 2>/dev/null | wc -l")
    assert not command_is_read_only("ls > out.txt")
    assert not command_is_read_only("cat a.txt >> b.txt 2>&1")
