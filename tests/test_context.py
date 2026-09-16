"""Tests for `open_harness.context.prompt` and `open_harness.context.compact`."""

from __future__ import annotations

from pathlib import Path

from open_harness.context.compact import (
    SUMMARY_PROMPT,
    apply_summary,
    build_compact_request,
    estimate_tokens,
    should_compact,
    strip_analysis,
)
from open_harness.context.prompt import build_system, instructions_message, load_instructions
from open_harness.model.types import BOUNDARY, Block, Message

# --------------------------------------------------------------------------- prompt.build_system


def test_build_system_contains_boundary_and_cache_breakpoint() -> None:
    blocks = build_system(
        Path("/repo"), tools_prompt="read, edit, grep", profile_suffix=None,
        env={"model": "anthropic:claude-test"},
    )
    boundary_positions = [i for i, b in enumerate(blocks) if b.text == BOUNDARY]
    assert len(boundary_positions) == 1
    boundary_idx = boundary_positions[0]
    assert boundary_idx > 0

    static_blocks = blocks[:boundary_idx]
    assert len(static_blocks) >= 2
    assert static_blocks[-1].cache_breakpoint is True
    assert all(not b.cache_breakpoint for b in static_blocks[:-1])

    dynamic_blocks = blocks[boundary_idx + 1:]
    assert any("cwd" in (b.text or "") for b in dynamic_blocks)
    assert any("anthropic:claude-test" in (b.text or "") for b in dynamic_blocks)


def test_build_system_folds_tools_prompt_into_static_section() -> None:
    blocks = build_system(
        Path("/repo"), tools_prompt="UNIQUE_TOOL_MARKER", profile_suffix=None, env={"model": "x"}
    )
    boundary_idx = next(i for i, b in enumerate(blocks) if b.text == BOUNDARY)
    static_text = "\n".join(b.text or "" for b in blocks[:boundary_idx])
    assert "UNIQUE_TOOL_MARKER" in static_text


def test_build_system_appends_profile_suffix_last() -> None:
    blocks = build_system(
        Path("/repo"), tools_prompt="", profile_suffix="Extra profile notes.", env={"model": "x"}
    )
    assert blocks[-1].text == "Extra profile notes."


def test_build_system_omits_profile_suffix_when_none() -> None:
    blocks_with = build_system(Path("/repo"), tools_prompt="", profile_suffix="suffix", env={"model": "x"})
    blocks_without = build_system(Path("/repo"), tools_prompt="", profile_suffix=None, env={"model": "x"})
    assert len(blocks_without) == len(blocks_with) - 1


def test_build_system_static_text_under_900_words() -> None:
    blocks = build_system(Path("/repo"), tools_prompt="", profile_suffix=None, env={"model": "x"})
    boundary_idx = next(i for i, b in enumerate(blocks) if b.text == BOUNDARY)
    words = sum(len((b.text or "").split()) for b in blocks[:boundary_idx])
    assert words < 900


# --------------------------------------------------------------------------- prompt.load_instructions


def test_load_instructions_finds_nested_agents_md_and_resolves_import(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    sub = root / "pkg"
    sub.mkdir(parents=True)
    (root / ".git").mkdir()
    (root / "shared.md").write_text("Shared rule text.", encoding="utf-8")
    (root / "AGENTS.md").write_text("Root rules.\n@shared.md\n", encoding="utf-8")
    (sub / "AGENTS.md").write_text(
        "Package rules.\n```\n@not-a-real-import.md\n```\n", encoding="utf-8"
    )

    result = load_instructions(sub)
    assert result is not None

    root_idx = result.index("Root rules.")
    pkg_idx = result.index("Package rules.")
    assert root_idx < pkg_idx  # root-most first

    assert "Shared rule text." in result
    assert "@not-a-real-import.md" in result  # left literal: inside a fence, not resolved
    assert f"Contents of {root / 'AGENTS.md'}" in result
    assert f"Contents of {sub / 'AGENTS.md'}" in result


def test_load_instructions_falls_back_to_claude_md(tmp_path: Path) -> None:
    root = tmp_path / "repo2"
    root.mkdir()
    (root / ".git").mkdir()
    (root / "CLAUDE.md").write_text("Fallback rules.", encoding="utf-8")

    result = load_instructions(root)
    assert result is not None
    assert "Fallback rules." in result


def test_load_instructions_caps_file_at_40000_chars(tmp_path: Path) -> None:
    root = tmp_path / "repo3"
    root.mkdir()
    (root / ".git").mkdir()
    (root / "AGENTS.md").write_text("x" * 50_000, encoding="utf-8")

    result = load_instructions(root)
    assert result is not None
    assert result.count("x") == 40_000


def test_instructions_message_wraps_with_harness_tag() -> None:
    msg = instructions_message("some text")
    assert msg.role == "user"
    assert msg.meta.get("harness") is True
    assert "harness-context" in msg.text
    assert "instructions" in msg.text
    assert "some text" in msg.text


# --------------------------------------------------------------------------- compact.should_compact


def test_should_compact_threshold() -> None:
    window = 200_000
    max_output = 8192
    threshold = window - min(max_output, 20_000) - 13_000

    assert should_compact(threshold - 1, window, max_output) is False
    assert should_compact(threshold, window, max_output) is True
    assert should_compact(threshold + 1000, window, max_output) is True


def test_should_compact_caps_max_output_at_20k() -> None:
    window = 200_000
    threshold_capped = window - 20_000 - 13_000
    assert should_compact(threshold_capped, window, max_output=100_000) is True
    assert should_compact(threshold_capped - 1, window, max_output=100_000) is False


def test_estimate_tokens_counts_images_as_2000() -> None:
    image_msg = Message(role="user", blocks=[Block(type="image", media_type="image/png", data="x" * 100)])
    est = estimate_tokens([image_msg], [], 0)
    assert est >= 2000


def test_estimate_tokens_grows_with_text_length() -> None:
    short = Message(role="user", blocks=[Block.text_block("hi")])
    long = Message(role="user", blocks=[Block.text_block("hello " * 1000)])
    assert estimate_tokens([long], [], 0) > estimate_tokens([short], [], 0)


# --------------------------------------------------------------------------- compact build/apply


def test_build_compact_request_splits_keep_recent() -> None:
    messages = [
        Message(role="user" if i % 2 == 0 else "assistant", blocks=[Block.text_block(str(i))])
        for i in range(10)
    ]
    to_summarise, kept = build_compact_request(messages, [], keep_recent=4)
    assert len(kept) == 4
    assert len(to_summarise) == 6
    assert kept == messages[-4:]
    assert to_summarise == messages[:-4]


def test_build_compact_request_keep_recent_exceeds_length() -> None:
    messages = [Message(role="user", blocks=[Block.text_block("only one")])]
    to_summarise, kept = build_compact_request(messages, [], keep_recent=4)
    assert to_summarise == []
    assert kept == messages


def test_apply_summary_strips_analysis_and_keeps_recent() -> None:
    summary_text = "<analysis>scratch notes, discard me</analysis><summary>1. intent...</summary>"
    kept = [Message(role="user", blocks=[Block.text_block("recent question")])]
    result = apply_summary(summary_text, kept, recent_files=["a.py", "b.py"])

    assert len(result) == 3
    assert "scratch notes" not in result[0].text
    assert "<summary>" in result[0].text
    assert result[0].role == "user"
    assert result[0].meta.get("harness") is True
    assert result[0].meta.get("compact") is True
    assert "a.py" in result[0].text and "b.py" in result[0].text

    assert result[1].role == "assistant"
    assert result[1].text == "Understood, continuing."

    assert result[2] is kept[0]


def test_strip_analysis_removes_block() -> None:
    text = "<analysis>hidden</analysis>\n\nvisible"
    assert strip_analysis(text) == "visible"


def test_strip_analysis_no_block_is_noop() -> None:
    assert strip_analysis("just a summary") == "just a summary"


def test_summary_prompt_forbids_tools_and_lists_nine_sections() -> None:
    assert "tool" in SUMMARY_PROMPT.lower()
    for section in [
        "Primary request and intent",
        "Key technical concepts",
        "Files and code sections",
        "Errors and fixes",
        "Problem solving",
        "All user messages verbatim",
        "Pending tasks",
        "Current work",
        "Next step",
    ]:
        assert section in SUMMARY_PROMPT
    assert "<analysis>" in SUMMARY_PROMPT
    assert "<summary>" in SUMMARY_PROMPT
    assert SUMMARY_PROMPT.index("Do not call any tools") < SUMMARY_PROMPT.index("<summary>")
