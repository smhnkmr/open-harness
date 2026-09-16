"""Token estimation and the summarise tier of compaction.

Spec: open-harness-spec.md sections 9.4, 9.7.
"""

from __future__ import annotations

import json
import re

from open_harness.model.types import Block, Message

# --------------------------------------------------------------------------- token estimation

_IMAGE_TOKENS = 2000


def _block_tokens(block: Block) -> int:
    if block.type == "image":
        return _IMAGE_TOKENS
    total = 0
    if block.text:
        total += len(block.text.encode("utf-8")) // 4
    if block.args is not None:
        # JSON-ish structured content compresses better than prose text.
        total += len(json.dumps(block.args).encode("utf-8")) // 4 // 2
    if block.native is not None:
        total += len(json.dumps(block.native).encode("utf-8")) // 4 // 2
    if block.data:
        total += len(block.data.encode("utf-8")) // 4
    if block.content:
        total += sum(_block_tokens(b) for b in block.content)
    return total


def estimate_tokens(messages: list[Message], system: list[Block], tools_chars: int) -> int:
    """Rough estimate per spec 9.7: bytes/4, JSON-ish content /2, images fixed at 2,000."""
    total = sum(_block_tokens(b) for m in messages for b in m.blocks)
    total += sum(_block_tokens(b) for b in system)
    total += tools_chars // 4
    return total


def should_compact(est: int, window: int, max_output: int) -> bool:
    """Spec 9.4 summarise-tier trigger: `window - min(max_output, 20_000) - 13_000`."""
    threshold = window - min(max_output, 20_000) - 13_000
    return est >= threshold


# --------------------------------------------------------------------------- summarise tier

SUMMARY_PROMPT = """Your context window is nearly full. Do not call any tools in this response.

Before writing the summary, think through the conversation in an <analysis> block: go chronologically through each user request and your work, noting technical decisions, files touched and why, and anything still open. This block is a private scratchpad and will be discarded; be thorough here.

Then write a <summary> containing exactly these nine sections, in order:

1. Primary request and intent
2. Key technical concepts
3. Files and code sections (with representative snippets)
4. Errors and fixes
5. Problem solving
6. All user messages verbatim
7. Pending tasks
8. Current work
9. Next step (quote the most recent instruction verbatim and state how it connects to the current work)

Output only <analysis>...</analysis> followed by <summary>...</summary>. No other text, and no tool calls."""

_ANALYSIS_RE = re.compile(r"<analysis>.*?</analysis>", re.DOTALL)


def strip_analysis(text: str) -> str:
    """Remove the `<analysis>` scratchpad, leaving the `<summary>` (and anything else)."""
    return _ANALYSIS_RE.sub("", text).strip()


def build_compact_request(
    messages: list[Message], system: list[Block], keep_recent: int = 4
) -> tuple[list[Message], list[Message]]:
    """Split `messages` into (to summarise, to keep verbatim).

    `keep_recent` messages stay off the chopping block; `system` is accepted
    for interface symmetry with the rest of the context pipeline (callers may
    want it to size the summarisation request) but is not otherwise consulted
    here.
    """
    del system
    if keep_recent <= 0:
        return list(messages), []
    if keep_recent >= len(messages):
        return [], list(messages)
    split = len(messages) - keep_recent
    return messages[:split], messages[split:]


def apply_summary(summary_text: str, kept: list[Message], recent_files: list[str]) -> list[Message]:
    """Build the post-compaction message list: summary, ack, then the kept tail."""
    body = strip_analysis(summary_text)
    if recent_files:
        files_block = "\n".join(f"- {f}" for f in recent_files[:5])
        body = f"{body}\n\nRecently read files:\n{files_block}"
    wrapped = f'<harness-context type="compact-summary">\n{body}\n</harness-context>'

    summary_message = Message(
        role="user", blocks=[Block.text_block(wrapped)], meta={"harness": True, "compact": True}
    )
    ack_message = Message(role="assistant", blocks=[Block.text_block("Understood, continuing.")])
    return [summary_message, ack_message, *kept]
