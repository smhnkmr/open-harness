"""Neutral model types. The kernel speaks only these; adapters translate.

Spec: open-harness-spec.md section 5.1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

# --------------------------------------------------------------------------- blocks

BlockType = Literal[
    "text", "thinking", "tool_call", "tool_result", "image", "file", "citation", "non_standard"
]


@dataclass
class Block:
    """One content block. Standard vocabulary plus a lossless escape hatch.

    `native` keeps the provider's original block so it can be resent verbatim.
    `needs_server_state` marks native blocks that cannot be resent without the
    provider's own session state (e.g. OpenAI reasoning items without
    encrypted_content). The gateway drops those when state is unavailable.
    """

    type: BlockType
    text: str | None = None                 # text, thinking
    tool_call_id: str | None = None         # tool_call, tool_result
    name: str | None = None                 # tool_call
    args: dict[str, Any] | None = None      # tool_call (parsed)
    is_error: bool = False                  # tool_result
    content: list[Block] | None = None    # tool_result may nest text/image blocks
    media_type: str | None = None           # image, file
    data: str | None = None                 # image (base64), file (base64), non_standard (json)
    path: str | None = None                 # file reference
    native: dict[str, Any] | None = None    # provider-native block, kept verbatim
    needs_server_state: bool = False
    cache_breakpoint: bool = False          # hint; adapters may honour or ignore

    @staticmethod
    def text_block(text: str, *, cache_breakpoint: bool = False) -> Block:
        return Block(type="text", text=text, cache_breakpoint=cache_breakpoint)

    @staticmethod
    def tool_result(tool_call_id: str, content: str | list[Block], *, is_error: bool = False) -> Block:
        if isinstance(content, str):
            return Block(type="tool_result", tool_call_id=tool_call_id, text=content, is_error=is_error)
        return Block(type="tool_result", tool_call_id=tool_call_id, content=content, is_error=is_error)


Role = Literal["user", "assistant"]


@dataclass
class Message:
    role: Role
    blocks: list[Block]
    # Harness bookkeeping. Never sent to the provider.
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def tool_calls(self) -> list[Block]:
        return [b for b in self.blocks if b.type == "tool_call"]

    @property
    def text(self) -> str:
        return "".join(b.text or "" for b in self.blocks if b.type == "text")


# --------------------------------------------------------------------------- request


@dataclass
class ToolSpec:
    name: str
    description: str
    json_schema: dict[str, Any]          # OpenAI-shaped parameters schema (canonical dialect)
    defer: bool = False


@dataclass
class ThinkingConfig:
    enabled: bool = True
    budget_tokens: int | None = None     # None means adaptive where supported


@dataclass
class ModelRequest:
    system: list[Block]                  # static prefix blocks, then BOUNDARY marker, then dynamic
    messages: list[Message]
    tools: list[ToolSpec]
    max_output_tokens: int = 8192
    thinking: ThinkingConfig | None = None
    role: str = "main"
    stop_sequences: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


BOUNDARY = "__OPEN_HARNESS_DYNAMIC_BOUNDARY__"
"""Literal marker block text. Adapters that support cache control place a
breakpoint on the last block before this marker and drop the marker itself."""


# --------------------------------------------------------------------------- events


@dataclass
class TextDelta:
    text: str


@dataclass
class ThinkingDelta:
    text: str
    native: dict[str, Any] | None = None   # e.g. signature, redacted data


@dataclass
class ToolCallFragment:
    """One streamed fragment of a tool call. The kernel reducer concatenates
    `args_fragment` by `index` and parses the result. Adapters that receive
    complete tool calls emit exactly one fragment per call with the full JSON."""

    index: int
    id: str | None = None
    name: str | None = None
    args_fragment: str = ""


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


StopReason = Literal["end_turn", "tool_use", "max_tokens", "stop_sequence", "other"]


@dataclass
class Stop:
    reason: StopReason
    native_reason: str | None = None


@dataclass
class ProviderError:
    kind: Literal["rate_limit", "overloaded", "context_too_long", "auth", "bad_request", "server", "network", "unknown"]
    message: str
    retryable: bool
    retry_after: float | None = None
    status: int | None = None


Event = TextDelta | ThinkingDelta | ToolCallFragment | Usage | Stop | ProviderError


# --------------------------------------------------------------------------- capability flags


@dataclass(frozen=True)
class CapabilityFlags:
    """Declared by every adapter. The kernel branches on these, never on adapter name."""

    requires_role_alternation: bool = False
    streams_complete_tool_calls: bool = False
    native_structured_output: bool = False
    supports_cache_control: bool = False
    supports_thinking: bool = False
    server_state_blocks: bool = False
    tool_schema_dialect: str = "openai"


# --------------------------------------------------------------------------- reduced response


@dataclass
class InvalidToolCall:
    index: int
    id: str | None
    name: str | None
    raw_args: str
    error: str


@dataclass
class ModelResponse:
    """What the gateway hands the loop after reducing a stream."""

    message: Message                       # assistant message with text/thinking/tool_call blocks
    invalid_tool_calls: list[InvalidToolCall]
    usage: Usage
    stop: Stop
    model: str                              # provider:model that produced it
    role: str
