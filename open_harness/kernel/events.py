"""Event log record types and the op/event protocol vocabulary.

Spec: open-harness-spec.md sections 3.1, 4.3, 12.1.

Every record is one JSON object per line. `kind` discriminates. `ts` is ISO-8601.
The log is append-only. Compaction writes a `compact_boundary`; it never deletes.
"""

from __future__ import annotations

from typing import Literal

OpKind = Literal[
    "turn_input", "interrupt", "approve", "set_mode", "set_role", "compact", "new_context", "shutdown",
]

EventKind = Literal[
    # loop lifecycle
    "session_start", "turn_start", "turn_end", "transition",
    # model
    "model_resolved", "text_delta", "thinking_delta", "assistant_message", "user_message", "tool_result",
    "usage", "provider_error",
    # tools
    "tool_call_start", "tool_call_end", "tool_result_persisted",
    # policy
    "permission_decision", "approval_request", "approval_response",
    # verification
    "verifier_result",
    # context
    "compact_boundary", "attachment",
    # async
    "task_notification",
    # misc
    "error", "info",
]

TerminalReason = Literal[
    "completed", "max_turns", "blocking_limit", "prompt_too_long", "aborted_streaming",
    "aborted_tools", "stop_hook_prevented", "hook_stopped", "model_error", "image_error",
    "budget_exhausted",
]

ContinueReason = Literal[
    "next_turn", "reactive_compact_retry", "max_output_tokens_escalate", "max_output_tokens_recovery",
    "stop_hook_blocking", "token_budget_continuation", "verifier_failed",
]
