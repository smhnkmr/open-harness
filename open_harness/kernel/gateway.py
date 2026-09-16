"""Model gateway: build the neutral request, call the adapter, reduce the
stream, retry with backoff, fall back to another role on overload.

Spec: open-harness-spec.md section 5.5.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass

from open_harness.kernel.roles import ResolvedRole, RoleResolver
from open_harness.model.reducer import ProviderErrorRaised, reduce
from open_harness.model.types import (
    Block,
    Message,
    ModelRequest,
    ModelResponse,
    ProviderError,
    TextDelta,
    ThinkingConfig,
    ToolSpec,
)

MAX_RETRIES = 10
BASE_DELAY = 0.5
MAX_DELAY = 32.0
OVERLOAD_RETRIES = 3


class GatewayError(Exception):
    def __init__(self, error: ProviderError, attempts: int) -> None:
        super().__init__(f"{error.kind}: {error.message} (after {attempts} attempts)")
        self.error = error
        self.attempts = attempts


class ContextTooLong(GatewayError):
    """Raised so the loop can attempt reactive compaction."""


@dataclass
class Gateway:
    resolver: RoleResolver
    on_event: Callable[[str, dict], None]          # log sink: kind, payload
    sleep: Callable[[float], None] = time.sleep     # injectable for tests
    latched_headers: set[str] | None = None         # spec P1: headers latch on for the session
    on_text_delta: Callable[[str, str], None] | None = None   # (role, text) for live rendering; not logged

    def _tap(self, role: str, events):
        """Forward text deltas to the client as they stream; pass everything through."""
        for ev in events:
            if self.on_text_delta is not None and isinstance(ev, TextDelta):
                self.on_text_delta(role, ev.text)
            yield ev

    def call(self, role: str, *, system: list[Block], messages: list[Message], tools: list[ToolSpec],
             max_output_tokens: int, thinking: ThinkingConfig | None = None) -> ModelResponse:
        resolved = self.resolver.resolve(role)
        self.on_event("model_resolved", {"role": role, "spec": resolved.spec})
        req = ModelRequest(system=list(system), messages=list(messages), tools=list(tools),
                           max_output_tokens=max_output_tokens, thinking=thinking, role=role)
        try:
            return self._call_with_retry(resolved, req)
        except GatewayError as e:
            if e.error.kind == "overloaded" and self.resolver.has_role("fallback") and role != "fallback":
                self.on_event("info", {"msg": f"overloaded on {resolved.spec}; using fallback role"})
                fb = self.resolver.resolve("fallback")
                self.on_event("model_resolved", {"role": "fallback", "spec": fb.spec})
                return self._call_with_retry(fb, req)
            raise

    def _call_with_retry(self, resolved: ResolvedRole, req: ModelRequest) -> ModelResponse:
        overloads = 0
        last: ProviderError | None = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = reduce(self._tap(req.role, resolved.adapter.stream(resolved.model, req)),
                              model=resolved.spec, role=req.role)
                self.on_event("usage", {"role": req.role, "spec": resolved.spec,
                                        "input": resp.usage.input_tokens, "output": resp.usage.output_tokens,
                                        "cache_read": resp.usage.cache_read_tokens,
                                        "cache_write": resp.usage.cache_write_tokens})
                return resp
            except ProviderErrorRaised as raised:
                err = raised.error
                last = err
                self.on_event("provider_error", {"kind": err.kind, "message": err.message[:500],
                                                 "attempt": attempt, "retryable": err.retryable})
                if err.kind == "context_too_long":
                    raise ContextTooLong(err, attempt) from None
                if not err.retryable:
                    raise GatewayError(err, attempt) from None
                if err.kind == "overloaded":
                    overloads += 1
                    if overloads >= OVERLOAD_RETRIES:
                        raise GatewayError(err, attempt) from None
                self.sleep(self._delay(attempt, err.retry_after))
        assert last is not None
        raise GatewayError(last, MAX_RETRIES)

    @staticmethod
    def _delay(attempt: int, retry_after: float | None) -> float:
        if retry_after:
            return min(retry_after, 60.0)
        base = min(BASE_DELAY * (2 ** (attempt - 1)), MAX_DELAY)
        return base * (1 + random.uniform(-0.25, 0.25))
