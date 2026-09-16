"""Adapter contract. Two required methods. Everything else is a default.

Spec: open-harness-spec.md section 5.2.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any

from open_harness.model.types import CapabilityFlags, Event, ModelRequest


class Adapter(ABC):
    """A vendor adapter. Constructed once per configured provider.

    `model` passed to generate/stream is the bare model id (without provider prefix).
    Adapters MUST NOT parse tool-call JSON; they emit ToolCallFragment events and
    the kernel reduces them. Adapters MUST convert canonical OpenAI-shaped tool
    schemas to their own dialect. Adapters MUST emit exactly one Stop or one
    ProviderError at the end of a stream.
    """

    name: str = "abstract"
    flags: CapabilityFlags = CapabilityFlags()

    def __init__(self, *, api_key: str | None = None, base_url: str | None = None,
                 timeout: float = 600.0, extra: dict[str, Any] | None = None) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.extra = extra or {}

    @abstractmethod
    def stream(self, model: str, req: ModelRequest) -> Iterator[Event]:
        """Stream events for one request. Must end with Stop or ProviderError."""

    def generate(self, model: str, req: ModelRequest) -> list[Event]:
        """Default: collect the stream. Adapters may override with a non-streaming call."""
        return list(self.stream(model, req))
