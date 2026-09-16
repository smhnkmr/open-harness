"""Adapter registry: entry-point discovery with built-in fallback.

Spec: open-harness-spec.md sections 3, 5.4, 5.6 ("entry-point discovery,
per-plugin failure isolation").
"""

from __future__ import annotations

import importlib
import warnings
from importlib import metadata as importlib_metadata

from open_harness.config import ProviderConfig
from open_harness.model.adapter import Adapter

_ENTRY_POINT_GROUP = "open_harness.adapters"

_BUILTINS = {
    "anthropic": ("open_harness.model.adapters.anthropic", "AnthropicAdapter"),
    "openai-compatible": ("open_harness.model.adapters.openai_compatible", "OpenAICompatibleAdapter"),
}


def load_adapters() -> dict[str, type[Adapter]]:
    """Discover adapters via entry points, then fill in any missing built-in
    with a direct import so the two shipped adapters are always available
    even if entry-point discovery finds nothing or a plugin fails to load.

    A single plugin's failure is isolated: it is skipped with a warning and
    does not prevent other adapters (built-in or plugin) from loading.
    """
    adapters: dict[str, type[Adapter]] = {}

    try:
        entry_points = importlib_metadata.entry_points(group=_ENTRY_POINT_GROUP)
    except Exception as exc:  # noqa: BLE001 - discovery must never crash the caller
        warnings.warn(f"adapter entry-point discovery failed: {exc}")
        entry_points = []

    for ep in entry_points:
        try:
            adapters[ep.name] = ep.load()
        except Exception as exc:  # noqa: BLE001 - isolate one bad plugin
            warnings.warn(f"failed to load adapter plugin {ep.name!r}: {exc}")

    for name, (module_name, attr) in _BUILTINS.items():
        if name in adapters:
            continue
        try:
            module = importlib.import_module(module_name)
            adapters[name] = getattr(module, attr)
        except Exception as exc:  # noqa: BLE001 - isolate a broken built-in
            warnings.warn(f"failed to load built-in adapter {name!r}: {exc}")

    return adapters


def build_adapter(provider: ProviderConfig) -> Adapter:
    """Instantiate the adapter configured for one provider."""
    adapters = load_adapters()
    cls = adapters.get(provider.adapter)
    if cls is None:
        raise ValueError(f"unknown adapter {provider.adapter!r}; available: {sorted(adapters)}")
    return cls(
        api_key=provider.api_key(),
        base_url=provider.base_url,
        timeout=provider.timeout,
        extra=provider.extra,
    )
