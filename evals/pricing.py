"""List prices per million tokens, used to cost both harnesses identically.

Sonnet 5 is assumed to share Sonnet 4.5's list price until confirmed; the
driver records the harness-reported cost next to the computed one so the
assumption can be checked against Claude Code's own number.
"""

from __future__ import annotations

from evals.types import TokenUsage

# (input, output, cache_read, cache_write) USD per 1M tokens
PRICES: dict[str, tuple[float, float, float, float]] = {
    "claude-sonnet-4-5": (3.0, 15.0, 0.30, 3.75),
    "claude-sonnet-5": (3.0, 15.0, 0.30, 3.75),
    "claude-haiku-4-5": (1.0, 5.0, 0.10, 1.25),
    "claude-opus-5": (15.0, 75.0, 1.50, 18.75),
}


def canonical(model: str) -> str:
    """'anthropic:claude-sonnet-5' or 'claude-sonnet-4-5-20250929' -> table key."""
    name = model.split(":", 1)[-1]
    for key in sorted(PRICES, key=len, reverse=True):
        if name.startswith(key):
            return key
    return name


def cost_usd_by_model(usage_by_model: dict[str, TokenUsage | dict[str, int]]) -> float:
    """Sum of per-model costs; each model priced at its own list rate."""
    total = 0.0
    for model, usage in usage_by_model.items():
        if isinstance(usage, dict):
            usage = TokenUsage(**usage)
        total += cost_usd(model, usage)
    return total


def cost_usd(model: str, usage: TokenUsage) -> float:
    key = canonical(model)
    if key not in PRICES:
        return 0.0
    i, o, cr, cw = PRICES[key]
    return (usage.input * i + usage.output * o + usage.cache_read * cr + usage.cache_write * cw) / 1e6
