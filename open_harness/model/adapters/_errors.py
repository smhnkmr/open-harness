"""Shared SDK-exception-to-`ProviderError` mapping.

Not part of the public deliverable surface; both `anthropic.py` and
`openai_compatible.py` import it because the anthropic and openai SDKs use
structurally identical exception hierarchies (`APIStatusError` subclasses
carrying `.status_code`, `.message`, `.response.headers`) and the spec (5.2,
5.5) calls for identical error-kind mapping in both adapters.
"""

from __future__ import annotations

from open_harness.model.types import ProviderError


def map_provider_exception(exc: Exception) -> ProviderError:
    """Map an SDK exception to a neutral `ProviderError`.

    Rules (spec sections 3, 5.2):
      429                              -> rate_limit (retryable)
      529, or "overloaded" in message  -> overloaded (retryable)
      400 with "prompt is too long"    -> context_too_long
      401 / 403                        -> auth
      other 4xx                        -> bad_request
      5xx                              -> server (retryable)
      connection/timeout errors        -> network (retryable)
      anything else                    -> unknown
    """
    status = getattr(exc, "status_code", None)
    message = getattr(exc, "message", None) or str(exc)

    if status is None:
        cls_name = exc.__class__.__name__
        if "Connection" in cls_name or "Timeout" in cls_name:
            return ProviderError(kind="network", message=message, retryable=True, status=None)
        return ProviderError(kind="unknown", message=message, retryable=False, status=None)

    retry_after = _retry_after(exc)
    lowered = message.lower()

    if status == 429:
        return ProviderError(
            kind="rate_limit", message=message, retryable=True, retry_after=retry_after, status=status
        )
    if status == 529 or "overloaded" in lowered:
        return ProviderError(
            kind="overloaded", message=message, retryable=True, retry_after=retry_after, status=status
        )
    if status == 400 and "prompt is too long" in lowered:
        return ProviderError(kind="context_too_long", message=message, retryable=False, status=status)
    if status in (401, 403):
        return ProviderError(kind="auth", message=message, retryable=False, status=status)
    if 400 <= status < 500:
        return ProviderError(kind="bad_request", message=message, retryable=False, status=status)
    if status >= 500:
        return ProviderError(kind="server", message=message, retryable=True, status=status)
    return ProviderError(kind="unknown", message=message, retryable=False, status=status)


def _retry_after(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    value = headers.get("retry-after")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
