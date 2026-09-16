"""Role resolution: bind slots to provider:model once per turn.

Spec: open-harness-spec.md section 5.7.
"""

from __future__ import annotations

from dataclasses import dataclass

from open_harness.config import Config, ProviderConfig
from open_harness.model.adapter import Adapter
from open_harness.model.registry import build_adapter


@dataclass
class ResolvedRole:
    role: str
    spec: str            # provider:model
    provider: ProviderConfig
    model: str           # bare model id
    adapter: Adapter


class RoleResolver:
    """Caches one adapter instance per provider. Resolution is per turn; the
    loop calls resolve() at turn start and records the result in the log."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._adapters: dict[str, Adapter] = {}

    def resolve(self, role: str) -> ResolvedRole:
        spec = self.config.resolve_role(role)
        provider, model = self.config.split_spec(spec)
        if provider.name not in self._adapters:
            self._adapters[provider.name] = build_adapter(provider)
        return ResolvedRole(role=role, spec=spec, provider=provider, model=model,
                            adapter=self._adapters[provider.name])

    def has_role(self, role: str) -> bool:
        return role in self.config.roles
