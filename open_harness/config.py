"""Configuration: providers, roles, policy, verification. TOML.

Spec: open-harness-spec.md sections 5.7 and 5.9.

Example open-harness.toml:

    [providers.anthropic]  adapter = "anthropic"          api_key_env = "ANTHROPIC_API_KEY"
    [providers.local]      adapter = "openai-compatible"  base_url = "http://localhost:11434/v1"

    [roles]
    main       = "anthropic:claude-sonnet-4-5"
    explore    = "local:qwen3-coder"

    [policy]
    mode = "default"
    allow = ["read", "grep", "glob", "shell(git status*)"]
    deny  = ["read(~/.ssh/**)"]

    [verify]
    lint = "ruff check {file}"
    test = "pytest -q"
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROLES = ("main", "explore", "classifier", "evaluator", "compactor", "memory", "fallback")


@dataclass
class ProviderConfig:
    name: str
    adapter: str
    api_key_env: str | None = None
    base_url: str | None = None
    timeout: float = 600.0
    extra: dict[str, Any] = field(default_factory=dict)

    def api_key(self) -> str | None:
        return os.environ.get(self.api_key_env) if self.api_key_env else None


@dataclass
class PolicyConfig:
    mode: str = "default"
    allow: list[str] = field(default_factory=list)
    deny: list[str] = field(default_factory=list)
    ask: list[str] = field(default_factory=list)
    additional_dirs: list[str] = field(default_factory=list)


@dataclass
class VerifyConfig:
    lint: str | None = None     # command with {file} placeholder, run after each edit
    test: str | None = None     # command run before a turn may end


@dataclass
class Config:
    providers: dict[str, ProviderConfig]
    roles: dict[str, str]                 # role -> "provider:model"
    policy: PolicyConfig
    verify: VerifyConfig
    max_turns: int = 50
    max_output_tokens: int = 8192
    session_root: Path = Path.home() / ".open-harness" / "sessions"

    def resolve_role(self, role: str) -> str:
        """Unbound roles inherit from main. (spec 5.7)"""
        if role in self.roles:
            return self.roles[role]
        if "main" not in self.roles:
            raise ValueError("roles.main is required")
        return self.roles["main"]

    def split_spec(self, spec: str) -> tuple[ProviderConfig, str]:
        provider, sep, model = spec.partition(":")
        if not sep or not model:
            raise ValueError(f"model spec must be provider:model, got {spec!r}")
        if provider not in self.providers:
            raise ValueError(f"unknown provider {provider!r}; configured: {sorted(self.providers)}")
        return self.providers[provider], model


def load_dotenv(path: Path) -> int:
    """Load KEY=VALUE lines from a .env file into os.environ without overriding
    variables that are already set. Returns the number of keys loaded. Values
    are never logged. Lines starting with # are ignored."""
    if not path.exists():
        return 0
    loaded = 0
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded += 1
    return loaded


def load_config(path: Path | None = None) -> Config:
    path = path or _find_config()
    data: dict[str, Any] = {}
    if path and path.exists():
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        load_dotenv(path.parent / ".env")
    load_dotenv(Path.home() / ".open-harness" / ".env")
    providers = {
        name: ProviderConfig(name=name, adapter=p.get("adapter", name), api_key_env=p.get("api_key_env"),
                             base_url=p.get("base_url"), timeout=float(p.get("timeout", 600)),
                             extra={k: v for k, v in p.items()
                                    if k not in {"adapter", "api_key_env", "base_url", "timeout"}})
        for name, p in data.get("providers", {}).items()
    }
    pol = data.get("policy", {})
    ver = data.get("verify", {})
    return Config(
        providers=providers,
        roles=dict(data.get("roles", {})),
        policy=PolicyConfig(mode=pol.get("mode", "default"), allow=list(pol.get("allow", [])),
                            deny=list(pol.get("deny", [])), ask=list(pol.get("ask", [])),
                            additional_dirs=list(pol.get("additional_dirs", []))),
        verify=VerifyConfig(lint=ver.get("lint"), test=ver.get("test")),
        max_turns=int(data.get("max_turns", 50)),
        max_output_tokens=int(data.get("max_output_tokens", 8192)),
        session_root=Path(data.get("session_root", Path.home() / ".open-harness" / "sessions")),
    )


def _find_config() -> Path | None:
    for candidate in (Path.cwd() / "open-harness.toml", Path.home() / ".open-harness" / "config.toml"):
        if candidate.exists():
            return candidate
    return None
