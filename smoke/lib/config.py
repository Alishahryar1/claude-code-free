"""Smoke-suite configuration loaded from the real developer environment."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from config.settings import Settings, get_settings

DEFAULT_TARGETS = frozenset(
    {
        "api",
        "auth",
        "cli",
        "contract",
        "optimizations",
        "providers",
        "thinking",
        "tools",
        "vscode",
    }
)
ALL_TARGETS = DEFAULT_TARGETS | frozenset({"discord", "telegram", "voice"})
SECRET_KEY_PARTS = ("KEY", "TOKEN", "SECRET", "WEBHOOK", "AUTH")


@dataclass(frozen=True, slots=True)
class ProviderModel:
    provider: str
    full_model: str
    source: str

    @property
    def model_name(self) -> str:
        return Settings.parse_model_name(self.full_model)


@dataclass(frozen=True, slots=True)
class SmokeConfig:
    root: Path
    results_dir: Path
    live: bool
    interactive: bool
    targets: frozenset[str]
    provider_matrix: frozenset[str]
    timeout_s: float
    prompt: str
    claude_bin: str
    worker_id: str
    settings: Settings

    @classmethod
    def load(cls) -> SmokeConfig:
        root = Path(__file__).resolve().parents[2]
        get_settings.cache_clear()
        settings = get_settings()
        return cls(
            root=root,
            results_dir=root / ".smoke-results",
            live=os.getenv("FCC_LIVE_SMOKE") == "1",
            interactive=os.getenv("FCC_SMOKE_INTERACTIVE") == "1",
            targets=_parse_targets(os.getenv("FCC_SMOKE_TARGETS")),
            provider_matrix=_parse_csv(os.getenv("FCC_SMOKE_PROVIDER_MATRIX")),
            timeout_s=float(os.getenv("FCC_SMOKE_TIMEOUT_S", "45")),
            prompt=os.getenv("FCC_SMOKE_PROMPT", "Reply with exactly: FCC_SMOKE_PONG"),
            claude_bin=os.getenv("FCC_SMOKE_CLAUDE_BIN", "claude"),
            worker_id=os.getenv("PYTEST_XDIST_WORKER", "main"),
            settings=settings,
        )

    def target_enabled(self, *names: str) -> bool:
        return any(name in self.targets for name in names)

    def provider_models(self) -> list[ProviderModel]:
        candidates = (
            ("MODEL", self.settings.model),
            ("MODEL_OPUS", self.settings.model_opus),
            ("MODEL_SONNET", self.settings.model_sonnet),
            ("MODEL_HAIKU", self.settings.model_haiku),
        )
        seen: set[str] = set()
        models: list[ProviderModel] = []
        for source, model in candidates:
            if not model or model in seen:
                continue
            provider = Settings.parse_provider_type(model)
            if self.provider_matrix and provider not in self.provider_matrix:
                continue
            if not self.has_provider_configuration(provider):
                continue
            seen.add(model)
            models.append(
                ProviderModel(provider=provider, full_model=model, source=source)
            )
        return models

    def has_provider_configuration(self, provider: str) -> bool:
        if provider == "nvidia_nim":
            return bool(self.settings.nvidia_nim_api_key.strip())
        if provider == "open_router":
            return bool(self.settings.open_router_api_key.strip())
        if provider == "deepseek":
            return bool(self.settings.deepseek_api_key.strip())
        if provider == "lmstudio":
            return bool(self.settings.lm_studio_base_url.strip())
        if provider == "llamacpp":
            return bool(self.settings.llamacpp_base_url.strip())
        return False


def _parse_csv(raw: str | None) -> frozenset[str]:
    if not raw:
        return frozenset()
    return frozenset(part.strip() for part in raw.split(",") if part.strip())


def _parse_targets(raw: str | None) -> frozenset[str]:
    if not raw:
        return DEFAULT_TARGETS
    parsed = _parse_csv(raw)
    if "all" in parsed:
        return ALL_TARGETS
    return parsed


def auth_headers(token: str | None = None) -> dict[str, str]:
    settings = get_settings()
    resolved = token if token is not None else settings.anthropic_auth_token
    headers = {
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    if resolved:
        headers["x-api-key"] = resolved
    return headers


def redacted(value: str, env: Mapping[str, str] | None = None) -> str:
    """Redact known secrets from a string before writing smoke artifacts."""
    if not value:
        return value

    source = env if env is not None else os.environ
    result = value
    for key, secret in source.items():
        if not secret or len(secret) < 4:
            continue
        if any(part in key.upper() for part in SECRET_KEY_PARTS):
            result = result.replace(secret, f"<redacted:{key}>")
    return result
