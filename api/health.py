"""Rich ``/health`` payload helpers.

Returns a runtime configuration snapshot so users can debug misrouted models,
missing credentials, and rate-limiter state without reading ``server.log`` or
opening source. Sensitive values (API keys, auth tokens, bot tokens) are NEVER
echoed; only ``*_present`` / ``*_configured`` booleans.
"""

from __future__ import annotations

import time
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from fastapi import FastAPI

from config.provider_catalog import PROVIDER_CATALOG, ProviderDescriptor
from config.settings import Settings
from providers.registry import ProviderRegistry, provider_rate_limit_snapshot

# Captured at import time so uptime is reported relative to process start.
_PROCESS_START_MONOTONIC = time.monotonic()


def _safe_version() -> str:
    """Return the installed package version or ``'unknown'`` when not packaged."""
    try:
        return version("free-claude-code")
    except PackageNotFoundError:
        return "unknown"


def _credential_present(descriptor: ProviderDescriptor, settings: Settings) -> bool:
    """True when this provider has the credential it needs to instantiate.

    Mirrors :func:`providers.registry._credential_for` without raising so the
    health endpoint can report ``credential_present: false`` instead of erroring.
    """
    if descriptor.static_credential is not None:
        return True
    if descriptor.credential_attr is None:
        # No credential mechanism declared; treat as "no credential needed".
        return descriptor.credential_env is None
    value = getattr(settings, descriptor.credential_attr, "")
    return isinstance(value, str) and bool(value.strip())


def _provider_referenced(provider_id: str, settings: Settings) -> bool:
    """True when any configured ``MODEL`` / ``MODEL_*`` uses this provider prefix."""
    candidates = (
        settings.model,
        settings.model_opus,
        settings.model_sonnet,
        settings.model_haiku,
    )
    for candidate in candidates:
        if not candidate:
            continue
        if "/" not in candidate:
            continue
        if Settings.parse_provider_type(candidate) == provider_id:
            return True
    return False


def _provider_snapshot(
    provider_id: str,
    descriptor: ProviderDescriptor,
    settings: Settings,
    registry: ProviderRegistry | None,
) -> dict[str, Any]:
    """Build the per-provider snapshot dict for the health payload."""
    snapshot: dict[str, Any] = {
        "transport": descriptor.transport_type,
        "configured": _provider_referenced(provider_id, settings),
        "credential_present": _credential_present(descriptor, settings),
        "instantiated": registry.is_cached(provider_id) if registry else False,
    }
    rate_limit = provider_rate_limit_snapshot(provider_id)
    if rate_limit is not None:
        snapshot["rate_limit"] = rate_limit
    return snapshot


def build_health_payload(app: FastAPI, settings: Settings) -> dict[str, Any]:
    """Return the full ``/health`` JSON payload.

    Always safe to expose: never includes API keys, bot tokens, or the
    ``ANTHROPIC_AUTH_TOKEN`` value itself — only ``*_configured`` booleans.
    """
    registry: ProviderRegistry | None = getattr(app.state, "provider_registry", None)
    handler = getattr(app.state, "message_handler", None)

    return {
        "status": "healthy",
        "version": _safe_version(),
        "uptime_seconds": round(time.monotonic() - _PROCESS_START_MONOTONIC, 1),
        "server": {
            "host": settings.host,
            "port": settings.port,
            "auth_token_configured": bool(settings.anthropic_auth_token),
        },
        "models": {
            "default": settings.model or None,
            "opus": settings.model_opus,
            "sonnet": settings.model_sonnet,
            "haiku": settings.model_haiku,
            "thinking": {
                "default": settings.enable_model_thinking,
                "opus": settings.enable_opus_thinking,
                "sonnet": settings.enable_sonnet_thinking,
                "haiku": settings.enable_haiku_thinking,
            },
        },
        "providers": {
            provider_id: _provider_snapshot(provider_id, descriptor, settings, registry)
            for provider_id, descriptor in PROVIDER_CATALOG.items()
        },
        "messaging": {
            "platform": settings.messaging_platform,
            "running": handler is not None,
        },
        "web_tools": {
            "enabled": settings.enable_web_server_tools,
            "private_networks_allowed": settings.web_fetch_allow_private_networks,
            "allowed_schemes": sorted(settings.web_fetch_allowed_scheme_set()),
        },
        "optimizations": {
            "fast_prefix_detection": settings.fast_prefix_detection,
            "network_probe_mock": settings.enable_network_probe_mock,
            "title_generation_skip": settings.enable_title_generation_skip,
            "suggestion_mode_skip": settings.enable_suggestion_mode_skip,
            "filepath_extraction_mock": settings.enable_filepath_extraction_mock,
        },
    }
