"""Tests for the rich ``/health`` payload."""

from __future__ import annotations

import json
from collections.abc import Generator
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.app import create_app
from api.health import (
    _credential_present,
    _provider_referenced,
    build_health_payload,
)
from config.provider_catalog import PROVIDER_CATALOG
from config.settings import Settings
from providers.rate_limit import GlobalRateLimiter


@pytest.fixture
def app() -> FastAPI:
    return create_app()


@pytest.fixture
def client(app: FastAPI) -> Generator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def _reset_rate_limiters() -> Generator[None]:
    """Clear scoped limiters between tests so snapshots stay deterministic."""
    GlobalRateLimiter.reset_instance()
    yield
    GlobalRateLimiter.reset_instance()


# =============================================================================
# Top-level payload shape
# =============================================================================
def test_health_returns_rich_payload(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()

    expected_top_level = {
        "status",
        "version",
        "uptime_seconds",
        "server",
        "models",
        "providers",
        "messaging",
        "web_tools",
        "optimizations",
    }
    assert expected_top_level.issubset(body.keys())
    assert body["status"] == "healthy"
    assert isinstance(body["uptime_seconds"], (int, float))
    assert body["uptime_seconds"] >= 0


def test_health_models_section_reports_routing(client: TestClient) -> None:
    body = client.get("/health").json()
    models = body["models"]

    # conftest sets MODEL=nvidia_nim/test-model
    assert models["default"] == "nvidia_nim/test-model"
    assert "thinking" in models
    for tier in ("default", "opus", "sonnet", "haiku"):
        assert tier in models["thinking"]


def test_health_lists_every_provider_in_catalog(client: TestClient) -> None:
    body = client.get("/health").json()
    assert set(body["providers"].keys()) == set(PROVIDER_CATALOG.keys())
    for snapshot in body["providers"].values():
        for required in (
            "transport",
            "configured",
            "credential_present",
            "instantiated",
        ):
            assert required in snapshot


def test_health_messaging_and_web_tools_sections(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["messaging"]["platform"] in {"discord", "telegram", "none"}
    assert isinstance(body["messaging"]["running"], bool)
    assert isinstance(body["web_tools"]["enabled"], bool)
    assert isinstance(body["web_tools"]["allowed_schemes"], list)


# =============================================================================
# Sensitive value redaction
# =============================================================================
def test_health_never_echoes_auth_token_value(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The literal auth token must never appear in the response body."""
    secret = "super-secret-auth-token-do-not-leak"
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", secret)
    Settings.model_config = {**Settings.model_config, "env_file": None}
    settings = Settings()

    payload = build_health_payload(app, settings)
    serialized = json.dumps(payload)

    assert secret not in serialized
    assert payload["server"]["auth_token_configured"] is True


def test_health_never_echoes_api_keys(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    nim_key = "nvapi-leak-canary-1234"
    or_key = "sk-or-leak-canary-5678"
    ds_key = "ds-leak-canary-9012"
    monkeypatch.setenv("NVIDIA_NIM_API_KEY", nim_key)
    monkeypatch.setenv("OPENROUTER_API_KEY", or_key)
    monkeypatch.setenv("DEEPSEEK_API_KEY", ds_key)
    Settings.model_config = {**Settings.model_config, "env_file": None}
    settings = Settings()

    serialized = json.dumps(build_health_payload(app, settings))

    assert nim_key not in serialized
    assert or_key not in serialized
    assert ds_key not in serialized


def test_health_never_echoes_bot_tokens(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    tg_token = "telegram-bot-token-leak-canary"
    discord_token = "discord-bot-token-leak-canary"
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", tg_token)
    monkeypatch.setenv("DISCORD_BOT_TOKEN", discord_token)
    Settings.model_config = {**Settings.model_config, "env_file": None}
    settings = Settings()

    serialized = json.dumps(build_health_payload(app, settings))

    assert tg_token not in serialized
    assert discord_token not in serialized


def test_auth_token_configured_false_when_unset(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "")
    Settings.model_config = {**Settings.model_config, "env_file": None}
    settings = Settings()

    payload = build_health_payload(app, settings)
    assert payload["server"]["auth_token_configured"] is False


# =============================================================================
# Auth gating
# =============================================================================
def test_health_requires_auth_when_token_set(
    app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "the-token")
    # Force a fresh Settings instance for the dependency override.
    from api.dependencies import get_settings as get_settings_dep

    Settings.model_config = {**Settings.model_config, "env_file": None}
    fresh_settings = Settings()
    app.dependency_overrides[get_settings_dep] = lambda: fresh_settings

    try:
        with TestClient(app) as test_client:
            unauth = test_client.get("/health")
            assert unauth.status_code == 401

            authed = test_client.get("/health", headers={"x-api-key": "the-token"})
            assert authed.status_code == 200
            assert authed.json()["status"] == "healthy"
    finally:
        app.dependency_overrides.pop(get_settings_dep, None)


def test_head_health_remains_unauthenticated(client: TestClient) -> None:
    """HEAD /health is the liveness path and must never gate on auth."""
    response = client.head("/health")
    assert response.status_code == 204
    assert "Allow" in response.headers


# =============================================================================
# Provider snapshot helpers
# =============================================================================
def test_provider_referenced_matches_default_model_prefix() -> None:
    settings = Settings(model="nvidia_nim/foo/bar")
    assert _provider_referenced("nvidia_nim", settings) is True
    assert _provider_referenced("open_router", settings) is False


def test_provider_referenced_picks_up_per_tier_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MODEL", "nvidia_nim/x")
    monkeypatch.setenv("MODEL_OPUS", "open_router/y")
    monkeypatch.setenv("MODEL_SONNET", "deepseek/z")
    monkeypatch.setenv("MODEL_HAIKU", "ollama/w")
    Settings.model_config = {**Settings.model_config, "env_file": None}
    settings = Settings()
    assert _provider_referenced("nvidia_nim", settings) is True
    assert _provider_referenced("open_router", settings) is True
    assert _provider_referenced("deepseek", settings) is True
    assert _provider_referenced("ollama", settings) is True
    assert _provider_referenced("lmstudio", settings) is False


def test_credential_present_for_static_credential_providers() -> None:
    """Local providers (lmstudio, llamacpp, ollama) ship static creds."""
    settings = Settings(model="nvidia_nim/x")
    for static_id in ("lmstudio", "llamacpp", "ollama"):
        assert _credential_present(PROVIDER_CATALOG[static_id], settings) is True


def test_credential_present_reflects_env_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NVIDIA_NIM_API_KEY", "nvapi-real")
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    Settings.model_config = {**Settings.model_config, "env_file": None}
    settings = Settings()

    assert _credential_present(PROVIDER_CATALOG["nvidia_nim"], settings) is True
    assert _credential_present(PROVIDER_CATALOG["open_router"], settings) is False


def test_provider_snapshot_includes_rate_limit_when_limiter_exists(
    app: FastAPI,
) -> None:
    """Once a provider has run, its scoped limiter shows up in /health."""
    GlobalRateLimiter.get_scoped_instance(
        "nvidia_nim", rate_limit=7, rate_window=11.0, max_concurrency=3
    )
    settings = Settings(model="nvidia_nim/test")
    payload = build_health_payload(app, settings)

    nim_snap: dict[str, Any] = payload["providers"]["nvidia_nim"]
    assert "rate_limit" in nim_snap
    assert nim_snap["rate_limit"] == {
        "limit": 7,
        "window_seconds": 11.0,
        "concurrency_max": 3,
        "reactive_blocked": False,
        "blocked_remaining_s": 0.0,
    }
    # Providers that have not run should NOT have a rate_limit entry.
    assert "rate_limit" not in payload["providers"]["open_router"]


def test_provider_snapshot_reports_reactive_block(app: FastAPI) -> None:
    limiter = GlobalRateLimiter.get_scoped_instance("nvidia_nim", rate_limit=1)
    limiter.set_blocked(seconds=60)
    settings = Settings(model="nvidia_nim/test")

    payload = build_health_payload(app, settings)
    rate_limit = payload["providers"]["nvidia_nim"]["rate_limit"]
    assert rate_limit["reactive_blocked"] is True
    assert rate_limit["blocked_remaining_s"] > 0
