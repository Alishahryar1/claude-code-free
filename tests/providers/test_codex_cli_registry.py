from __future__ import annotations

from config.settings import Settings
from providers.codex_cli import CodexCliProvider
from providers.registry import create_provider


def test_model_codex_cli_default_validates(monkeypatch):
    monkeypatch.setitem(Settings.model_config, "env_file", ())
    monkeypatch.setenv("MODEL", "codex_cli/default")

    settings = Settings()

    assert settings.model == "codex_cli/default"
    assert settings.provider_type == "codex_cli"
    assert settings.model_name == "default"


def test_registry_builds_codex_cli_without_api_key(monkeypatch):
    monkeypatch.setitem(Settings.model_config, "env_file", ())
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("MODEL", "codex_cli/default")
    monkeypatch.setenv("CODEX_CLI_BIN", "codex-test")
    monkeypatch.setenv("CODEX_WORKSPACE", "/tmp")
    monkeypatch.setenv("CODEX_TIMEOUT", "12.5")

    settings = Settings()
    provider = create_provider("codex_cli", settings)

    assert isinstance(provider, CodexCliProvider)
    assert provider.codex_bin == "codex-test"
    assert provider.workspace == "/tmp"
    assert provider.timeout == 12.5
