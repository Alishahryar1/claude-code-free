from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from smoke.lib.config import DEFAULT_TARGETS, TARGET_REQUIRED_ENV, SmokeConfig


def _settings(**overrides):
    values = {
        "model": "ollama/llama3.1",
        "model_opus": None,
        "model_sonnet": None,
        "model_haiku": None,
        "nvidia_nim_api_key": "",
        "open_router_api_key": "",
        "deepseek_api_key": "",
        "lm_studio_base_url": "",
        "llamacpp_base_url": "",
        "ollama_base_url": "http://localhost:11434",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _smoke_config(**overrides) -> SmokeConfig:
    values = {
        "root": Path("."),
        "results_dir": Path(".smoke-results"),
        "live": False,
        "interactive": False,
        "targets": DEFAULT_TARGETS,
        "provider_matrix": frozenset(),
        "timeout_s": 45.0,
        "prompt": "Reply with exactly: FCC_SMOKE_PONG",
        "claude_bin": "claude",
        "worker_id": "main",
        "settings": _settings(),
    }
    values.update(overrides)
    return SmokeConfig(**values)


def test_ollama_is_default_smoke_target() -> None:
    assert "ollama" in DEFAULT_TARGETS
    assert "ollama" in TARGET_REQUIRED_ENV


def test_ollama_provider_configuration_uses_base_url() -> None:
    config = _smoke_config()

    assert config.has_provider_configuration("ollama")
    assert config.provider_models()[0].full_model == "ollama/llama3.1"


def test_ollama_provider_matrix_filters_models() -> None:
    config = _smoke_config(provider_matrix=frozenset({"ollama"}))

    assert [model.provider for model in config.provider_models()] == ["ollama"]
