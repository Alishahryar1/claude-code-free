from __future__ import annotations

from urllib.parse import urljoin

import httpx
import pytest

from smoke.lib.config import SmokeConfig
from smoke.lib.e2e import ConversationDriver, SmokeServerDriver, assert_product_stream

pytestmark = [pytest.mark.live]


@pytest.mark.smoke_target("lmstudio")
def test_lmstudio_native_messages_e2e(smoke_config: SmokeConfig) -> None:
    _local_native_messages_e2e(
        smoke_config,
        provider="lmstudio",
        base_url=smoke_config.settings.lm_studio_base_url,
    )


@pytest.mark.smoke_target("llamacpp")
def test_llamacpp_native_messages_e2e(smoke_config: SmokeConfig) -> None:
    _local_native_messages_e2e(
        smoke_config,
        provider="llamacpp",
        base_url=smoke_config.settings.llamacpp_base_url,
    )


@pytest.mark.smoke_target("ollama")
def test_ollama_native_messages_e2e(smoke_config: SmokeConfig) -> None:
    _local_native_messages_e2e(
        smoke_config,
        provider="ollama",
        base_url=smoke_config.settings.ollama_base_url,
    )


def _local_native_messages_e2e(
    smoke_config: SmokeConfig,
    *,
    provider: str,
    base_url: str,
) -> None:
    if not base_url.strip():
        pytest.skip(f"missing_env: {provider} base URL is not configured")

    model_id = (
        _first_ollama_tag_model_id(base_url)
        if provider == "ollama"
        else (_first_non_ollama_model_id(provider, base_url))
    )

    with SmokeServerDriver(
        smoke_config,
        name=f"product-{provider}-native",
        env_overrides={"MODEL": f"{provider}/{model_id}", "MESSAGING_PLATFORM": "none"},
    ).run() as server:
        turn = ConversationDriver(server, smoke_config).ask(
            "Reply with one short sentence."
        )

    assert_product_stream(turn.events)


def _first_non_ollama_model_id(provider: str, base_url: str) -> str:
    models_url = urljoin(base_url.rstrip("/") + "/", "models")
    try:
        response = httpx.get(models_url, timeout=5)
    except httpx.ConnectError as exc:
        pytest.skip(f"upstream_unavailable: {provider} models endpoint: {exc}")
    except httpx.TimeoutException as exc:
        pytest.skip(f"upstream_unavailable: {provider} models endpoint: {exc}")
    assert response.status_code == 200, response.text
    payload = response.json()
    data = payload.get("data") if isinstance(payload, dict) else None
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and isinstance(item.get("id"), str):
                return item["id"]
        pytest.skip(f"upstream_unavailable: {provider} has no local models")
    pytest.fail("product_failure: local /models did not expose a model id")


def _first_ollama_tag_model_id(base_url: str) -> str:
    tags_url = f"{_ollama_root_url(base_url)}/api/tags"
    try:
        response = httpx.get(tags_url, timeout=5)
    except httpx.ConnectError as exc:
        pytest.skip(f"upstream_unavailable: ollama tags endpoint: {exc}")
    except httpx.TimeoutException as exc:
        pytest.skip(f"upstream_unavailable: ollama tags endpoint: {exc}")

    assert response.status_code == 200, response.text
    payload = response.json()
    models = payload.get("models") if isinstance(payload, dict) else None
    if isinstance(models, list):
        for item in models:
            if isinstance(item, dict) and isinstance(item.get("name"), str):
                return item["name"]
        pytest.skip("upstream_unavailable: ollama has no pulled models")
    pytest.fail("product_failure: ollama /api/tags did not expose models")


def _ollama_root_url(base_url: str) -> str:
    return base_url.rstrip("/")
