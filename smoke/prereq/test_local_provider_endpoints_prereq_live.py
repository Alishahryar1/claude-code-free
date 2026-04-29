from __future__ import annotations

import httpx
import pytest

from smoke.lib.config import SmokeConfig
from smoke.lib.skips import skip_if_upstream_unavailable_exception


@pytest.mark.live
@pytest.mark.smoke_target("lmstudio")
def test_lmstudio_models_endpoint_when_available(smoke_config: SmokeConfig) -> None:
    _assert_models_endpoint(
        smoke_config.settings.lm_studio_base_url,
        timeout_s=smoke_config.timeout_s,
        provider_name="LM Studio",
    )


@pytest.mark.live
@pytest.mark.smoke_target("llamacpp")
def test_llamacpp_models_endpoint_when_available(smoke_config: SmokeConfig) -> None:
    _assert_models_endpoint(
        smoke_config.settings.llamacpp_base_url,
        timeout_s=smoke_config.timeout_s,
        provider_name="llama.cpp",
    )


@pytest.mark.live
@pytest.mark.smoke_target("ollama")
def test_ollama_models_endpoint_when_available(smoke_config: SmokeConfig) -> None:
    _assert_ollama_tags_endpoint(
        smoke_config.settings.ollama_base_url, timeout_s=smoke_config.timeout_s
    )


def _assert_models_endpoint(
    base_url: str, *, timeout_s: float, provider_name: str
) -> None:
    url = f"{base_url.rstrip('/')}/models"
    try:
        response = httpx.get(url, timeout=timeout_s)
    except Exception as exc:
        skip_if_upstream_unavailable_exception(exc)
        raise

    if response.status_code in {404, 405, 502, 503}:
        pytest.skip(
            f"upstream_unavailable: {provider_name} models endpoint "
            f"{url} returned HTTP {response.status_code}"
        )

    assert response.status_code == 200, response.text
    payload = response.json()
    data = payload.get("data")
    if isinstance(data, list) and data:
        return
    if isinstance(data, list):
        pytest.skip(f"upstream_unavailable: {provider_name} has no local models")
    assert isinstance(data, list), payload


def _assert_ollama_tags_endpoint(base_url: str, *, timeout_s: float) -> None:
    url = f"{_ollama_root_url(base_url)}/api/tags"
    try:
        response = httpx.get(url, timeout=timeout_s)
    except Exception as exc:
        skip_if_upstream_unavailable_exception(exc)
        raise

    if response.status_code in {404, 405, 502, 503}:
        pytest.skip(
            f"upstream_unavailable: Ollama tags endpoint {url} "
            f"returned HTTP {response.status_code}"
        )

    assert response.status_code == 200, response.text
    models = response.json().get("models")
    if isinstance(models, list) and models:
        return
    pytest.skip("upstream_unavailable: Ollama has no pulled models")


def _ollama_root_url(base_url: str) -> str:
    return base_url.rstrip("/")
