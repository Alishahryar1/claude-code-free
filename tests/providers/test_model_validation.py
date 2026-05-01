from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from config.nim import NimSettings
from config.settings import Settings
from providers.base import BaseProvider, ProviderConfig
from providers.deepseek import DeepSeekProvider
from providers.exceptions import ModelListResponseError, ServiceUnavailableError
from providers.lmstudio import LMStudioProvider
from providers.nvidia_nim import NvidiaNimProvider
from providers.ollama import OllamaProvider
from providers.registry import ProviderRegistry


def _settings(
    *,
    model: str = "nvidia_nim/org/nim-model",
    model_opus: str | None = None,
    model_sonnet: str | None = None,
    model_haiku: str | None = None,
) -> Settings:
    return Settings.model_construct(
        model=model,
        model_opus=model_opus,
        model_sonnet=model_sonnet,
        model_haiku=model_haiku,
        log_api_error_tracebacks=False,
    )


def _response(status_code: int, payload: object) -> httpx.Response:
    return httpx.Response(
        status_code,
        json=payload,
        request=httpx.Request("GET", "https://example.test/models"),
    )


@pytest.mark.asyncio
async def test_nim_lists_openai_compatible_model_ids() -> None:
    config = ProviderConfig(api_key="test-key")
    with patch("providers.openai_compat.AsyncOpenAI"):
        provider = NvidiaNimProvider(config, nim_settings=NimSettings())

    with patch.object(
        provider._client.models,
        "list",
        new_callable=AsyncMock,
        return_value=SimpleNamespace(data=[SimpleNamespace(id="nvidia/model")]),
    ):
        assert await provider.list_model_ids() == frozenset({"nvidia/model"})


@pytest.mark.asyncio
async def test_native_openai_compatible_provider_lists_model_ids() -> None:
    provider = LMStudioProvider(
        ProviderConfig(api_key="lm-studio", base_url="http://localhost:1234/v1")
    )
    with patch.object(
        provider._client,
        "get",
        new_callable=AsyncMock,
        return_value=_response(200, {"data": [{"id": "local/model"}]}),
    ) as mock_get:
        assert await provider.list_model_ids() == frozenset({"local/model"})

    mock_get.assert_awaited_once_with("/models", headers={})


@pytest.mark.asyncio
async def test_deepseek_lists_models_from_root_endpoint() -> None:
    provider = DeepSeekProvider(ProviderConfig(api_key="deepseek-key"))
    with patch.object(
        provider._client,
        "get",
        new_callable=AsyncMock,
        return_value=_response(200, {"data": [{"id": "deepseek-chat"}]}),
    ) as mock_get:
        assert await provider.list_model_ids() == frozenset({"deepseek-chat"})

    mock_get.assert_awaited_once_with(
        "https://api.deepseek.com/models",
        headers={"Authorization": "Bearer deepseek-key"},
    )


@pytest.mark.asyncio
async def test_ollama_lists_native_tag_model_ids() -> None:
    provider = OllamaProvider(
        ProviderConfig(api_key="ollama", base_url="http://localhost:11434")
    )
    with patch.object(
        provider._client,
        "get",
        new_callable=AsyncMock,
        return_value=_response(
            200,
            {
                "models": [
                    {"name": "llama3.1:latest", "model": "llama3.1:latest"},
                    {"name": "qwen3"},
                ]
            },
        ),
    ) as mock_get:
        assert await provider.list_model_ids() == frozenset(
            {"llama3.1:latest", "qwen3"}
        )

    mock_get.assert_awaited_once_with("http://localhost:11434/api/tags")


@pytest.mark.asyncio
async def test_model_listing_rejects_malformed_payload() -> None:
    provider = LMStudioProvider(
        ProviderConfig(api_key="lm-studio", base_url="http://localhost:1234/v1")
    )
    with (
        patch.object(
            provider._client,
            "get",
            new_callable=AsyncMock,
            return_value=_response(200, {"data": [{}]}),
        ),
        pytest.raises(ModelListResponseError, match="malformed"),
    ):
        await provider.list_model_ids()


@pytest.mark.asyncio
async def test_model_listing_raises_http_status_errors() -> None:
    provider = LMStudioProvider(
        ProviderConfig(api_key="lm-studio", base_url="http://localhost:1234/v1")
    )
    with (
        patch.object(
            provider._client,
            "get",
            new_callable=AsyncMock,
            return_value=_response(503, {"error": "down"}),
        ),
        pytest.raises(httpx.HTTPStatusError),
    ):
        await provider.list_model_ids()


class FakeProvider(BaseProvider):
    def __init__(
        self,
        model_ids: frozenset[str] | None = None,
        *,
        error: BaseException | None = None,
        started: asyncio.Event | None = None,
        peer_started: asyncio.Event | None = None,
    ):
        super().__init__(ProviderConfig(api_key="test"))
        self._model_ids = model_ids or frozenset()
        self._error = error
        self._started = started
        self._peer_started = peer_started
        self.cleaned = False

    async def cleanup(self) -> None:
        self.cleaned = True

    async def list_model_ids(self) -> frozenset[str]:
        if self._started is not None:
            self._started.set()
        if self._peer_started is not None:
            await self._peer_started.wait()
        if self._error is not None:
            raise self._error
        return self._model_ids

    async def stream_response(
        self,
        request: Any,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        thinking_enabled: bool | None = None,
    ) -> AsyncIterator[str]:
        if False:
            yield ""


@pytest.mark.asyncio
async def test_registry_validation_succeeds_for_all_configured_models() -> None:
    registry = ProviderRegistry(
        {
            "nvidia_nim": FakeProvider(frozenset({"org/nim-model"})),
            "open_router": FakeProvider(frozenset({"anthropic/claude-opus"})),
        }
    )
    settings = _settings(model_opus="open_router/anthropic/claude-opus")

    await registry.validate_configured_models(settings)


@pytest.mark.asyncio
async def test_registry_validation_reports_missing_model_with_sources() -> None:
    registry = ProviderRegistry(
        {"nvidia_nim": FakeProvider(frozenset({"org/different-model"}))}
    )
    settings = _settings(model_sonnet="nvidia_nim/org/nim-model")

    with pytest.raises(ServiceUnavailableError) as exc_info:
        await registry.validate_configured_models(settings)

    message = exc_info.value.message
    assert "sources=MODEL,MODEL_SONNET" in message
    assert "provider=nvidia_nim" in message
    assert "model=org/nim-model" in message
    assert "problem=missing model" in message


@pytest.mark.asyncio
async def test_registry_validation_aggregates_multiple_failures() -> None:
    registry = ProviderRegistry(
        {
            "nvidia_nim": FakeProvider(frozenset({"org/different-model"})),
            "open_router": FakeProvider(
                error=ModelListResponseError("bad model-list shape")
            ),
        }
    )
    settings = _settings(model_opus="open_router/anthropic/claude-opus")

    with pytest.raises(ServiceUnavailableError) as exc_info:
        await registry.validate_configured_models(settings)

    message = exc_info.value.message
    assert "sources=MODEL provider=nvidia_nim model=org/nim-model" in message
    assert "problem=missing model" in message
    assert "sources=MODEL_OPUS provider=open_router model=anthropic/claude-opus" in (
        message
    )
    assert "problem=malformed model-list response" in message


@pytest.mark.asyncio
async def test_registry_validation_queries_providers_concurrently() -> None:
    nim_started = asyncio.Event()
    router_started = asyncio.Event()
    registry = ProviderRegistry(
        {
            "nvidia_nim": FakeProvider(
                frozenset({"org/nim-model"}),
                started=nim_started,
                peer_started=router_started,
            ),
            "open_router": FakeProvider(
                frozenset({"anthropic/claude-opus"}),
                started=router_started,
                peer_started=nim_started,
            ),
        }
    )
    settings = _settings(model_opus="open_router/anthropic/claude-opus")

    await asyncio.wait_for(registry.validate_configured_models(settings), timeout=1.0)


@pytest.mark.asyncio
async def test_registry_validation_rejects_nim_model_without_vendor_prefix() -> None:
    registry = ProviderRegistry({"nvidia_nim": FakeProvider(frozenset({"google/gemma-3-27b-it"}))})
    settings = _settings(model="nvidia_nim/gemma-3-27b-it")

    with pytest.raises(ServiceUnavailableError) as exc_info:
        await registry.validate_configured_models(settings)

    message = exc_info.value.message
    assert "provider=nvidia_nim" in message
    assert "model=gemma-3-27b-it" in message
    assert "vendor prefix" in message
    assert "build.nvidia.com" in message


@pytest.mark.asyncio
async def test_registry_validation_accepts_nim_model_with_vendor_prefix() -> None:
    registry = ProviderRegistry({"nvidia_nim": FakeProvider(frozenset({"google/gemma-3-27b-it"}))})
    settings = _settings(model="nvidia_nim/google/gemma-3-27b-it")

    await registry.validate_configured_models(settings)
