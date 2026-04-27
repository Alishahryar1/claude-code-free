"""Tests for Google Vertex AI Express provider."""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from api.models.anthropic import Message, MessagesRequest
from config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from config.provider_catalog import GOOGLE_VERTEX_DEFAULT_BASE
from providers.base import ProviderConfig
from providers.google_vertex import VertexExpressProvider


@pytest.fixture
def vertex_config():
    return ProviderConfig(
        api_key="test_vertex_key",
        base_url=GOOGLE_VERTEX_DEFAULT_BASE,
        rate_limit=10,
        rate_window=60,
        enable_thinking=True,
    )


@pytest.fixture(autouse=True)
def mock_rate_limiter():
    @asynccontextmanager
    async def _slot():
        yield

    with patch("providers.anthropic_messages.GlobalRateLimiter") as mock:
        instance = mock.get_scoped_instance.return_value

        async def _passthrough(fn, *args, **kwargs):
            return await fn(*args, **kwargs)

        instance.execute_with_retry = AsyncMock(side_effect=_passthrough)
        instance.concurrency_slot.side_effect = _slot
        yield instance


@pytest.fixture
def vertex_provider(vertex_config):
    return VertexExpressProvider(vertex_config)


def test_default_base_url():
    assert GOOGLE_VERTEX_DEFAULT_BASE == "https://aiplatform.googleapis.com/v1"


def test_init(vertex_config):
    with patch("httpx.AsyncClient") as mock_client:
        provider = VertexExpressProvider(vertex_config)
    assert provider._api_key == "test_vertex_key"
    assert provider._base_url == "https://aiplatform.googleapis.com/v1"
    assert mock_client.called


def test_request_headers(vertex_provider):
    h = vertex_provider._request_headers()
    assert h["Content-Type"] == "application/json"
    assert "Accept" not in h


def test_get_model_path(vertex_provider):
    path = vertex_provider._get_model_path("gemini-2.5-flash")
    assert path == "/publishers/google/models/gemini-2.5-flash:streamGenerateContent"


def test_get_model_path_with_different_model(vertex_provider):
    path = vertex_provider._get_model_path("gemini-3-pro-preview")
    assert path == "/publishers/google/models/gemini-3-pro-preview:streamGenerateContent"


def test_build_request_body_vertex_format(vertex_provider):
    request = MessagesRequest(
        model="gemini-2.5-flash",
        max_tokens=100,
        messages=[Message(role="user", content="Hello")],
        system="You are a helpful assistant.",
    )
    body = vertex_provider._build_request_body(request)

    assert body["model"] == "gemini-2.5-flash"
    assert "contents" in body
    assert len(body["contents"]) >= 1
    assert body["contents"][0]["role"] == "user"
    assert "parts" in body["contents"][0]
    assert body["contents"][0]["parts"][0]["text"] == "Hello"

    assert "generationConfig" in body
    assert body["generationConfig"]["maxOutputTokens"] == 100


def test_build_request_body_with_system_prompt(vertex_provider):
    request = MessagesRequest(
        model="gemini-2.5-flash",
        max_tokens=50,
        messages=[Message(role="user", content="Hi")],
        system="You are a chatbot.",
    )
    body = vertex_provider._build_request_body(request)

    assert "contents" in body
    first_msg = body["contents"][0]
    assert first_msg["role"] == "user"


def test_build_request_body_default_max_tokens(vertex_provider):
    request = MessagesRequest(
        model="gemini-2.5-flash",
        messages=[Message(role="user", content="x")],
    )
    body = vertex_provider._build_request_body(request)

    assert body["generationConfig"]["maxOutputTokens"] == ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS


def test_build_request_body_with_thinking(vertex_provider):
    request = MessagesRequest.model_validate(
        {
            "model": "gemini-2.5-flash",
            "messages": [{"role": "user", "content": "x"}],
            "thinking": {"type": "enabled", "budget_tokens": 4096},
        }
    )
    body = vertex_provider._build_request_body(request)

    assert "generationConfig" in body
    assert "thinkingSettings" in body["generationConfig"]
    assert body["generationConfig"]["thinkingSettings"]["mode"] == "enabled"
    assert body["generationConfig"]["thinkingSettings"]["budgetTokens"] == 4096


def test_build_request_body_with_temperature(vertex_provider):
    request = MessagesRequest.model_validate(
        {
            "model": "gemini-2.5-flash",
            "messages": [{"role": "user", "content": "x"}],
            "temperature": 0.7,
        }
    )
    body = vertex_provider._build_request_body(request)

    assert "generationConfig" in body
    assert body["generationConfig"]["temperature"] == 0.7


def test_build_request_body_with_top_p(vertex_provider):
    request = MessagesRequest.model_validate(
        {
            "model": "gemini-2.5-flash",
            "messages": [{"role": "user", "content": "x"}],
            "top_p": 0.9,
        }
    )
    body = vertex_provider._build_request_body(request)

    assert "generationConfig" in body
    assert body["generationConfig"]["topP"] == 0.9


def test_build_request_body_respects_global_thinking_disable():
    provider = VertexExpressProvider(
        ProviderConfig(
            api_key="test",
            base_url=GOOGLE_VERTEX_DEFAULT_BASE,
            enable_thinking=False,
        )
    )
    request = MessagesRequest.model_validate(
        {
            "model": "gemini-2.5-flash",
            "messages": [{"role": "user", "content": "x"}],
            "thinking": {"type": "enabled", "budget_tokens": 2000},
        }
    )
    body = provider._build_request_body(request)
    assert "thinkingSettings" not in body.get("generationConfig", {})