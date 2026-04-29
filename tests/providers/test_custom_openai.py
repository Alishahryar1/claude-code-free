"""Tests for Custom OpenAI provider."""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.anthropic import SSEBuilder
from core.anthropic.stream_contracts import (
    assert_anthropic_stream_contract,
    parse_sse_text,
    text_content,
)
from providers.base import ProviderConfig
from providers.custom_openai import CustomOpenAIProvider
from providers.exceptions import InvalidRequestError


class MockMessage:
    def __init__(self, role, content):
        self.role = role
        self.content = content


class MockRequest:
    def __init__(self, **kwargs):
        self.model = "custom_openai/gpt-4"
        self.messages = [MockMessage("user", "Hello")]
        self.max_tokens = 100
        self.temperature = 0.5
        self.top_p = 0.9
        self.system = "System prompt"
        self.stop_sequences = None
        self.tools = []
        self.extra_body = {}
        self.thinking = MagicMock()
        self.thinking.enabled = True
        for k, v in kwargs.items():
            setattr(self, k, v)


@pytest.fixture
def custom_openai_config():
    return ProviderConfig(
        api_key="test_custom_key",
        base_url="https://api.example.com/v1",
        rate_limit=10,
        rate_window=60,
    )


@pytest.fixture(autouse=True)
def mock_rate_limiter():
    """Mock the global rate limiter to prevent waiting."""

    @asynccontextmanager
    async def _slot():
        yield

    with patch("providers.openai_compat.GlobalRateLimiter") as mock:
        instance = mock.get_scoped_instance.return_value

        async def _passthrough(fn, *args, **kwargs):
            return await fn(*args, **kwargs)

        instance.execute_with_retry = AsyncMock(side_effect=_passthrough)
        instance.concurrency_slot.side_effect = _slot
        yield instance


@pytest.fixture
def custom_openai_provider(custom_openai_config):
    return CustomOpenAIProvider(custom_openai_config)


def test_init(custom_openai_config):
    """Test provider initialization."""
    with patch("providers.openai_compat.AsyncOpenAI") as mock_openai:
        provider = CustomOpenAIProvider(custom_openai_config)
        assert provider._api_key == "test_custom_key"
        assert provider._base_url == "https://api.example.com/v1"
        assert provider._provider_name == "CUSTOM_OPENAI"
        mock_openai.assert_called_once()


def test_init_with_custom_base_url():
    """Test provider with custom base URL."""
    config = ProviderConfig(
        api_key="test_key",
        base_url="https://custom.api.com/v1",
    )
    with patch("providers.openai_compat.AsyncOpenAI"):
        provider = CustomOpenAIProvider(config)
        assert provider._base_url == "https://custom.api.com/v1"


def test_build_request_body(custom_openai_provider):
    """Test request body building."""
    req = MockRequest()
    body = custom_openai_provider._build_request_body(req, thinking_enabled=True)

    assert body["model"] == "custom_openai/gpt-4"
    assert body["temperature"] == 0.5
    assert len(body["messages"]) == 2  # System + User
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][0]["content"] == "System prompt"


def test_build_request_body_with_tools(custom_openai_provider):
    """Test request body with tools."""
    mock_tool = MagicMock()
    mock_tool.name = "test_tool"
    mock_tool.description = "A test tool"
    mock_tool.input_schema = {"type": "object"}

    req = MockRequest(tools=[mock_tool])
    body = custom_openai_provider._build_request_body(req, thinking_enabled=True)

    assert "tools" in body
    assert len(body["tools"]) == 1
    assert body["tools"][0]["type"] == "function"
    assert body["tools"][0]["function"]["name"] == "test_tool"


def test_build_request_body_passes_extra_body_through(custom_openai_provider):
    req = MockRequest(extra_body={"provider_hint": "custom"})

    body = custom_openai_provider._build_request_body(req, thinking_enabled=True)

    assert body["extra_body"] == {"provider_hint": "custom"}


def test_build_request_body_rejects_reserved_extra_body_keys(custom_openai_provider):
    req = MockRequest(extra_body={"model": "hijack"})

    with pytest.raises(InvalidRequestError, match="model"):
        custom_openai_provider._build_request_body(req, thinking_enabled=True)


def test_handle_extra_reasoning(custom_openai_provider):
    """Test that extra reasoning handler returns empty iterator."""
    sse = SSEBuilder("msg_id", "model", 0)
    delta = MagicMock()
    result = list(
        custom_openai_provider._handle_extra_reasoning(
            delta, sse, thinking_enabled=True
        )
    )
    assert result == []


@pytest.mark.asyncio
async def test_stream_response_text(custom_openai_provider):
    """Test streaming text response."""
    req = MockRequest()

    mock_chunk1 = MagicMock()
    mock_chunk1.choices = [
        MagicMock(
            delta=MagicMock(content="Hello", reasoning_content=None),
            finish_reason=None,
        )
    ]
    mock_chunk1.usage = None

    mock_chunk2 = MagicMock()
    mock_chunk2.choices = [
        MagicMock(
            delta=MagicMock(content=" World", reasoning_content=None),
            finish_reason="stop",
        )
    ]
    mock_chunk2.usage = MagicMock(completion_tokens=10)

    async def mock_stream():
        yield mock_chunk1
        yield mock_chunk2

    with patch.object(
        custom_openai_provider._client.chat.completions,
        "create",
        new_callable=AsyncMock,
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = [e async for e in custom_openai_provider.stream_response(req)]

    parsed = parse_sse_text("".join(events))
    assert_anthropic_stream_contract(parsed)
    assert text_content(parsed) == "Hello World"


@pytest.mark.asyncio
async def test_stream_response_error_path(custom_openai_provider):
    """Test stream error handling."""
    req = MockRequest()

    async def mock_stream():
        raise RuntimeError("API failed")
        yield  # unreachable, makes it a generator

    with patch.object(
        custom_openai_provider._client.chat.completions,
        "create",
        new_callable=AsyncMock,
    ) as mock_create:
        mock_create.return_value = mock_stream()
        events = [e async for e in custom_openai_provider.stream_response(req)]

    parsed = parse_sse_text("".join(events))
    assert_anthropic_stream_contract(parsed, allow_error=True)
    assert not any(event.event == "error" for event in parsed)
    assert any("API failed" in event.raw for event in parsed)
