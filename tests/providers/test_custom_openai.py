"""Tests for Custom OpenAI provider."""

import json
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from providers.base import ProviderConfig
from providers.custom_openai import CustomOpenAIProvider
from providers.exceptions import AuthenticationError


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
    with patch("providers.openai_compat.GlobalRateLimiter") as mock:
        instance = mock.get_instance.return_value
        instance.wait_if_blocked = AsyncMock(return_value=False)

        async def _passthrough(fn, *args, **kwargs):
            return await fn(*args, **kwargs)

        instance.execute_with_retry = AsyncMock(side_effect=_passthrough)
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
    body = custom_openai_provider._build_request_body(req)

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
    body = custom_openai_provider._build_request_body(req)

    assert "tools" in body
    assert len(body["tools"]) == 1
    assert body["tools"][0]["type"] == "function"
    assert body["tools"][0]["function"]["name"] == "test_tool"


def test_handle_extra_reasoning(custom_openai_provider):
    """Test that extra reasoning handler returns empty iterator."""
    from providers.common import SSEBuilder

    sse = SSEBuilder("msg_id", "model", 0)
    delta = MagicMock()
    result = list(custom_openai_provider._handle_extra_reasoning(delta, sse))
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

        assert len(events) > 0
        assert "event: message_start" in events[0]

        text_content = ""
        for e in events:
            if "event: content_block_delta" in e and '"text_delta"' in e:
                for line in e.splitlines():
                    if line.startswith("data: "):
                        data = json.loads(line[6:])
                        if "delta" in data and "text" in data["delta"]:
                            text_content += data["delta"]["text"]

        assert "Hello World" in text_content


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
        # Error is emitted; message_stop/done indicates stream completed
        assert any("API failed" in e for e in events)
        assert any("message_stop" in e for e in events)


# Tests for provider registration and validation


def test_provider_requires_api_key():
    """Test that provider raises AuthenticationError when API key is missing."""

    from api.dependencies import _create_provider_for_type

    # Create a mock settings with empty API key
    mock_settings = Mock()
    mock_settings.custom_openai_api_key = ""
    mock_settings.custom_openai_base_url = "https://api.example.com/v1"
    mock_settings.provider_rate_limit = 10
    mock_settings.provider_rate_window = 60
    mock_settings.provider_max_concurrency = 5
    mock_settings.http_read_timeout = 300.0
    mock_settings.http_write_timeout = 10.0
    mock_settings.http_connect_timeout = 2.0

    with pytest.raises(AuthenticationError) as exc_info:
        _create_provider_for_type("custom_openai", mock_settings)
    assert "CUSTOM_OPENAI_API_KEY is not set" in str(exc_info.value)


def test_provider_requires_base_url():
    """Test that provider raises error when base URL is missing."""
    from api.dependencies import _create_provider_for_type

    # Create a mock settings with empty base URL
    mock_settings = Mock()
    mock_settings.custom_openai_api_key = "test_key"
    mock_settings.custom_openai_base_url = ""
    mock_settings.provider_rate_limit = 10
    mock_settings.provider_rate_window = 60
    mock_settings.provider_max_concurrency = 5
    mock_settings.http_read_timeout = 300.0
    mock_settings.http_write_timeout = 10.0
    mock_settings.http_connect_timeout = 2.0

    with pytest.raises(ValueError) as exc_info:
        _create_provider_for_type("custom_openai", mock_settings)
    assert "CUSTOM_OPENAI_BASE_URL is not set" in str(exc_info.value)


def test_provider_initialization_with_valid_config():
    """Test successful provider initialization with valid configuration."""
    from api.dependencies import _create_provider_for_type
    from providers.custom_openai import CustomOpenAIProvider

    # Create a mock settings with valid config
    mock_settings = Mock()
    mock_settings.custom_openai_api_key = "test_key"
    mock_settings.custom_openai_base_url = "https://api.example.com/v1"
    mock_settings.provider_rate_limit = 10
    mock_settings.provider_rate_window = 60
    mock_settings.provider_max_concurrency = 5
    mock_settings.http_read_timeout = 300.0
    mock_settings.http_write_timeout = 10.0
    mock_settings.http_connect_timeout = 2.0

    with patch("providers.openai_compat.AsyncOpenAI"):
        provider = _create_provider_for_type("custom_openai", mock_settings)
        assert isinstance(provider, CustomOpenAIProvider)
        assert provider._api_key == "test_key"
        assert provider._base_url == "https://api.example.com/v1"
