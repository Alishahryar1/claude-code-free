"""Tests for MiniMax provider."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from providers.base import ProviderConfig
from providers.minimax import MINIMAX_BASE_URL, MinimaxProvider


class MockMessage:
    def __init__(self, role, content):
        self.role = role
        self.content = content


class MockRequest:
    def __init__(self, **kwargs):
        self.model = "MiniMax-M2.7"
        self.messages = [MockMessage("user", "Hello")]
        self.max_tokens = 100
        self.temperature = 1.0
        self.top_p = 0.9
        self.system = "System prompt"
        self.stop_sequences = None
        self.tools = []
        self.extra_body = {}
        self.thinking = MagicMock()
        self.thinking.enabled = True
        for key, value in kwargs.items():
            setattr(self, key, value)


@pytest.fixture
def minimax_config():
    return ProviderConfig(
        api_key="test_minimax_key",
        base_url=MINIMAX_BASE_URL,
        rate_limit=10,
        rate_window=60,
        enable_thinking=True,
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
def minimax_provider(minimax_config):
    return MinimaxProvider(minimax_config)


def test_init(minimax_config):
    """Test provider initialization."""
    with patch("providers.openai_compat.AsyncOpenAI") as mock_openai:
        provider = MinimaxProvider(minimax_config)
        assert provider._api_key == "test_minimax_key"
        assert provider._base_url == MINIMAX_BASE_URL
        mock_openai.assert_called_once()


def test_base_url_constant():
    """MiniMax base URL points to the official API endpoint."""
    assert MINIMAX_BASE_URL == "https://api.minimax.io/v1"


def test_build_request_body_basic(minimax_provider):
    """Basic request builds correctly."""
    req = MockRequest(model="MiniMax-M2.7")
    body = minimax_provider._build_request_body(req)

    assert body["model"] == "MiniMax-M2.7"
    assert body["messages"][0]["role"] == "system"


def test_build_request_body_temperature_zero_clamped(minimax_provider):
    """Temperature of 0 is clamped to 0.01 (MiniMax requires temperature > 0)."""
    req = MockRequest(temperature=0)
    body = minimax_provider._build_request_body(req)

    assert body["temperature"] == 0.01


def test_build_request_body_temperature_nonzero_preserved(minimax_provider):
    """Non-zero temperature is passed through unchanged."""
    req = MockRequest(temperature=0.7)
    body = minimax_provider._build_request_body(req)

    assert body["temperature"] == 0.7


def test_build_request_body_highspeed_model(minimax_provider):
    """MiniMax-M2.7-highspeed model name is forwarded without modification."""
    req = MockRequest(model="MiniMax-M2.7-highspeed")
    body = minimax_provider._build_request_body(req)

    assert body["model"] == "MiniMax-M2.7-highspeed"


@pytest.mark.asyncio
async def test_stream_response_text(minimax_provider):
    """Text content is emitted as text delta SSE events."""
    req = MockRequest()

    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(content="Hello!", reasoning_content=None, tool_calls=None),
            finish_reason="stop",
        )
    ]
    mock_chunk.usage = MagicMock(completion_tokens=5)

    async def mock_stream():
        yield mock_chunk

    with patch.object(
        minimax_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = [event async for event in minimax_provider.stream_response(req)]

        assert any('"text_delta"' in event and "Hello!" in event for event in events)
