"""Tests for OpenRouter provider."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from api.models.anthropic import MessagesRequest, Message
from providers.openrouter import OpenRouterProvider
from providers.base import ProviderConfig
from config.nim import NimSettings


class MockMessage:
    """Minimal message for converter."""

    def __init__(self, role, content):
        self.role = role
        self.content = content


class MockRequest:
    """Mock request with fields needed for build_request_body and stream_response."""

    def __init__(self, **kwargs):
        self.model = "google/gemini-2.0-flash-001"
        self.messages = [MockMessage("user", "Hello")]
        self.max_tokens = 100
        self.temperature = 0.5
        self.top_p = 0.9
        self.system = None
        self.stop_sequences = None
        self.tools = []
        self.tool_choice = None
        for k, v in kwargs.items():
            setattr(self, k, v)


@pytest.fixture
def provider_config():
    return ProviderConfig(
        api_key="test_openrouter_key",
        base_url="https://openrouter.ai/api/v1",
        rate_limit=20,
        rate_window=60,
        nim_settings=NimSettings(),
    )


@pytest.fixture(autouse=True)
def mock_rate_limiter():
    """Mock the global rate limiter to prevent waiting."""
    with patch("providers.openrouter.client.GlobalRateLimiter") as mock:
        instance = mock.get_instance.return_value
        instance.wait_if_blocked = AsyncMock(return_value=False)

        async def _passthrough(fn, *args, **kwargs):
            return await fn(*args, **kwargs)

        instance.execute_with_retry = AsyncMock(side_effect=_passthrough)
        yield instance


@pytest.fixture
def openrouter_provider(provider_config):
    with patch("providers.openrouter.client.AsyncOpenAI"):
        return OpenRouterProvider(provider_config)


@pytest.mark.asyncio
async def test_init(provider_config):
    """Test provider initialization."""
    with patch("providers.openrouter.client.AsyncOpenAI") as mock_openai:
        provider = OpenRouterProvider(provider_config)
        assert provider._api_key == "test_openrouter_key"
        assert provider._base_url == "https://openrouter.ai/api/v1"
        mock_openai.assert_called_once_with(
            api_key="test_openrouter_key",
            base_url="https://openrouter.ai/api/v1",
            max_retries=0,
            timeout=300.0,
        )


@pytest.mark.asyncio
async def test_build_request_body(openrouter_provider):
    """Test request body conversion."""
    req = MessagesRequest(
        model="google/gemini-2.0-flash-001",
        messages=[Message(role="user", content="Hi")],
        max_tokens=100,
    )
    body = openrouter_provider._build_request_body(req)
    assert body["model"] == "google/gemini-2.0-flash-001"
    assert len(body["messages"]) >= 1
    assert body["max_tokens"] == 100


@pytest.mark.asyncio
async def test_build_request_body_no_nim_extras(openrouter_provider):
    """OpenRouter request should not include NIM-specific extra_body."""
    req = MockRequest()
    body = openrouter_provider._build_request_body(req)
    assert "extra_body" not in body
    assert body["model"] == "google/gemini-2.0-flash-001"
    assert body["temperature"] == 0.5


@pytest.mark.asyncio
async def test_stream_response_text(openrouter_provider):
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
        openrouter_provider._client.chat.completions,
        "create",
        new_callable=AsyncMock,
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = []
        async for event in openrouter_provider.stream_response(req):
            events.append(event)

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
async def test_stream_response_thinking_reasoning_content(openrouter_provider):
    """Test streaming with native reasoning_content (some OpenRouter models support this)."""
    req = MockRequest()

    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(content=None, reasoning_content="Thinking..."),
            finish_reason=None,
        )
    ]
    mock_chunk.usage = None

    async def mock_stream():
        yield mock_chunk

    with patch.object(
        openrouter_provider._client.chat.completions,
        "create",
        new_callable=AsyncMock,
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = []
        async for event in openrouter_provider.stream_response(req):
            events.append(event)

        found_thinking = any(
            "event: content_block_delta" in e
            and '"thinking_delta"' in e
            and "Thinking..." in e
            for e in events
        )
        assert found_thinking


@pytest.mark.asyncio
async def test_tool_call_stream(openrouter_provider):
    """Test streaming tool calls."""
    req = MockRequest()

    mock_tc = MagicMock()
    mock_tc.index = 0
    mock_tc.id = "call_1"
    mock_tc.function.name = "search"
    mock_tc.function.arguments = '{"q": "test"}'

    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(
                content=None, reasoning_content=None, tool_calls=[mock_tc]
            ),
            finish_reason=None,
        )
    ]
    mock_chunk.usage = None

    async def mock_stream():
        yield mock_chunk

    with patch.object(
        openrouter_provider._client.chat.completions,
        "create",
        new_callable=AsyncMock,
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = []
        async for event in openrouter_provider.stream_response(req):
            events.append(event)

        starts = [
            e
            for e in events
            if "event: content_block_start" in e and '"tool_use"' in e
        ]
        assert len(starts) == 1
        assert "search" in starts[0]


@pytest.mark.asyncio
async def test_stream_response_api_error_emits_sse_error(openrouter_provider):
    """When API raises during streaming, SSE error event is emitted."""
    req = MockRequest()

    mock_stream = AsyncMock()
    mock_stream.__aiter__ = MagicMock(side_effect=RuntimeError("API failed"))

    with patch.object(
        openrouter_provider._client.chat.completions,
        "create",
        new_callable=AsyncMock,
        return_value=mock_stream,
    ):
        events = []
        async for event in openrouter_provider.stream_response(req):
            events.append(event)

        # Should have message_start, then error content, then message_stop, done
        assert any("event: message_start" in e for e in events)
        assert any("event: content_block_delta" in e and "API failed" in e for e in events)
        assert any("[DONE]" in e for e in events)
