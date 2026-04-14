"""Tests for Modal Research provider."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from providers.base import ProviderConfig
from providers.modal import MODAL_BASE_URL, ModalProvider
from providers.modal.request import MODAL_DEFAULT_MAX_TOKENS


class MockMessage:
    def __init__(self, role, content):
        self.role = role
        self.content = content


class MockRequest:
    def __init__(self, **kwargs):
        self.model = "test-model"
        self.messages = [MockMessage("user", "Hello")]
        self.max_tokens = 100
        self.temperature = 0.5
        self.top_p = 0.9
        self.system = "System prompt"
        self.stop_sequences = None
        self.tools = []
        self.extra_body = {}
        self.thinking = MagicMock()
        self.thinking.enabled = False
        for k, v in kwargs.items():
            setattr(self, k, v)


@pytest.fixture
def modal_config():
    return ProviderConfig(
        api_key="test_modal_key",
        base_url="https://api.us-west-2.modal.direct/v1",
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
def modal_provider(modal_config):
    return ModalProvider(modal_config)


def test_init(modal_config):
    """Test provider initialization."""
    with patch("providers.openai_compat.AsyncOpenAI") as mock_openai:
        provider = ModalProvider(modal_config)
        assert provider._api_key == "test_modal_key"
        assert provider._base_url == "https://api.us-west-2.modal.direct/v1"
        mock_openai.assert_called_once()


def test_init_default_base_url():
    """Test that provider uses MODAL_BASE_URL when no base_url is configured."""
    config = ProviderConfig(api_key="test_modal_key")
    with patch("providers.openai_compat.AsyncOpenAI"):
        provider = ModalProvider(config)
        assert provider._base_url == MODAL_BASE_URL


def test_init_uses_configurable_timeouts():
    """Test that provider passes configurable read/write/connect timeouts to client."""
    config = ProviderConfig(
        api_key="test_modal_key",
        base_url="https://api.us-west-2.modal.direct/v1",
        http_read_timeout=600.0,
        http_write_timeout=15.0,
        http_connect_timeout=5.0,
    )
    with patch("providers.openai_compat.AsyncOpenAI") as mock_openai:
        ModalProvider(config)
        call_kwargs = mock_openai.call_args[1]
        timeout = call_kwargs["timeout"]
        assert timeout.read == 600.0
        assert timeout.write == 15.0
        assert timeout.connect == 5.0


def test_build_request_body(modal_provider):
    """Test request body construction."""
    req = MockRequest()
    body = modal_provider._build_request_body(req)

    assert body["model"] == "test-model"
    assert body["temperature"] == 0.5
    assert body["max_tokens"] == 100
    assert len(body["messages"]) == 2  # System + User
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][0]["content"] == "System prompt"
    assert body["messages"][1]["role"] == "user"
    assert body["messages"][1]["content"] == "Hello"


def test_build_request_body_default_max_tokens(modal_provider):
    """max_tokens=None uses MODAL_DEFAULT_MAX_TOKENS (81920)."""
    req = MockRequest(max_tokens=None)
    body = modal_provider._build_request_body(req)
    assert body["max_tokens"] == MODAL_DEFAULT_MAX_TOKENS
    assert body["max_tokens"] == 81920


def test_build_request_body_with_tools(modal_provider):
    """Test request body includes converted tools."""

    class MockTool:
        def __init__(self, name, description, input_schema):
            self.name = name
            self.description = description
            self.input_schema = input_schema

    tools = [
        MockTool(
            name="search",
            description="Search the web",
            input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
        )
    ]
    req = MockRequest(tools=tools)
    body = modal_provider._build_request_body(req)

    assert "tools" in body
    assert len(body["tools"]) == 1
    assert body["tools"][0]["function"]["name"] == "search"


def test_build_request_body_stop_sequences(modal_provider):
    """Test stop sequences are included when provided."""
    req = MockRequest(stop_sequences=["STOP", "END"])
    body = modal_provider._build_request_body(req)
    assert body["stop"] == ["STOP", "END"]


@pytest.mark.asyncio
async def test_stream_response_text(modal_provider):
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
        modal_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = [e async for e in modal_provider.stream_response(req)]

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
async def test_stream_response_reasoning_content(modal_provider):
    """Test streaming with reasoning_content delta."""
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
        modal_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = [e async for e in modal_provider.stream_response(req)]

        found_thinking = False
        for e in events:
            if (
                "event: content_block_delta" in e
                and '"thinking_delta"' in e
                and "Thinking..." in e
            ):
                found_thinking = True
        assert found_thinking


@pytest.mark.asyncio
async def test_stream_response_empty_choices_skipped(modal_provider):
    """Chunks with empty choices are skipped."""
    req = MockRequest()

    async def mock_stream():
        yield MagicMock(choices=[], usage=None)
        yield MagicMock(
            choices=[
                MagicMock(
                    delta=MagicMock(content="ok", reasoning_content=None),
                    finish_reason="stop",
                )
            ],
            usage=MagicMock(completion_tokens=2),
        )

    with patch.object(
        modal_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()
        events = [e async for e in modal_provider.stream_response(req)]
        assert any("content_block_delta" in e and "ok" in e for e in events)


@pytest.mark.asyncio
async def test_stream_response_delta_none_skipped(modal_provider):
    """Chunks with delta=None are skipped."""
    req = MockRequest()

    async def mock_stream():
        yield MagicMock(
            choices=[MagicMock(delta=None, finish_reason=None)],
            usage=None,
        )
        yield MagicMock(
            choices=[
                MagicMock(
                    delta=MagicMock(content="x", reasoning_content=None),
                    finish_reason="stop",
                )
            ],
            usage=MagicMock(completion_tokens=1),
        )

    with patch.object(
        modal_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()
        events = [e async for e in modal_provider.stream_response(req)]
        assert any("x" in e for e in events)


@pytest.mark.asyncio
async def test_tool_call_stream(modal_provider):
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
            delta=MagicMock(content=None, reasoning_content=None, tool_calls=[mock_tc]),
            finish_reason=None,
        )
    ]
    mock_chunk.usage = None

    async def mock_stream():
        yield mock_chunk

    with patch.object(
        modal_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = [e async for e in modal_provider.stream_response(req)]

        starts = [
            e for e in events if "event: content_block_start" in e and '"tool_use"' in e
        ]
        assert len(starts) == 1
        assert "search" in starts[0]


@pytest.mark.asyncio
async def test_stream_response_error_path(modal_provider):
    """Stream raises exception -> error event emitted."""
    req = MockRequest()

    async def mock_stream():
        raise RuntimeError("API failed")
        yield  # unreachable, makes it a generator

    with patch.object(
        modal_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()
        events = [e async for e in modal_provider.stream_response(req)]
        assert any("API failed" in e for e in events)
        assert any("message_stop" in e for e in events)


@pytest.mark.asyncio
async def test_stream_response_finish_reason_only(modal_provider):
    """Chunk with finish_reason but no content still completes."""
    req = MockRequest()

    async def mock_stream():
        yield MagicMock(
            choices=[
                MagicMock(
                    delta=MagicMock(content=None, reasoning_content=None),
                    finish_reason="stop",
                )
            ],
            usage=MagicMock(completion_tokens=0),
        )

    with patch.object(
        modal_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()
        events = [e async for e in modal_provider.stream_response(req)]
        assert any("message_delta" in e for e in events)
        assert any("message_stop" in e for e in events)
