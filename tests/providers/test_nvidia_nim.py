import json
from unittest.mock import AsyncMock, MagicMock, patch

import openai
import pytest
from httpx import Request, Response

from providers.nvidia_nim import NvidiaNimProvider


# Mock data classes
class MockMessage:
    def __init__(self, role, content):
        self.role = role
        self.content = content


class MockTool:
    def __init__(self, name, description, input_schema):
        self.name = name
        self.description = description
        self.input_schema = input_schema


class MockRequest:
    def __init__(self, **kwargs):
        self.model = "test-model"
        self.messages = [MockMessage("user", "Hello")]
        self.max_tokens = 100
        self.temperature = 0.5
        self.top_p = 0.9
        self.system = "System prompt"
        self.stop_sequences = ["STOP"]
        self.tools = []
        self.extra_body = {}
        self.thinking = MagicMock()
        self.thinking.enabled = True
        for k, v in kwargs.items():
            setattr(self, k, v)


@pytest.fixture(autouse=True)
def mock_rate_limiter():
    """Mock the global rate limiter to prevent waiting."""
    with patch("providers.openai_compat.GlobalRateLimiter") as mock:
        instance = mock.get_instance.return_value
        instance.wait_if_blocked = AsyncMock(return_value=False)

        # execute_with_retry should call through to the actual function
        async def _passthrough(fn, *args, **kwargs):
            return await fn(*args, **kwargs)

        instance.execute_with_retry = AsyncMock(side_effect=_passthrough)
        yield instance


@pytest.mark.asyncio
async def test_init(provider_config):
    """Test provider initialization."""
    with patch("providers.openai_compat.AsyncOpenAI") as mock_openai:
        from config.nim import NimSettings

        provider = NvidiaNimProvider(provider_config, nim_settings=NimSettings())
        assert provider._api_key == "test_key"
        assert provider._base_url == "https://test.api.nvidia.com/v1"
        mock_openai.assert_called_once()


@pytest.mark.asyncio
async def test_init_uses_configurable_timeouts():
    """Test that provider passes configurable read/write/connect timeouts to client."""
    from config.nim import NimSettings
    from providers.base import ProviderConfig

    config = ProviderConfig(
        api_key="test_key",
        base_url="https://test.api.nvidia.com/v1",
        http_read_timeout=600.0,
        http_write_timeout=15.0,
        http_connect_timeout=5.0,
    )
    with patch("providers.openai_compat.AsyncOpenAI") as mock_openai:
        NvidiaNimProvider(config, nim_settings=NimSettings())
        call_kwargs = mock_openai.call_args[1]
        timeout = call_kwargs["timeout"]
        assert timeout.read == 600.0
        assert timeout.write == 15.0
        assert timeout.connect == 5.0


@pytest.mark.asyncio
async def test_build_request_body(provider_config):
    """Test request body construction."""
    from config.nim import NimSettings

    provider = NvidiaNimProvider(provider_config, nim_settings=NimSettings())
    req = MockRequest()
    body = provider._build_request_body(req)

    assert body["model"] == "test-model"
    assert body["temperature"] == 0.5
    assert len(body["messages"]) == 2  # System + User
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][0]["content"] == "System prompt"

    assert "extra_body" in body
    ctk = body["extra_body"]["chat_template_kwargs"]
    assert ctk["thinking"] is True
    assert ctk["enable_thinking"] is True
    assert ctk["reasoning_budget"] == body["max_tokens"]
    assert "reasoning_budget" not in body["extra_body"]


@pytest.mark.asyncio
async def test_build_request_body_omits_reasoning_when_globally_disabled(
    provider_config,
):
    from config.nim import NimSettings

    provider = NvidiaNimProvider(
        provider_config.model_copy(update={"enable_thinking": False}),
        nim_settings=NimSettings(),
    )
    req = MockRequest()
    body = provider._build_request_body(req)

    extra = body.get("extra_body", {})
    assert "chat_template_kwargs" not in extra
    assert "reasoning_budget" not in extra


@pytest.mark.asyncio
async def test_build_request_body_omits_reasoning_when_request_disables_thinking(
    provider_config,
):
    from config.nim import NimSettings

    provider = NvidiaNimProvider(provider_config, nim_settings=NimSettings())
    req = MockRequest()
    req.thinking.enabled = False
    body = provider._build_request_body(req)

    extra = body.get("extra_body", {})
    assert "chat_template_kwargs" not in extra
    assert "reasoning_budget" not in extra


@pytest.mark.asyncio
async def test_stream_response_text(nim_provider):
    """Test streaming text response."""
    req = MockRequest()

    # Create mock chunks
    mock_chunk1 = MagicMock()
    mock_chunk1.choices = [
        MagicMock(
            delta=MagicMock(content="Hello", reasoning_content=""), finish_reason=None
        )
    ]
    mock_chunk1.usage = None

    mock_chunk2 = MagicMock()
    mock_chunk2.choices = [
        MagicMock(
            delta=MagicMock(content=" World", reasoning_content=""),
            finish_reason="stop",
        )
    ]
    mock_chunk2.usage = MagicMock(completion_tokens=10)

    async def mock_stream():
        yield mock_chunk1
        yield mock_chunk2

    with patch.object(
        nim_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = [e async for e in nim_provider.stream_response(req)]

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
async def test_stream_response_thinking_reasoning_content(nim_provider):
    """Test streaming with native reasoning_content."""
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
        nim_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = [e async for e in nim_provider.stream_response(req)]

        # Check for thinking_delta
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
async def test_stream_response_suppresses_thinking_when_disabled(provider_config):
    from config.nim import NimSettings

    provider = NvidiaNimProvider(
        provider_config.model_copy(update={"enable_thinking": False}),
        nim_settings=NimSettings(),
    )
    req = MockRequest()

    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(
                content="<think>secret</think>Answer", reasoning_content="Thinking..."
            ),
            finish_reason="stop",
        )
    ]
    mock_chunk.usage = None

    async def mock_stream():
        yield mock_chunk

    with patch.object(
        provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = [e async for e in provider.stream_response(req)]

    event_text = "".join(events)
    assert "thinking_delta" not in event_text
    assert "Thinking..." not in event_text
    assert "secret" not in event_text
    assert "Answer" in event_text


def _make_bad_request_error(message: str) -> openai.BadRequestError:
    response = Response(status_code=400, request=Request("POST", "http://test"))
    body = {"error": {"message": message}}
    return openai.BadRequestError(message, response=response, body=body)


@pytest.mark.asyncio
async def test_tool_call_stream(nim_provider):
    """Test streaming tool calls."""
    req = MockRequest()

    # Mock tool call delta
    mock_tc = MagicMock()
    mock_tc.index = 0
    mock_tc.id = "call_1"
    mock_tc.function.name = "search"
    mock_tc.function.arguments = '{"q": "test"}'

    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(content=None, reasoning_content="", tool_calls=[mock_tc]),
            finish_reason=None,
        )
    ]
    mock_chunk.usage = None

    async def mock_stream():
        yield mock_chunk

    with patch.object(
        nim_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = [e async for e in nim_provider.stream_response(req)]

        starts = [
            e for e in events if "event: content_block_start" in e and '"tool_use"' in e
        ]
        assert len(starts) == 1
        assert "search" in starts[0]


@pytest.mark.asyncio
async def test_stream_response_retries_without_reasoning_budget(nim_provider):
    req = MockRequest()

    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(content="Recovered", reasoning_content=""),
            finish_reason="stop",
        )
    ]
    mock_chunk.usage = MagicMock(completion_tokens=5)

    async def mock_stream():
        yield mock_chunk

    error = _make_bad_request_error("Unsupported field: reasoning_budget")

    with patch.object(
        nim_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.side_effect = [error, mock_stream()]

        events = [e async for e in nim_provider.stream_response(req)]

    assert mock_create.await_count == 2
    first_call = mock_create.await_args_list[0].kwargs
    second_call = mock_create.await_args_list[1].kwargs
    assert (
        first_call["extra_body"]["chat_template_kwargs"]["reasoning_budget"]
        == first_call["max_tokens"]
    )
    assert "reasoning_budget" not in second_call["extra_body"]
    assert "reasoning_budget" not in second_call["extra_body"]["chat_template_kwargs"]
    assert second_call["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True
    assert any("Recovered" in event for event in events)
    assert any("message_stop" in event for event in events)


@pytest.mark.asyncio
async def test_stream_response_bad_request_without_reasoning_budget_does_not_retry(
    nim_provider,
):
    req = MockRequest()
    error = _make_bad_request_error("Unsupported field: top_k")

    with patch.object(
        nim_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.side_effect = error

        events = [e async for e in nim_provider.stream_response(req)]

    assert mock_create.await_count == 1
    assert any("Unsupported field: top_k" in event for event in events)
    assert any("message_stop" in event for event in events)
