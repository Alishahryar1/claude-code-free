"""Tests for RelayGPU provider."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from providers.base import ProviderConfig
from providers.relaygpu import RELAYGPU_BASE_URL, RelayGpuProvider


class MockMessage:
    def __init__(self, role, content):
        self.role = role
        self.content = content


class MockBlock:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class MockRequest:
    def __init__(self, **kwargs):
        self.model = "openai/gpt-5.4"
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
        for key, value in kwargs.items():
            setattr(self, key, value)


@pytest.fixture
def relaygpu_config():
    return ProviderConfig(
        api_key="relay_sk_test",
        base_url=RELAYGPU_BASE_URL,
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
def relaygpu_provider(relaygpu_config):
    return RelayGpuProvider(relaygpu_config)


def test_init(relaygpu_config):
    """Test provider initialization uses the fixed base URL and Bearer auth."""
    with patch("providers.openai_compat.AsyncOpenAI") as mock_openai:
        provider = RelayGpuProvider(relaygpu_config)
        assert provider._api_key == "relay_sk_test"
        assert provider._base_url == RELAYGPU_BASE_URL
        mock_openai.assert_called_once()
        call_kwargs = mock_openai.call_args[1]
        assert call_kwargs["api_key"] == "relay_sk_test"
        assert call_kwargs["base_url"] == RELAYGPU_BASE_URL


def test_base_url_constant():
    """RelayGPU base URL points at the OpenAI-compatible endpoint."""
    assert RELAYGPU_BASE_URL == "https://relay.opengpu.network/v2/openai/v1"


def test_build_request_body_passes_model_and_messages(relaygpu_provider):
    """Basic request fields are copied into the OpenAI-format body."""
    req = MockRequest(model="deepseek-ai/DeepSeek-V3.1")
    body = relaygpu_provider._build_request_body(req)

    assert body["model"] == "deepseek-ai/DeepSeek-V3.1"
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][1]["role"] == "user"
    assert body["max_tokens"] == 100
    assert body["temperature"] == 0.5


def test_build_request_body_enables_thinking_via_chat_template_kwargs(
    relaygpu_provider,
):
    """Thinking-enabled requests pass enable_thinking=True through chat_template_kwargs."""
    req = MockRequest(model="Qwen/Qwen3.5-397B-A17B-FP8")
    body = relaygpu_provider._build_request_body(req)

    assert body["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True


def test_build_request_body_global_disable_blocks_request_thinking():
    """Global disable suppresses provider-side thinking even if the request enables it."""
    provider = RelayGpuProvider(
        ProviderConfig(
            api_key="relay_sk_test",
            base_url=RELAYGPU_BASE_URL,
            rate_limit=10,
            rate_window=60,
            enable_thinking=False,
        )
    )
    req = MockRequest(model="Qwen/Qwen3.5-397B-A17B-FP8")
    body = provider._build_request_body(req)

    extra_body = body.get("extra_body", {})
    assert "chat_template_kwargs" not in extra_body


def test_build_request_body_request_disable_blocks_global_thinking(relaygpu_provider):
    """Request-level disable suppresses provider-side thinking when global is enabled."""
    req = MockRequest(model="Qwen/Qwen3.5-397B-A17B-FP8")
    req.thinking.enabled = False
    body = relaygpu_provider._build_request_body(req)

    extra_body = body.get("extra_body", {})
    assert "chat_template_kwargs" not in extra_body


def test_build_request_body_preserves_caller_chat_template_kwargs(relaygpu_provider):
    """Caller-provided chat_template_kwargs keys are preserved, not overwritten."""
    req = MockRequest(
        model="Qwen/Qwen3.5-397B-A17B-FP8",
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    body = relaygpu_provider._build_request_body(req)

    assert body["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False


def test_build_request_body_preserves_reasoning_content(relaygpu_provider):
    """Thinking blocks are mirrored into reasoning_content for continuation."""
    req = MockRequest(
        system=None,
        messages=[
            MockMessage(
                "assistant",
                [
                    MockBlock(type="thinking", thinking="First think"),
                    MockBlock(type="text", text="Then answer"),
                ],
            )
        ],
    )

    body = relaygpu_provider._build_request_body(req)

    assert body["messages"][0]["reasoning_content"] == "First think"


@pytest.mark.asyncio
async def test_stream_response_reasoning_content(relaygpu_provider):
    """reasoning_content deltas are emitted as thinking blocks."""
    req = MockRequest()

    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(
            delta=MagicMock(content=None, reasoning_content="Thinking..."),
            finish_reason="stop",
        )
    ]
    mock_chunk.usage = MagicMock(completion_tokens=2)

    async def mock_stream():
        yield mock_chunk

    with patch.object(
        relaygpu_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = [event async for event in relaygpu_provider.stream_response(req)]

        assert any(
            '"thinking_delta"' in event and "Thinking..." in event for event in events
        )
