"""Tests for DeepSeek provider."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from api.models.anthropic import ContentBlockImage, Message, MessagesRequest
from providers.base import ProviderConfig
from providers.deepseek import DEEPSEEK_DEFAULT_BASE, DeepSeekProvider
from providers.exceptions import InvalidRequestError


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
        self.model = "deepseek-chat"
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
def deepseek_config():
    return ProviderConfig(
        api_key="test_deepseek_key",
        base_url=DEEPSEEK_DEFAULT_BASE,
        rate_limit=10,
        rate_window=60,
        enable_thinking=True,
    )


@pytest.fixture(autouse=True)
def mock_rate_limiter():
    """Mock the global rate limiter to prevent waiting."""
    with patch("providers.openai_compat.GlobalRateLimiter") as mock:
        instance = mock.get_scoped_instance.return_value
        instance.wait_if_blocked = AsyncMock(return_value=False)

        async def _passthrough(fn, *args, **kwargs):
            return await fn(*args, **kwargs)

        instance.execute_with_retry = AsyncMock(side_effect=_passthrough)
        yield instance


@pytest.fixture
def deepseek_provider(deepseek_config):
    return DeepSeekProvider(deepseek_config)


def test_init(deepseek_config):
    """Test provider initialization."""
    with patch("providers.openai_compat.AsyncOpenAI") as mock_openai:
        provider = DeepSeekProvider(deepseek_config)
        assert provider._api_key == "test_deepseek_key"
        assert provider._base_url == DEEPSEEK_DEFAULT_BASE
        mock_openai.assert_called_once()


def test_build_request_body_enables_thinking_for_chat_model(deepseek_provider):
    """Thinking-enabled requests add DeepSeek's thinking payload for chat model."""
    req = MockRequest(model="deepseek-chat")
    body = deepseek_provider._build_request_body(req)

    assert body["model"] == "deepseek-chat"
    assert body["extra_body"]["thinking"] == {"type": "enabled"}
    assert body["messages"][0]["role"] == "system"


def test_build_request_body_global_disable_blocks_request_thinking():
    """Global disable suppresses provider-side thinking even if the request enables it."""
    provider = DeepSeekProvider(
        ProviderConfig(
            api_key="test_deepseek_key",
            base_url=DEEPSEEK_DEFAULT_BASE,
            rate_limit=10,
            rate_window=60,
            enable_thinking=False,
        )
    )
    req = MockRequest(model="deepseek-chat")
    body = provider._build_request_body(req)

    assert "extra_body" not in body or "thinking" not in body["extra_body"]


def test_build_request_body_request_disable_blocks_global_thinking(deepseek_provider):
    """Request-level disable suppresses provider-side thinking when global is enabled."""
    req = MockRequest(model="deepseek-chat")
    req.thinking.enabled = False
    body = deepseek_provider._build_request_body(req)

    assert "extra_body" not in body or "thinking" not in body["extra_body"]


def test_build_request_body_reasoner_skips_thinking_extra(deepseek_provider):
    """deepseek-reasoner does not need an extra thinking payload."""
    req = MockRequest(model="deepseek-reasoner")
    body = deepseek_provider._build_request_body(req)

    assert body["model"] == "deepseek-reasoner"
    assert "extra_body" not in body or "thinking" not in body["extra_body"]


def test_build_request_body_preserves_caller_thinking_override(deepseek_provider):
    """Caller-provided thinking payload should not be overwritten."""
    req = MockRequest(
        model="deepseek-chat",
        extra_body={"thinking": {"type": "manual"}},
    )
    body = deepseek_provider._build_request_body(req)

    assert body["extra_body"]["thinking"] == {"type": "manual"}


def test_build_request_body_preserves_reasoning_content(deepseek_provider):
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

    body = deepseek_provider._build_request_body(req)

    assert body["messages"][0]["reasoning_content"] == "First think"


def test_build_request_body_disabled_thinking_omits_reasoning_and_thinking_tags():
    """Resolved thinking policy must strip assistant thinking from OpenAI history."""
    provider = DeepSeekProvider(
        ProviderConfig(
            api_key="test_deepseek_key",
            base_url=DEEPSEEK_DEFAULT_BASE,
            rate_limit=10,
            rate_window=60,
            enable_thinking=False,
        )
    )
    req = MockRequest(
        system=None,
        model="deepseek-chat",
        messages=[
            MockMessage(
                "assistant",
                [
                    MockBlock(type="thinking", thinking="secret"),
                    MockBlock(type="text", text="hi"),
                ],
            )
        ],
    )

    body = provider._build_request_body(req)

    assistant = body["messages"][0]
    assert "reasoning_content" not in assistant
    assert "secret" not in assistant["content"]
    assert assistant["content"] == "hi"


def test_build_request_body_disabled_thinking_omits_redacted_blocks():
    """redacted_thinking is not sent as OpenAI text when thinking is disabled."""
    provider = DeepSeekProvider(
        ProviderConfig(
            api_key="test_deepseek_key",
            base_url=DEEPSEEK_DEFAULT_BASE,
            rate_limit=10,
            rate_window=60,
            enable_thinking=False,
        )
    )
    req = MockRequest(
        system=None,
        model="deepseek-chat",
        messages=[
            MockMessage(
                "assistant",
                [
                    MockBlock(type="redacted_thinking", data="opaque-xyz"),
                    MockBlock(type="text", text="hi"),
                ],
            )
        ],
    )
    body = provider._build_request_body(req)
    assert "opaque-xyz" not in body["messages"][0]["content"]


@pytest.mark.asyncio
async def test_stream_response_reasoning_content(deepseek_provider):
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
        deepseek_provider._client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = mock_stream()

        events = [event async for event in deepseek_provider.stream_response(req)]

        assert any(
            '"thinking_delta"' in event and "Thinking..." in event for event in events
        )


def test_preflight_stream_rejects_unsupported_user_image_for_openai_conversion():
    """Eager preflight: image block fails before a stream would be opened."""
    request = MessagesRequest(
        model="deepseek/deepseek-chat",
        max_tokens=100,
        messages=[
            Message(
                role="user",
                content=[
                    ContentBlockImage(
                        type="image",
                        source={
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "YQ==",
                        },
                    )
                ],
            )
        ],
    )
    provider = DeepSeekProvider(
        ProviderConfig(
            api_key="k",
            base_url=DEEPSEEK_DEFAULT_BASE,
            rate_limit=10,
            rate_window=60,
        )
    )
    with pytest.raises(InvalidRequestError) as exc:
        provider.preflight_stream(request, thinking_enabled=True)
    assert "image" in str(exc.value).lower()
