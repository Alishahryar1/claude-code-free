"""Tests for DeepSeek native Anthropic Messages provider."""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from api.models.anthropic import (
    ContentBlockImage,
    Message,
    MessagesRequest,
    Tool,
)
from config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from providers.base import ProviderConfig
from providers.deepseek import (
    DEEPSEEK_ANTHROPIC_DEFAULT_BASE,
    DEEPSEEK_DEFAULT_BASE,
    DeepSeekProvider,
)
from providers.exceptions import InvalidRequestError


@pytest.fixture
def deepseek_config():
    return ProviderConfig(
        api_key="test_deepseek_key",
        base_url=DEEPSEEK_ANTHROPIC_DEFAULT_BASE,
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
def deepseek_provider(deepseek_config):
    return DeepSeekProvider(deepseek_config)


def test_default_base_url_alias():
    assert DEEPSEEK_DEFAULT_BASE == DEEPSEEK_ANTHROPIC_DEFAULT_BASE
    assert DEEPSEEK_ANTHROPIC_DEFAULT_BASE == "https://api.deepseek.com/anthropic"


def test_init(deepseek_config):
    with patch("httpx.AsyncClient") as mock_client:
        provider = DeepSeekProvider(deepseek_config)
    assert provider._api_key == "test_deepseek_key"
    assert provider._base_url == "https://api.deepseek.com/anthropic"
    assert mock_client.called


def test_request_headers_includes_x_api_key(deepseek_provider):
    h = deepseek_provider._request_headers()
    assert h["x-api-key"] == "test_deepseek_key"
    assert h["Content-Type"] == "application/json"
    assert h["Accept"] == "text/event-stream"


def test_build_request_body_native_shape(deepseek_provider):
    request = MessagesRequest(
        model="deepseek-v4-pro",
        max_tokens=100,
        messages=[Message(role="user", content="Hello")],
        system="S",
    )
    body = deepseek_provider._build_request_body(request)
    assert body["model"] == "deepseek-v4-pro"
    assert body["stream"] is True
    assert body["messages"][0]["role"] == "user"
    assert body["messages"][0]["content"] in (
        "Hello",
        [{"type": "text", "text": "Hello"}],
    )
    assert body["system"] == "S"
    assert body["max_tokens"] == 100


def test_build_request_body_default_max_tokens(deepseek_provider):
    request = MessagesRequest(
        model="m",
        messages=[Message(role="user", content="x")],
    )
    body = deepseek_provider._build_request_body(request)
    assert body["max_tokens"] == ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS


def test_build_request_body_thinking_enabled(deepseek_provider):
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "x"}],
            "thinking": {"type": "enabled", "budget_tokens": 2000},
        }
    )
    body = deepseek_provider._build_request_body(request)
    assert body["thinking"] == {"type": "enabled", "budget_tokens": 2000}
    assert "extra_body" not in body


def test_build_request_body_tool_list_keeps_thinking(deepseek_provider):
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "x"}],
            "tools": [
                {
                    "name": "Read",
                    "description": "Read a file",
                    "input_schema": {"type": "object", "properties": {}},
                }
            ],
            "thinking": {"type": "enabled", "budget_tokens": 2000},
        }
    )

    body = deepseek_provider._build_request_body(request)

    assert body["thinking"] == {"type": "enabled", "budget_tokens": 2000}
    assert body["tools"][0]["name"] == "Read"


def test_build_request_body_tool_choice_keeps_thinking(deepseek_provider):
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "x"}],
            "tool_choice": {"type": "auto"},
            "thinking": {"type": "enabled", "budget_tokens": 2000},
        }
    )

    body = deepseek_provider._build_request_body(request)

    assert body["thinking"] == {"type": "enabled", "budget_tokens": 2000}
    assert body["tool_choice"] == {"type": "auto"}


def test_build_request_body_respects_global_thinking_disable():
    provider = DeepSeekProvider(
        ProviderConfig(
            api_key="k",
            base_url=DEEPSEEK_ANTHROPIC_DEFAULT_BASE,
            rate_limit=1,
            rate_window=1,
            enable_thinking=False,
        )
    )
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [{"role": "user", "content": "x"}],
            "thinking": {"type": "enabled", "budget_tokens": 1},
        }
    )
    body = provider._build_request_body(request)
    assert "thinking" not in body


def test_preserve_unsigned_thinking_when_thinking_on(deepseek_provider):
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": "plain",
                            "signature": None,
                        },
                        {"type": "text", "text": "out"},
                    ],
                }
            ],
        }
    )
    body = deepseek_provider._build_request_body(request)
    blocks = body["messages"][0]["content"]
    assert len(blocks) == 2
    assert blocks[0]["type"] == "thinking"
    assert blocks[0]["thinking"] == "plain"


def test_strip_redacted_thinking_when_thinking_on(deepseek_provider):
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "redacted_thinking", "data": "opaque"},
                        {"type": "text", "text": "out"},
                    ],
                }
            ],
        }
    )
    body = deepseek_provider._build_request_body(request)
    types = {b["type"] for b in body["messages"][0]["content"]}
    assert "redacted_thinking" not in types
    assert "text" in types


def test_tool_history_with_replayable_thinking_preserves_thinking(deepseek_provider):
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": "hidden",
                            "signature": "sig_123",
                        },
                        {"type": "redacted_thinking", "data": "opaque"},
                        {
                            "type": "tool_use",
                            "id": "t1",
                            "name": "Read",
                            "input": {"file_path": "x"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": "ok",
                        }
                    ],
                },
            ],
            "thinking": {"type": "enabled", "budget_tokens": 2000},
            "context_management": {
                "edits": [{"type": "clear_thinking_20251015", "keep": "all"}]
            },
            "output_config": {"effort": "high"},
        }
    )

    body = deepseek_provider._build_request_body(request)

    assert body["thinking"] == {"type": "enabled", "budget_tokens": 2000}
    assert body["context_management"] == {
        "edits": [{"type": "clear_thinking_20251015", "keep": "all"}]
    }
    assert body["output_config"] == {"effort": "high"}
    assistant_blocks = body["messages"][0]["content"]
    assert [block["type"] for block in assistant_blocks] == ["thinking", "tool_use"]
    assert assistant_blocks[0]["thinking"] == "hidden"
    assert assistant_blocks[0]["signature"] == "sig_123"
    assert assistant_blocks[1]["name"] == "Read"
    assert body["messages"][1]["content"][0]["type"] == "tool_result"


def test_tool_history_with_unsigned_thinking_preserves_thinking(deepseek_provider):
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "plain"},
                        {
                            "type": "tool_use",
                            "id": "t1",
                            "name": "Read",
                            "input": {"file_path": "x"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": "ok",
                        }
                    ],
                },
            ],
            "thinking": {"type": "enabled"},
        }
    )

    body = deepseek_provider._build_request_body(request)

    assert body["thinking"] == {"type": "enabled"}
    assert body["messages"][0]["content"][0] == {
        "type": "thinking",
        "thinking": "plain",
    }


def test_tool_history_without_thinking_disables_thinking_and_hints(deepseek_provider):
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "t1",
                            "name": "Read",
                            "input": {"file_path": "x"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": "ok",
                        }
                    ],
                },
            ],
            "tools": [
                {
                    "name": "Read",
                    "description": "Read a file",
                    "input_schema": {"type": "object", "properties": {}},
                }
            ],
            "tool_choice": {"type": "auto"},
            "thinking": {"type": "enabled", "budget_tokens": 2000},
            "context_management": {
                "edits": [
                    {"type": "clear_thinking_20251015", "keep": "all"},
                    {"type": "other_edit", "keep": "all"},
                ],
                "other": True,
            },
            "output_config": {"effort": "high", "format": "text"},
        }
    )

    body = deepseek_provider._build_request_body(request)

    assert "thinking" not in body
    assert body["context_management"] == {
        "edits": [{"type": "other_edit", "keep": "all"}],
        "other": True,
    }
    assert body["output_config"] == {"format": "text"}
    assert body["tools"][0]["name"] == "Read"
    assert body["tool_choice"] == {"type": "auto"}
    assert body["messages"][0]["content"][0]["type"] == "tool_use"
    assert body["messages"][1]["content"][0]["type"] == "tool_result"


def test_tool_history_with_empty_thinking_disables_thinking(deepseek_provider):
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": ""},
                        {
                            "type": "tool_use",
                            "id": "t1",
                            "name": "Read",
                            "input": {"file_path": "x"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": "ok",
                        }
                    ],
                },
            ],
            "thinking": {"type": "enabled"},
        }
    )

    body = deepseek_provider._build_request_body(request)

    assert "thinking" not in body
    assert [block["type"] for block in body["messages"][0]["content"]] == ["tool_use"]


def test_thinking_off_strips_thinking_history():
    provider = DeepSeekProvider(
        ProviderConfig(
            api_key="k",
            base_url=DEEPSEEK_ANTHROPIC_DEFAULT_BASE,
            rate_limit=1,
            rate_window=1,
            enable_thinking=False,
        )
    )
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "sec"},
                        {"type": "text", "text": "hi"},
                    ],
                }
            ],
        }
    )
    body = provider._build_request_body(request)
    for b in body["messages"][0]["content"]:
        assert b["type"] != "thinking"
    assert "sec" not in str(body["messages"])


def test_passthrough_tool_use_and_result(deepseek_provider):
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "t1",
                            "name": "n",
                            "input": {"a": 1},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": "ok",
                        }
                    ],
                },
            ],
        }
    )
    body = deepseek_provider._build_request_body(request)
    assert body["messages"][0]["content"][0]["type"] == "tool_use"
    assert body["messages"][1]["content"][0]["type"] == "tool_result"


def test_preflight_rejects_user_image():
    request = MessagesRequest(
        model="m",
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
            base_url=DEEPSEEK_ANTHROPIC_DEFAULT_BASE,
            rate_limit=1,
            rate_window=1,
        )
    )
    with pytest.raises(InvalidRequestError, match="image"):
        provider.preflight_stream(request, thinking_enabled=True)


def test_preflight_rejects_mcp_servers():
    request = MessagesRequest(
        model="m",
        messages=[Message(role="user", content="x")],
        mcp_servers=[{"type": "url", "url": "https://x"}],
    )
    provider = DeepSeekProvider(
        ProviderConfig(
            api_key="k",
            base_url=DEEPSEEK_ANTHROPIC_DEFAULT_BASE,
            rate_limit=1,
            rate_window=1,
        )
    )
    with pytest.raises(InvalidRequestError, match="mcp_servers"):
        provider.preflight_stream(request)


def test_preflight_rejects_listed_server_tools_in_tools_list():
    request = MessagesRequest(
        model="m",
        messages=[Message(role="user", content="x")],
        tools=[Tool(name="web_search", type="web_search_20250305", input_schema={})],
    )
    provider = DeepSeekProvider(
        ProviderConfig(
            api_key="k",
            base_url=DEEPSEEK_ANTHROPIC_DEFAULT_BASE,
            rate_limit=1,
            rate_window=1,
        )
    )
    with pytest.raises(InvalidRequestError, match="web_search"):
        provider.preflight_stream(request)


def test_preflight_rejects_server_tool_result_blocks():
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "server_tool_use",
                            "id": "s1",
                            "name": "web_search",
                            "input": {"q": "a"},
                        },
                        {
                            "type": "web_search_tool_result",
                            "tool_use_id": "s1",
                            "content": [],
                        },
                    ],
                }
            ],
        }
    )
    provider = DeepSeekProvider(
        ProviderConfig(
            api_key="k",
            base_url=DEEPSEEK_ANTHROPIC_DEFAULT_BASE,
            rate_limit=1,
            rate_window=1,
        )
    )
    with pytest.raises(InvalidRequestError, match=r"web_search_tool_result|server"):
        provider.preflight_stream(request)


def test_strip_reasoning_content_not_forwarded(deepseek_provider):
    # ``reasoning_content`` is for OpenAI helpers only, not in native body.
    request = MessagesRequest(
        model="m",
        messages=[
            Message(
                role="assistant",
                content="hi",
                reasoning_content="r",
            )
        ],
    )
    body = deepseek_provider._build_request_body(request)
    assert "reasoning_content" not in body["messages"][0]


@pytest.mark.asyncio
async def test_stream_uses_post_messages_path(deepseek_provider):
    request = MessagesRequest(
        model="m",
        messages=[Message(role="user", content="hi")],
    )
    called: dict[str, str] = {}

    async def fake_send(request, *args, **kwargs):
        called["path"] = request.url.path
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.is_closed = False
        mock_resp.raise_for_status = lambda: None

        async def aiter():
            if False:  # pragma: no cover
                yield ""

        mock_resp.aiter_lines = aiter
        mock_resp.aclose = AsyncMock()
        return mock_resp

    deepseek_provider._client.send = fake_send
    _ = [x async for x in deepseek_provider.stream_response(request, request_id="r1")]

    assert called["path"] == "/anthropic/messages"


def test_drops_extra_body_from_canonical_request(deepseek_provider):
    raw = {
        "model": "m",
        "max_tokens": 3,
        "messages": [{"role": "user", "content": "x"}],
        "extra_body": {"note": 1},
    }
    r = MessagesRequest.model_validate(raw)
    body = deepseek_provider._build_request_body(r)
    assert "extra_body" not in body
