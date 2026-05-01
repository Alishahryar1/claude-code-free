from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from api.models.anthropic import Message, MessagesRequest
from api.services import ClaudeProxyService
from config.settings import Settings
from core.anthropic import iter_provider_stream_error_sse_events
from core.anthropic.sse import SSEBuilder
from providers.base import BaseProvider, ProviderConfig


class FallbackProvider(BaseProvider):
    def __init__(self, provider_id: str, outcomes: dict[str, str]):
        super().__init__(ProviderConfig(api_key="test"))
        self.provider_id = provider_id
        self.outcomes = outcomes
        self.models_seen: list[str] = []

    async def cleanup(self) -> None:
        return None

    async def list_model_ids(self) -> frozenset[str]:
        return frozenset(self.outcomes)

    def preflight_stream(self, *_args, **_kwargs) -> None:
        return None

    async def stream_response(
        self,
        request,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        thinking_enabled: bool | None = None,
    ) -> AsyncIterator[str]:
        self.models_seen.append(request.model)
        outcome = self.outcomes.get(request.model, "ok")
        if outcome == "timeout":
            for event in iter_provider_stream_error_sse_events(
                request=request,
                input_tokens=input_tokens,
                error_message="Provider request timed out after 120s.",
                sent_any_event=False,
                log_raw_sse_events=False,
            ):
                yield event
            return

        sse = SSEBuilder("msg_ok", request.model, input_tokens)
        yield sse.message_start()
        yield sse.content_block_start(0, "text")
        yield sse.content_block_delta(0, "text_delta", f"ok from {request.model}")
        yield sse.content_block_stop(0)
        yield sse.message_delta("end_turn", 1)
        yield sse.message_stop()


async def _collect_response_text(response) -> str:
    parts = [
        chunk.decode() if isinstance(chunk, bytes) else chunk
        async for chunk in response.body_iterator
    ]
    return "".join(parts)


@pytest.fixture
def settings() -> Settings:
    settings = Settings()
    settings.model = "open_router/broken:free"
    settings.model_opus = None
    settings.model_sonnet = None
    settings.model_haiku = None
    settings.enable_model_thinking = True
    settings.enable_opus_thinking = None
    settings.enable_sonnet_thinking = None
    settings.enable_haiku_thinking = None
    return settings


@pytest.mark.asyncio
async def test_initial_openrouter_failure_tries_next_free_model(settings: Settings):
    openrouter = FallbackProvider(
        "open_router",
        {
            "broken:free": "timeout",
            "openai/gpt-oss-20b:free": "ok",
        },
    )
    service = ClaudeProxyService(settings, provider_getter=lambda _: openrouter)

    response = service.create_message(
        MessagesRequest(
            model="claude-3-haiku-20240307",
            max_tokens=100,
            messages=[Message(role="user", content="hello")],
        )
    )

    body = await _collect_response_text(response)

    assert openrouter.models_seen == ["broken:free", "openai/gpt-oss-20b:free"]
    assert "ok from openai/gpt-oss-20b:free" in body
    assert "Provider request timed out" not in body


@pytest.mark.asyncio
async def test_two_free_model_failures_fall_back_to_openrouter_free(
    settings: Settings,
):
    settings.model = "open_router/openai/gpt-oss-20b:free"
    openrouter = FallbackProvider(
        "open_router",
        {
            "openai/gpt-oss-20b:free": "timeout",
            "nvidia/nemotron-nano-9b-v2:free": "timeout",
            "openrouter/free": "ok",
        },
    )
    service = ClaudeProxyService(settings, provider_getter=lambda _: openrouter)

    response = service.create_message(
        MessagesRequest(
            model="claude-3-haiku-20240307",
            max_tokens=100,
            messages=[Message(role="user", content="hello")],
        )
    )

    body = await _collect_response_text(response)

    assert openrouter.models_seen == [
        "openai/gpt-oss-20b:free",
        "nvidia/nemotron-nano-9b-v2:free",
        "openrouter/free",
    ]
    assert "ok from openrouter/free" in body


@pytest.mark.asyncio
async def test_openrouter_router_failure_falls_back_to_deepseek_v4_pro(
    settings: Settings,
):
    settings.model = "open_router/openrouter/free"
    openrouter = FallbackProvider(
        "open_router",
        {
            "openrouter/free": "timeout",
            "openai/gpt-oss-20b:free": "timeout",
            "nvidia/nemotron-nano-9b-v2:free": "timeout",
        },
    )
    deepseek = FallbackProvider("deepseek", {"deepseek-v4-pro": "ok"})

    def get_provider(provider_id: str):
        return {"open_router": openrouter, "deepseek": deepseek}[provider_id]

    service = ClaudeProxyService(settings, provider_getter=get_provider)

    response = service.create_message(
        MessagesRequest(
            model="claude-3-haiku-20240307",
            max_tokens=100,
            messages=[Message(role="user", content="hello")],
        )
    )

    body = await _collect_response_text(response)

    assert deepseek.models_seen == ["deepseek-v4-pro"]
    assert "ok from deepseek-v4-pro" in body
