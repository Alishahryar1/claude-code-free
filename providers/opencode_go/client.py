"""OpenCode Go provider implementation."""

from collections.abc import AsyncIterator
from typing import Any

from providers.anthropic_messages import AnthropicMessagesTransport
from providers.base import BaseProvider, ProviderConfig
from providers.defaults import OPENCODE_GO_DEFAULT_BASE
from providers.openai_compat import OpenAIChatTransport


class OpenCodeGoProvider(BaseProvider):
    """OpenCode Go provider dynamic transport selector."""

    def __init__(self, config: ProviderConfig):
        super().__init__(config)

        # Anthropic endpoint handles the 'minimax' models (which use @ai-sdk/anthropic).
        self._anthropic_client = AnthropicMessagesTransport(
            config,
            provider_name="OPENCODE_GO",
            default_base_url=config.base_url or OPENCODE_GO_DEFAULT_BASE,
        )

        # OpenAI Chat Completions endpoint handles 'glm', 'qwen', 'kimi', 'mimo', 'deepseek'.
        self._openai_client = OpenAIChatTransport(
            config,
            provider_name="OPENCODE_GO",
            base_url=config.base_url or OPENCODE_GO_DEFAULT_BASE,
            api_key=config.api_key,
        )

    def _is_anthropic(self, request: Any) -> bool:
        """Return True if this model requires the Anthropic endpoint."""
        model = getattr(request, "model", "")
        return model.startswith("minimax")

    def preflight_stream(
        self, request: Any, *, thinking_enabled: bool | None = None
    ) -> None:
        """Validate stream payload prior to dispatch."""
        if self._is_anthropic(request):
            self._anthropic_client.preflight_stream(
                request, thinking_enabled=thinking_enabled
            )
        else:
            self._openai_client.preflight_stream(
                request, thinking_enabled=thinking_enabled
            )

    async def cleanup(self) -> None:
        """Cleanup sessions for both nested transports."""
        await self._openai_client.cleanup()
        await self._anthropic_client.cleanup()

    async def stream_response(
        self,
        request: Any,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        thinking_enabled: bool | None = None,
    ) -> AsyncIterator[str]:
        """Stream provider response with automatic endpoint translation."""
        if self._is_anthropic(request):
            async for chunk in self._anthropic_client.stream_response(
                request,
                input_tokens,
                request_id=request_id,
                thinking_enabled=thinking_enabled,
            ):
                yield chunk
        else:
            async for chunk in self._openai_client.stream_response(
                request,
                input_tokens,
                request_id=request_id,
                thinking_enabled=thinking_enabled,
            ):
                yield chunk
