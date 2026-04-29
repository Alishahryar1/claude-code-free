"""Custom OpenAI provider implementation."""

from collections.abc import Iterator
from typing import Any

from core.anthropic import SSEBuilder
from providers.base import ProviderConfig
from providers.openai_compat import OpenAIChatTransport

from .request import build_request_body


class CustomOpenAIProvider(OpenAIChatTransport):
    """Custom OpenAI-compatible provider for user-specified endpoints."""

    def __init__(self, config: ProviderConfig):
        super().__init__(
            config,
            provider_name="CUSTOM_OPENAI",
            base_url=config.base_url or "",
            api_key=config.api_key,
        )

    def _build_request_body(
        self, request: Any, thinking_enabled: bool | None = None
    ) -> dict:
        """Build request body using standard OpenAI format."""
        return build_request_body(
            request,
            thinking_enabled=self._is_thinking_enabled(request, thinking_enabled),
        )

    def _handle_extra_reasoning(
        self, delta: Any, sse: SSEBuilder, *, thinking_enabled: bool
    ) -> Iterator[str]:
        """No provider-specific reasoning handling for custom OpenAI."""
        return iter(())
