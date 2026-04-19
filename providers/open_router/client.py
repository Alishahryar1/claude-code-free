"""OpenRouter provider implementation."""

from collections.abc import Iterator
from typing import Any

from providers.base import ProviderConfig
from providers.common import SSEBuilder
from providers.openai_compat import OpenAICompatibleProvider

from .request import build_request_body

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterProvider(OpenAICompatibleProvider):
    """OpenRouter provider using OpenAI-compatible API."""

    def __init__(self, config: ProviderConfig):
        super().__init__(
            config,
            provider_name="OPENROUTER",
            base_url=config.base_url or OPENROUTER_BASE_URL,
            api_key=config.api_key,
        )

    def _build_request_body(self, request: Any) -> dict:
        """Internal helper for tests and shared building."""
        return build_request_body(
            request,
            thinking_enabled=self._is_thinking_enabled(request),
        )

    def _handle_extra_reasoning(
        self, delta: Any, sse: SSEBuilder, *, thinking_enabled: bool
    ) -> Iterator[str]:
        """Handle reasoning_details for StepFun models."""
        if not thinking_enabled:
            return
        reasoning_details = getattr(delta, "reasoning_details", None)
        if reasoning_details and isinstance(reasoning_details, list):
            for item in reasoning_details:
                text = item.get("text", "") if isinstance(item, dict) else ""
                if text:
                    yield from sse.ensure_thinking_block()
                    yield sse.emit_thinking_delta(text)
