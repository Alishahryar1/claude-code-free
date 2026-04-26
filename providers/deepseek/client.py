"""DeepSeek provider implementation."""

from typing import Any

from providers.base import ProviderConfig
from providers.defaults import DEEPSEEK_DEFAULT_BASE
from providers.openai_compat import OpenAIChatTransport

from .request import build_request_body


class DeepSeekProvider(OpenAIChatTransport):
    """DeepSeek provider using OpenAI-compatible chat completions."""

    def __init__(self, config: ProviderConfig):
        super().__init__(
            config,
            provider_name="DEEPSEEK",
            base_url=config.base_url or DEEPSEEK_DEFAULT_BASE,
            api_key=config.api_key,
        )

    def _build_request_body(
        self, request: Any, thinking_enabled: bool | None = None
    ) -> dict:
        """Internal helper for tests and shared building."""
        return build_request_body(
            request,
            thinking_enabled=self._is_thinking_enabled(request, thinking_enabled),
        )
