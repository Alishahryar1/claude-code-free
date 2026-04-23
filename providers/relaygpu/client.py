"""RelayGPU provider implementation."""

from typing import Any

from providers.base import ProviderConfig
from providers.openai_compat import OpenAICompatibleProvider

from .request import build_request_body

RELAYGPU_BASE_URL = "https://relay.opengpu.network/v2/openai/v1"


class RelayGpuProvider(OpenAICompatibleProvider):
    """RelayGPU provider using OpenAI-compatible chat completions."""

    def __init__(self, config: ProviderConfig):
        super().__init__(
            config,
            provider_name="RELAYGPU",
            base_url=config.base_url or RELAYGPU_BASE_URL,
            api_key=config.api_key,
        )

    def _build_request_body(self, request: Any) -> dict:
        """Internal helper for tests and shared building."""
        return build_request_body(
            request,
            thinking_enabled=self._is_thinking_enabled(request),
        )
