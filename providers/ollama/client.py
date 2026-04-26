"""Ollama provider implementation."""

import httpx

from providers.anthropic_messages import AnthropicMessagesTransport
from providers.base import ProviderConfig
from providers.defaults import OLLAMA_DEFAULT_BASE

OLLAMA_BASE_URL = OLLAMA_DEFAULT_BASE


class OllamaProvider(AnthropicMessagesTransport):
    """Ollama provider using native Anthropic Messages API."""

    def __init__(self, config: ProviderConfig):
        super().__init__(
            config,
            provider_name="OLLAMA",
            default_base_url=OLLAMA_BASE_URL,
        )
        self._api_key = config.api_key or "ollama"

    async def _send_stream_request(self, body: dict) -> httpx.Response:
        """Create a streaming native Anthropic messages response."""
        request = self._client.build_request(
            "POST",
            "/v1/messages",
            json=body,
            headers=self._request_headers(),
        )
        return await self._client.send(request, stream=True)
