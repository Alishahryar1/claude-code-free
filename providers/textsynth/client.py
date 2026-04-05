"""TEXTSYNTH provider implementation."""

from typing import Any

from providers.base import ProviderConfig
from providers.openai_compat import OpenAICompatibleProvider

from .request import build_request_body

TEXTSYNTH_BASE_URL = "https://api.textsynth.com/v1"


class TextsynthProvider(OpenAICompatibleProvider):
    """Textsynth provider using OpenAI-compatible API."""

    def __init__(self, config: ProviderConfig):
        super().__init__(
            config,
            provider_name="TEXTSYNTH",
            base_url=config.base_url or TEXTSYNTH_BASE_URL,
            api_key=config.api_key,
        )

    def _build_request_body(self, request: Any) -> dict:
        """Internal helper for tests and shared building."""
        return build_request_body(request)
