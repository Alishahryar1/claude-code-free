"""COHERE provider implementation."""

from typing import Any

from providers.base import ProviderConfig
from providers.openai_compat import OpenAICompatibleProvider

from .request import build_request_body

COHERE_BASE_URL = "https://api.cohere.com/v1"


class CohereProvider(OpenAICompatibleProvider):
    """Cohere provider using OpenAI-compatible API."""

    def __init__(self, config: ProviderConfig):
        super().__init__(
            config,
            provider_name="COHERE",
            base_url=config.base_url or COHERE_BASE_URL,
            api_key=config.api_key,
        )

    def _build_request_body(self, request: Any) -> dict:
        """Internal helper for tests and shared building."""
        return build_request_body(request)
