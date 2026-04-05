"""NOVITA provider implementation."""

from typing import Any

from providers.base import ProviderConfig
from providers.openai_compat import OpenAICompatibleProvider

from .request import build_request_body

NOVITA_BASE_URL = "https://api.novita.ai/v1"


class NovitaProvider(OpenAICompatibleProvider):
    """Novita provider using OpenAI-compatible API."""

    def __init__(self, config: ProviderConfig):
        super().__init__(
            config,
            provider_name="NOVITA",
            base_url=config.base_url or NOVITA_BASE_URL,
            api_key=config.api_key,
        )

    def _build_request_body(self, request: Any) -> dict:
        """Internal helper for tests and shared building."""
        return build_request_body(request)
