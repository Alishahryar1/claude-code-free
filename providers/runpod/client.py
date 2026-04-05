"""RUNPOD provider implementation."""

from typing import Any

from providers.base import ProviderConfig
from providers.openai_compat import OpenAICompatibleProvider

from .request import build_request_body

RUNPOD_BASE_URL = "https://api.runpod.io/v1"


class RunpodProvider(OpenAICompatibleProvider):
    """Runpod provider using OpenAI-compatible API."""

    def __init__(self, config: ProviderConfig):
        super().__init__(
            config,
            provider_name="RUNPOD",
            base_url=config.base_url or RUNPOD_BASE_URL,
            api_key=config.api_key,
        )

    def _build_request_body(self, request: Any) -> dict:
        """Internal helper for tests and shared building."""
        return build_request_body(request)
