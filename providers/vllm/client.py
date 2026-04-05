"""VLLM provider implementation."""

from typing import Any

from providers.base import ProviderConfig
from providers.openai_compat import OpenAICompatibleProvider

from .request import build_request_body

VLLM_BASE_URL = "http://localhost:8000/v1"


class VllmProvider(OpenAICompatibleProvider):
    """Vllm provider using OpenAI-compatible API."""

    def __init__(self, config: ProviderConfig):
        super().__init__(
            config,
            provider_name="VLLM",
            base_url=config.base_url or VLLM_BASE_URL,
            api_key=config.api_key,
        )

    def _build_request_body(self, request: Any) -> dict:
        """Internal helper for tests and shared building."""
        return build_request_body(request)
