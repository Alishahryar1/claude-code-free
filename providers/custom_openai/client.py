"""Custom OpenAI provider implementation."""

from collections.abc import Iterator
from typing import Any

from providers.base import ProviderConfig
from providers.common import SSEBuilder
from providers.openai_compat import OpenAICompatibleProvider

from .request import build_request_body


class CustomOpenAIProvider(OpenAICompatibleProvider):
    """Custom OpenAI-compatible provider for user-specified endpoints."""

    def __init__(self, config: ProviderConfig):
        # base_url is guaranteed to be non-None by validation in dependencies.py
        super().__init__(
            config,
            provider_name="CUSTOM_OPENAI",
            base_url=config.base_url or "",
            api_key=config.api_key,
        )

    def _build_request_body(self, request: Any) -> dict:
        """Build request body using standard OpenAI format."""
        return build_request_body(request)

    def _handle_extra_reasoning(self, delta: Any, sse: SSEBuilder) -> Iterator[str]:
        """No provider-specific reasoning handling for custom OpenAI."""
        return iter(())
