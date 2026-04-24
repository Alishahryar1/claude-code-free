"""Cloudflare Workers AI provider implementation."""

from typing import Any

from providers.base import ProviderConfig
from providers.openai_compat import OpenAICompatibleProvider

from .request import build_request_body

CLOUDFLARE_BASE_URL_TEMPLATE = (
    "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1"
)


class CloudflareProvider(OpenAICompatibleProvider):
    """Cloudflare Workers AI provider using its OpenAI-compatible endpoint."""

    def __init__(self, config: ProviderConfig):
        super().__init__(
            config,
            provider_name="CLOUDFLARE",
            base_url=config.base_url,
            api_key=config.api_key,
        )

    def _build_request_body(self, request: Any) -> dict:
        return build_request_body(
            request,
            thinking_enabled=self._is_thinking_enabled(request),
        )
