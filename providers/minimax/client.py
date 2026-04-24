"""MiniMax provider implementation."""

from typing import Any

from providers.base import ProviderConfig
from providers.openai_compat import OpenAICompatibleProvider

from .request import build_request_body

MINIMAX_BASE_URL = "https://api.minimax.io/v1"


class MinimaxProvider(OpenAICompatibleProvider):
    """MiniMax provider using OpenAI-compatible chat completions."""

    def __init__(self, config: ProviderConfig):
        super().__init__(
            config,
            provider_name="MINIMAX",
            base_url=config.base_url or MINIMAX_BASE_URL,
            api_key=config.api_key,
        )

    def _build_request_body(self, request: Any) -> dict:
        return build_request_body(request)
