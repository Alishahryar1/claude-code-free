"""Modal provider implementation."""

from typing import Any

from providers.base import ProviderConfig
from providers.openai_compat import OpenAICompatibleProvider

from .request import build_request_body

MODAL_GLV5_BASE_URL = "https://api.us-west-2.modal.direct/v1"


class ModalProvider(OpenAICompatibleProvider):
    """Modal GLV5 provider using OpenAI-compatible API."""

    def __init__(self, config: ProviderConfig):
        super().__init__(
            config,
            provider_name="MODAL",
            base_url=config.base_url or MODAL_GLV5_BASE_URL,
            api_key=config.api_key,
        )

    def _build_request_body(self, request: Any) -> dict:
        """Internal helper for tests and shared building."""
        return build_request_body(request)
