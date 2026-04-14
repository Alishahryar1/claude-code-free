"""Modal Research provider implementation."""

from providers.base import ProviderConfig
from providers.openai_compat import OpenAICompatibleProvider
from .request import build_request_body

# Modal's OpenAI-compatible endpoint
MODAL_BASE_URL = "https://api.us-west-2.modal.direct/v1"


class ModalProvider(OpenAICompatibleProvider):
    """Modal Research provider using OpenAI-compatible API."""

    def __init__(self, config: ProviderConfig):
        super().__init__(
            config,
            provider_name="MODAL",
            base_url=config.base_url or MODAL_BASE_URL,
            api_key=config.api_key,
        )

    def _build_request_body(self, request) -> dict:
        """Internal helper for tests and shared building."""
        return build_request_body(request)
