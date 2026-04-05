"""HuggingFace provider implementation."""

from typing import Any

from providers.base import ProviderConfig
from providers.openai_compat import OpenAICompatibleProvider

from .request import build_request_body

# HuggingFace requires model ID in the URL path
# Format: https://api-inference.huggingface.co/models/{model_id}/v1
# Users must set HUGGINGFACE_BASE_URL in .env with their specific model
HUGGINGFACE_BASE_URL = ""


class HuggingFaceProvider(OpenAICompatibleProvider):
    """HuggingFace provider using OpenAI-compatible API.

    Note: HuggingFace requires the model ID to be part of the base URL.
    Users must set HUGGINGFACE_BASE_URL environment variable with format:
    https://api-inference.huggingface.co/models/{model_id}/v1
    """

    def __init__(self, config: ProviderConfig):
        base_url = config.base_url or HUGGINGFACE_BASE_URL
        if not base_url or not base_url.strip():
            raise ValueError(
                "HUGGINGFACE_BASE_URL must be set and include model ID. "
                "Format: https://api-inference.huggingface.co/models/{model_id}/v1"
            )

        super().__init__(
            config,
            provider_name="HUGGINGFACE",
            base_url=base_url,
            api_key=config.api_key,
        )

    def _build_request_body(self, request: Any) -> dict:
        """Internal helper for tests and shared building."""
        return build_request_body(request)
