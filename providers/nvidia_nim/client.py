"""NVIDIA NIM provider implementation."""

import json
from typing import Any

import openai
from loguru import logger

from config.nim import NimSettings
from providers.base import ProviderConfig
from providers.openai_compat import OpenAICompatibleProvider

from .request import build_request_body, clone_body_without_reasoning_budget

NVIDIA_NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"


class NvidiaNimProvider(OpenAICompatibleProvider):
    """NVIDIA NIM provider using official OpenAI client."""

    def __init__(self, config: ProviderConfig, *, nim_settings: NimSettings):
        super().__init__(
            config,
            provider_name="NIM",
            base_url=config.base_url or NVIDIA_NIM_BASE_URL,
            api_key=config.api_key,
        )
        self._nim_settings = nim_settings

    def _build_request_body(self, request: Any) -> dict:
        """Internal helper for tests and shared building."""
        return build_request_body(
            request,
            self._nim_settings,
            thinking_enabled=self._is_thinking_enabled(request),
        )

    def _get_retry_request_body(self, error: Exception, body: dict) -> dict | None:
        """Retry once without reasoning_budget when NIM rejects that field."""
        status_code = getattr(error, "status_code", None)
        if not isinstance(error, openai.BadRequestError) and status_code != 400:
            return None

        error_text = str(error)
        error_body = getattr(error, "body", None)
        if error_body is not None:
            error_text = f"{error_text} {json.dumps(error_body, default=str)}"
        if "reasoning_budget" not in error_text.lower():
            return None

        retry_body = clone_body_without_reasoning_budget(body)
        if retry_body is None:
            return None

        logger.warning("NIM_STREAM: retrying without reasoning_budget after 400 error")
        return retry_body
