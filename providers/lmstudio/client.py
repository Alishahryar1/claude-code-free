"""LM Studio provider implementation."""

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
from loguru import logger

from providers.base import BaseProvider, ProviderConfig
from providers.common import get_user_facing_error_message, map_error
from providers.rate_limit import GlobalRateLimiter

LMSTUDIO_DEFAULT_BASE_URL = "http://localhost:1234/v1"


class LMStudioProvider(BaseProvider):
    """LM Studio provider using native Anthropic Messages API endpoint."""

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._provider_name = "LMSTUDIO"
        self._base_url = (config.base_url or LMSTUDIO_DEFAULT_BASE_URL).rstrip("/")

        # We need the base URL without /v1 if the user provided it with /v1,
        # so we can append /v1/messages safely.
        # Actually, if they provided http://localhost:1234/v1, we can just use
        # {base_url}/messages which becomes http://localhost:1234/v1/messages

        self._global_rate_limiter = GlobalRateLimiter.get_instance(
            rate_limit=config.rate_limit,
            rate_window=config.rate_window,
            max_concurrency=config.max_concurrency,
        )
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            proxy=config.proxy or None,
            timeout=httpx.Timeout(
                config.http_read_timeout,
                connect=config.http_connect_timeout,
                read=config.http_read_timeout,
                write=config.http_write_timeout,
            ),
        )

    async def cleanup(self) -> None:
        """Release HTTP client resources."""
        await self._client.aclose()

    async def stream_response(
        self,
        request: Any,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
    ) -> AsyncIterator[str]:
        """Stream response natively via LM Studio's Anthropic-compatible endpoint."""
        tag = self._provider_name
        req_tag = f" request_id={request_id}" if request_id else ""
        thinking_enabled = self._is_thinking_enabled(request)
        response: httpx.Response | None = None

        # Dump the Anthropic Pydantic model directly into a dict
        body = request.model_dump(exclude_none=True)

        # Remove extra_body, original_model, resolved_provider_model which are internal
        body.pop("extra_body", None)
        body.pop("original_model", None)
        body.pop("resolved_provider_model", None)

        # Translate internal ThinkingConfig to Anthropic API schema
        if "thinking" in body:
            thinking_cfg = body.pop("thinking")
            if (
                thinking_enabled
                and isinstance(thinking_cfg, dict)
                and thinking_cfg.get("enabled")
            ):
                # Anthropic API requires a budget_tokens value when enabled
                body["thinking"] = {"type": "enabled"}

        # Ensure max_tokens is present (Claude API requires it)
        if "max_tokens" not in body:
            body["max_tokens"] = 81920

        logger.info(
            "{}_STREAM:{} natively passing Anthropic request to LMStudio model={} msgs={} tools={}",
            tag,
            req_tag,
            body.get("model"),
            len(body.get("messages", [])),
            len(body.get("tools", [])),
        )

        async with self._global_rate_limiter.concurrency_slot():
            try:
                # We use execute_with_retry around the streaming request context
                # To do this safely with httpx streaming, we await the chunk stream

                async def _make_request():
                    request_obj = self._client.build_request(
                        "POST",
                        "/messages",
                        json=body,
                        headers={"Content-Type": "application/json"},
                    )
                    return await self._client.send(request_obj, stream=True)

                response = await self._global_rate_limiter.execute_with_retry(
                    _make_request
                )

                if response.status_code != 200:
                    try:
                        response.raise_for_status()
                    except httpx.HTTPStatusError as e:
                        logger.error(
                            "{}_ERROR:{} HTTP {}",
                            tag,
                            req_tag,
                            response.status_code,
                        )
                        raise e

                async for line in response.aiter_lines():
                    if line:
                        yield f"{line}\n"
                    else:
                        yield "\n"

            except Exception as e:
                logger.error("{}_ERROR:{} {}", tag, req_tag, type(e).__name__)
                mapped_e = map_error(e)
                if getattr(mapped_e, "status_code", None) == 405:
                    error_message = (
                        f"Upstream provider {tag} rejected the request method "
                        "or endpoint (HTTP 405)."
                    )
                else:
                    error_message = get_user_facing_error_message(
                        mapped_e, read_timeout_s=self._config.http_read_timeout
                    )
                if request_id:
                    error_message += f"\nRequest ID: {request_id}"

                logger.info(
                    "{}_STREAM: Emitting native SSE error event for {}{}",
                    tag,
                    type(e).__name__,
                    req_tag,
                )

                # Emit an Anthropic-compatible error event
                error_event = {
                    "type": "error",
                    "error": {"type": "api_error", "message": error_message},
                }
                yield f"event: error\ndata: {json.dumps(error_event)}\n\n"
            finally:
                if response is not None:
                    await response.aclose()
