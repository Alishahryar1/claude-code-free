"""Provider fallback orchestration.

When a primary provider fails before streaming begins, this module
transparently retries with a configured fallback provider.
"""

from collections.abc import AsyncIterator

import httpx
import openai
from loguru import logger

from config.settings import Settings
from providers.base import BaseProvider
from providers.exceptions import (
    APIError,
    AuthenticationError,
    InvalidRequestError,
    OverloadedError,
    RateLimitError,
)

from .dependencies import get_provider_for_type
from .models.anthropic import MessagesRequest


def is_fallback_trigger(error: Exception) -> bool:
    """Return True if this error should trigger a provider fallback.

    Only errors indicating the provider is unavailable or overloaded
    trigger fallback. Client errors (bad request, auth) do not.
    """
    # Auth / bad request — configuration or request problem, not provider
    if isinstance(error, (AuthenticationError, openai.AuthenticationError)):
        return False
    if isinstance(error, (InvalidRequestError, openai.BadRequestError)):
        return False

    # Rate limit after retries exhausted
    if isinstance(error, (openai.RateLimitError, RateLimitError)):
        return True

    # Server errors / overloaded
    if isinstance(error, OverloadedError):
        return True
    if isinstance(error, openai.InternalServerError):
        return True
    if isinstance(error, APIError) and error.status_code >= 500:
        return True

    # Connection failures
    if isinstance(error, (httpx.ConnectError, httpx.ConnectTimeout)):
        return True

    # Read timeout
    if isinstance(error, (httpx.ReadTimeout, TimeoutError)):
        return True

    # HTTPX status errors for 5xx / 429
    if isinstance(error, httpx.HTTPStatusError):
        status = error.response.status_code
        return status >= 500 or status == 429

    # Generic openai.APIError with 5xx or 429 status
    if isinstance(error, openai.APIError):
        status = getattr(error, "status_code", None)
        return isinstance(status, int) and (status >= 500 or status == 429)

    return False


async def _passthrough(
    provider: BaseProvider,
    request: MessagesRequest,
    input_tokens: int,
    request_id: str | None,
) -> AsyncIterator[str]:
    """Yield all events from a provider with no fallback wrapping."""
    async for event in provider.stream_response(
        request, input_tokens, request_id=request_id
    ):
        yield event


async def _stream_with_fallback_impl(
    primary_provider: BaseProvider,
    request: MessagesRequest,
    input_tokens: int,
    request_id: str | None,
    fallback_model: str,
) -> AsyncIterator[str]:
    """Stream response, falling back to another provider on failure."""
    try:
        async for event in primary_provider.stream_response(
            request, input_tokens, request_id=request_id
        ):
            yield event
        return
    except Exception as primary_error:
        if not is_fallback_trigger(primary_error):
            raise

        logger.warning(
            "FALLBACK: primary provider failed ({}), "
            "trying fallback {} for request_id={}",
            type(primary_error).__name__,
            fallback_model,
            request_id,
        )

    # Try fallback provider
    fallback_provider_type = Settings.parse_provider_type(fallback_model)
    fallback_provider = get_provider_for_type(fallback_provider_type)
    fallback_model_name = Settings.parse_model_name(fallback_model)

    original_model = request.model
    request.model = fallback_model_name
    try:
        async for event in fallback_provider.stream_response(
            request, input_tokens, request_id=request_id
        ):
            yield event
    except Exception as fallback_error:
        logger.error(
            "FALLBACK: both primary and fallback ({}) failed for request_id={}. "
            "Fallback error: {}",
            fallback_provider_type,
            request_id,
            type(fallback_error).__name__,
        )
        raise
    finally:
        request.model = original_model


def stream_with_fallback(
    primary_provider: BaseProvider,
    request: MessagesRequest,
    input_tokens: int,
    request_id: str | None,
    *,
    fallback_model: str | None,
) -> AsyncIterator[str]:
    """Stream response with optional provider fallback.

    Tries the primary provider first. If its generator raises before
    yielding any events and a fallback is configured for a triggering
    error, the fallback provider is tried instead.

    When no fallback is configured, this is a transparent passthrough
    that preserves the original calling convention.
    """
    if fallback_model is None:
        return primary_provider.stream_response(
            request, input_tokens, request_id=request_id
        )
    return _stream_with_fallback_impl(
        primary_provider, request, input_tokens, request_id, fallback_model
    )
