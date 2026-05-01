"""Application services for the Claude-compatible API."""

from __future__ import annotations

import json
import traceback
import uuid
from collections.abc import AsyncIterator, Callable
from typing import Any

from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger

from config.settings import Settings
from core.anthropic import (
    get_token_count,
    get_user_facing_error_message,
    iter_provider_stream_error_sse_events,
)
from core.anthropic.sse import ANTHROPIC_SSE_RESPONSE_HEADERS
from providers.base import BaseProvider
from providers.exceptions import InvalidRequestError, ProviderError

from .model_router import ModelRouter, RoutedMessagesRequest
from .models.anthropic import MessagesRequest, TokenCountRequest
from .models.responses import TokenCountResponse
from .optimization_handlers import try_optimizations
from .web_tools.egress import WebFetchEgressPolicy
from .web_tools.request import (
    is_web_server_tool_request,
    openai_chat_upstream_server_tool_error,
)
from .web_tools.streaming import stream_web_server_tool_response

TokenCounter = Callable[[list[Any], str | list[Any] | None, list[Any] | None], int]

ProviderGetter = Callable[[str], BaseProvider]

# Providers that use ``/chat/completions`` + Anthropic-to-OpenAI conversion (not native Messages).
_OPENAI_CHAT_UPSTREAM_IDS = frozenset({"nvidia_nim"})

_FALLBACK_MODEL_REFS = (
    # Current OpenRouter free, tool-capable variants checked from /api/v1/models.
    "open_router/openai/gpt-oss-20b:free",
    "open_router/nvidia/nemotron-nano-9b-v2:free",
    "open_router/openrouter/free",
    "deepseek/deepseek-v4-pro",
)

_INITIAL_PROVIDER_ERROR_PREFIXES = (
    "Provider request timed out",
    "Could not connect to provider",
    "Request timed out",
    "Provider rate limit reached",
    "Provider authentication failed",
    "Invalid request sent to provider",
    "Provider is currently overloaded",
    "Provider is temporarily unavailable",
    "Provider API request failed",
    "Provider request failed",
    "Upstream provider ",
)


def anthropic_sse_streaming_response(
    body: AsyncIterator[str],
) -> StreamingResponse:
    """Return a :class:`StreamingResponse` for Anthropic-style SSE streams."""
    return StreamingResponse(
        body,
        media_type="text/event-stream",
        headers=ANTHROPIC_SSE_RESPONSE_HEADERS,
    )


def _http_status_for_unexpected_service_exception(_exc: BaseException) -> int:
    """HTTP status for uncaught non-provider failures (stable client contract)."""
    return 500


def _log_unexpected_service_exception(
    settings: Settings,
    exc: BaseException,
    *,
    context: str,
    request_id: str | None = None,
) -> None:
    """Log service-layer failures without echoing exception text unless opted in."""
    if settings.log_api_error_tracebacks:
        if request_id is not None:
            logger.error("{} request_id={}: {}", context, request_id, exc)
        else:
            logger.error("{}: {}", context, exc)
        logger.error(traceback.format_exc())
        return
    if request_id is not None:
        logger.error(
            "{} request_id={} exc_type={}",
            context,
            request_id,
            type(exc).__name__,
        )
    else:
        logger.error("{} exc_type={}", context, type(exc).__name__)


def _require_non_empty_messages(messages: list[Any]) -> None:
    if not messages:
        raise InvalidRequestError("messages cannot be empty")


def _dedupe_routed_attempts(
    attempts: list[RoutedMessagesRequest],
) -> tuple[RoutedMessagesRequest, ...]:
    seen: set[tuple[str, str, bool]] = set()
    unique: list[RoutedMessagesRequest] = []
    for attempt in attempts:
        key = (
            attempt.resolved.provider_id,
            attempt.resolved.provider_model,
            attempt.resolved.thinking_enabled,
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(attempt)
    return tuple(unique)


def _sse_events_from_chunk(chunk: str) -> list[tuple[str | None, dict[str, Any]]]:
    events: list[tuple[str | None, dict[str, Any]]] = []
    for frame in chunk.split("\n\n"):
        if not frame.strip():
            continue
        event_name: str | None = None
        data_lines: list[str] = []
        for line in frame.splitlines():
            if line.startswith("event:"):
                event_name = line.removeprefix("event:").strip()
            elif line.startswith("data:"):
                data_lines.append(line.removeprefix("data:").strip())
        if not data_lines:
            continue
        try:
            data = json.loads("\n".join(data_lines))
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            events.append((event_name, data))
    return events


def _initial_provider_error_message(chunks: list[str]) -> str | None:
    text_deltas: list[str] = []
    saw_non_error_content = False

    for chunk in chunks:
        for event_name, data in _sse_events_from_chunk(chunk):
            if event_name == "error":
                error = data.get("error")
                if isinstance(error, dict):
                    message = error.get("message")
                    if isinstance(message, str) and message.strip():
                        return message

            if data.get("type") == "content_block_delta":
                delta = data.get("delta")
                if isinstance(delta, dict) and delta.get("type") == "text_delta":
                    text = delta.get("text")
                    if isinstance(text, str):
                        text_deltas.append(text)
                    continue
                saw_non_error_content = True

            if data.get("type") == "content_block_start":
                content_block = data.get("content_block")
                if (
                    isinstance(content_block, dict)
                    and content_block.get("type") != "text"
                ):
                    saw_non_error_content = True

    if saw_non_error_content or len(text_deltas) != 1:
        return None

    message = text_deltas[0].strip()
    if message.startswith(_INITIAL_PROVIDER_ERROR_PREFIXES):
        return message
    return None


def _should_release_initial_chunks(chunks: list[str]) -> bool:
    for chunk in chunks:
        for event_name, data in _sse_events_from_chunk(chunk):
            if event_name == "error":
                return False

            event_type = data.get("type")
            if event_type == "content_block_delta":
                delta = data.get("delta")
                if isinstance(delta, dict) and delta.get("type") == "text_delta":
                    text = delta.get("text")
                    if isinstance(text, str):
                        return not text.strip().startswith(
                            _INITIAL_PROVIDER_ERROR_PREFIXES
                        )
                return True

            if event_type == "content_block_start":
                content_block = data.get("content_block")
                if (
                    isinstance(content_block, dict)
                    and content_block.get("type") != "text"
                ):
                    return True

            if event_type == "message_stop":
                return True

    return False


class ClaudeProxyService:
    """Coordinate request optimization, model routing, token count, and providers."""

    def __init__(
        self,
        settings: Settings,
        provider_getter: ProviderGetter,
        model_router: ModelRouter | None = None,
        token_counter: TokenCounter = get_token_count,
    ):
        self._settings = settings
        self._provider_getter = provider_getter
        self._model_router = model_router or ModelRouter(settings)
        self._token_counter = token_counter

    def create_message(self, request_data: MessagesRequest) -> object:
        """Create a message response or streaming response."""
        try:
            _require_non_empty_messages(request_data.messages)

            routed = self._model_router.resolve_messages_request(request_data)
            if routed.resolved.provider_id in _OPENAI_CHAT_UPSTREAM_IDS:
                tool_err = openai_chat_upstream_server_tool_error(
                    routed.request,
                    web_tools_enabled=self._settings.enable_web_server_tools,
                )
                if tool_err is not None:
                    raise InvalidRequestError(tool_err)

            if self._settings.enable_web_server_tools and is_web_server_tool_request(
                routed.request
            ):
                input_tokens = self._token_counter(
                    routed.request.messages, routed.request.system, routed.request.tools
                )
                logger.info("Optimization: Handling Anthropic web server tool")
                egress = WebFetchEgressPolicy(
                    allow_private_network_targets=self._settings.web_fetch_allow_private_networks,
                    allowed_schemes=self._settings.web_fetch_allowed_scheme_set(),
                )
                return anthropic_sse_streaming_response(
                    stream_web_server_tool_response(
                        routed.request,
                        input_tokens=input_tokens,
                        web_fetch_egress=egress,
                        verbose_client_errors=self._settings.log_api_error_tracebacks,
                    ),
                )

            optimized = try_optimizations(routed.request, self._settings)
            if optimized is not None:
                return optimized
            logger.debug("No optimization matched, routing to provider")

            request_id = f"req_{uuid.uuid4().hex[:12]}"
            logger.info(
                "API_REQUEST: request_id={} model={} messages={}",
                request_id,
                routed.request.model,
                len(routed.request.messages),
            )
            if self._settings.log_raw_api_payloads:
                logger.debug(
                    "FULL_PAYLOAD [{}]: {}", request_id, routed.request.model_dump()
                )

            input_tokens = self._token_counter(
                routed.request.messages, routed.request.system, routed.request.tools
            )
            provider = self._provider_getter(routed.resolved.provider_id)
            provider.preflight_stream(
                routed.request,
                thinking_enabled=routed.resolved.thinking_enabled,
            )
            first_stream = provider.stream_response(
                routed.request,
                input_tokens=input_tokens,
                request_id=request_id,
                thinking_enabled=routed.resolved.thinking_enabled,
            )
            return anthropic_sse_streaming_response(
                self._stream_with_provider_fallbacks(
                    request_data,
                    routed,
                    input_tokens=input_tokens,
                    request_id=request_id,
                    first_stream=first_stream,
                ),
            )

        except ProviderError:
            raise
        except Exception as e:
            _log_unexpected_service_exception(
                self._settings, e, context="CREATE_MESSAGE_ERROR"
            )
            raise HTTPException(
                status_code=_http_status_for_unexpected_service_exception(e),
                detail=get_user_facing_error_message(e),
            ) from e

    def _message_attempts(
        self, original_request: MessagesRequest, routed: RoutedMessagesRequest
    ) -> tuple[RoutedMessagesRequest, ...]:
        attempts = [routed]
        attempts.extend(
            self._model_router.resolve_messages_request(
                original_request.model_copy(update={"model": model_ref}, deep=True)
            )
            for model_ref in _FALLBACK_MODEL_REFS
        )
        return _dedupe_routed_attempts(attempts)

    async def _stream_with_provider_fallbacks(
        self,
        original_request: MessagesRequest,
        routed: RoutedMessagesRequest,
        *,
        input_tokens: int,
        request_id: str,
        first_stream: AsyncIterator[str] | None = None,
    ) -> AsyncIterator[str]:
        attempts = self._message_attempts(original_request, routed)
        last_error_message: str | None = None

        for attempt_index, attempt in enumerate(attempts, start=1):
            if attempt_index == 1 and first_stream is not None:
                stream = first_stream
            else:
                try:
                    provider = self._provider_getter(attempt.resolved.provider_id)
                    provider.preflight_stream(
                        attempt.request,
                        thinking_enabled=attempt.resolved.thinking_enabled,
                    )
                    stream = provider.stream_response(
                        attempt.request,
                        input_tokens=input_tokens,
                        request_id=request_id,
                        thinking_enabled=attempt.resolved.thinking_enabled,
                    )
                except Exception as exc:
                    last_error_message = get_user_facing_error_message(exc)
                    logger.warning(
                        "MODEL FALLBACK: preflight failed request_id={} attempt={}/{} provider={} model={} exc_type={}",
                        request_id,
                        attempt_index,
                        len(attempts),
                        attempt.resolved.provider_id,
                        attempt.resolved.provider_model,
                        type(exc).__name__,
                    )
                    continue

            is_last_attempt = attempt_index == len(attempts)
            chunks: list[str] = []
            error_message: str | None = None

            if is_last_attempt:
                async for chunk in stream:
                    yield chunk
                return

            released = False
            async for chunk in stream:
                if released:
                    yield chunk
                    continue

                chunks.append(chunk)
                error_message = _initial_provider_error_message(chunks)
                if error_message is not None:
                    break

                if _should_release_initial_chunks(chunks):
                    released = True
                    for buffered_chunk in chunks:
                        yield buffered_chunk
                    chunks.clear()

            if error_message is None:
                for buffered_chunk in chunks:
                    yield buffered_chunk
                return

            last_error_message = error_message
            logger.warning(
                "MODEL FALLBACK: upstream failed before content request_id={} attempt={}/{} provider={} model={} next_attempt=true",
                request_id,
                attempt_index,
                len(attempts),
                attempt.resolved.provider_id,
                attempt.resolved.provider_model,
            )

        fallback_request = routed.request
        error_message = last_error_message or "Provider request failed unexpectedly."
        for event in iter_provider_stream_error_sse_events(
            request=fallback_request,
            input_tokens=input_tokens,
            error_message=error_message,
            sent_any_event=False,
            log_raw_sse_events=self._settings.log_raw_sse_events,
        ):
            yield event

    def count_tokens(self, request_data: TokenCountRequest) -> TokenCountResponse:
        """Count tokens for a request after applying configured model routing."""
        request_id = f"req_{uuid.uuid4().hex[:12]}"
        with logger.contextualize(request_id=request_id):
            try:
                _require_non_empty_messages(request_data.messages)
                routed = self._model_router.resolve_token_count_request(request_data)
                tokens = self._token_counter(
                    routed.request.messages, routed.request.system, routed.request.tools
                )
                logger.info(
                    "COUNT_TOKENS: request_id={} model={} messages={} input_tokens={}",
                    request_id,
                    routed.request.model,
                    len(routed.request.messages),
                    tokens,
                )
                return TokenCountResponse(input_tokens=tokens)
            except ProviderError:
                raise
            except Exception as e:
                _log_unexpected_service_exception(
                    self._settings,
                    e,
                    context="COUNT_TOKENS_ERROR",
                    request_id=request_id,
                )
                raise HTTPException(
                    status_code=_http_status_for_unexpected_service_exception(e),
                    detail=get_user_facing_error_message(e),
                ) from e
