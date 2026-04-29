"""Request builder for Custom OpenAI provider."""

from typing import Any

from loguru import logger

from core.anthropic import (
    OpenAIConversionError,
    ReasoningReplayMode,
    build_base_request_body,
)
from providers.exceptions import InvalidRequestError

_RESERVED_EXTRA_BODY_KEYS = frozenset(
    {
        "model",
        "messages",
        "max_tokens",
        "temperature",
        "top_p",
        "stop",
        "tools",
        "tool_choice",
    }
)


def build_request_body(request_data: Any, *, thinking_enabled: bool) -> dict:
    """Build OpenAI-format request body from Anthropic request for custom OpenAI API."""
    logger.debug(
        "CUSTOM_OPENAI_REQUEST: conversion start model={} msgs={}",
        getattr(request_data, "model", "?"),
        len(getattr(request_data, "messages", [])),
    )
    try:
        body = build_base_request_body(
            request_data,
            reasoning_replay=ReasoningReplayMode.REASONING_CONTENT
            if thinking_enabled
            else ReasoningReplayMode.DISABLED,
        )
    except OpenAIConversionError as exc:
        raise InvalidRequestError(str(exc)) from exc

    # Pass through any extra_body parameters for provider-specific customization
    extra_body = getattr(request_data, "extra_body", None)
    if extra_body:
        reserved = sorted(_RESERVED_EXTRA_BODY_KEYS.intersection(extra_body))
        if reserved:
            joined = ", ".join(reserved)
            raise InvalidRequestError(
                f"custom_openai extra_body must not override reserved fields: {joined}"
            )
        body["extra_body"] = extra_body

    logger.debug(
        "CUSTOM_OPENAI_REQUEST: conversion done model={} msgs={} tools={}",
        body.get("model"),
        len(body.get("messages", [])),
        len(body.get("tools", [])),
    )
    return body
