"""Request builder for Kilo AI Gateway provider."""

from typing import Any

from loguru import logger

from providers.common.message_converter import build_base_request_body

KILO_DEFAULT_MAX_TOKENS = 81920


def build_request_body(request_data: Any, *, thinking_enabled: bool) -> dict:
    """Build OpenAI-format request body from Anthropic request for Kilo."""
    logger.debug(
        "KILO_REQUEST: conversion start model={} msgs={}",
        getattr(request_data, "model", "?"),
        len(getattr(request_data, "messages", [])),
    )
    body = build_base_request_body(
        request_data,
        include_thinking=thinking_enabled,
        default_max_tokens=KILO_DEFAULT_MAX_TOKENS,
        include_reasoning_for_openrouter=False,
    )

    logger.debug(
        "KILO_REQUEST: conversion done model={} msgs={} tools={}",
        body.get("model"),
        len(body.get("messages", [])),
        len(body.get("tools", [])),
    )
    return body
