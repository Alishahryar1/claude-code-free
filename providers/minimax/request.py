"""Request builder for MiniMax provider."""

from typing import Any

from loguru import logger

from providers.common.message_converter import build_base_request_body

MINIMAX_DEFAULT_MAX_TOKENS = 81920


def build_request_body(request_data: Any) -> dict:
    """Build OpenAI-format request body from Anthropic request for MiniMax."""
    logger.debug(
        "MINIMAX_REQUEST: conversion start model={} msgs={}",
        getattr(request_data, "model", "?"),
        len(getattr(request_data, "messages", [])),
    )
    body = build_base_request_body(
        request_data,
        default_max_tokens=MINIMAX_DEFAULT_MAX_TOKENS,
    )

    # MiniMax requires temperature > 0; clamp to 0.01 if caller passes 0.
    if body.get("temperature", 1.0) == 0:
        body["temperature"] = 0.01

    logger.debug(
        "MINIMAX_REQUEST: conversion done model={} msgs={} tools={}",
        body.get("model"),
        len(body.get("messages", [])),
        len(body.get("tools", [])),
    )
    return body
