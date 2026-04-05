"""Request builder for Anthropic provider."""

from typing import Any

from loguru import logger

from providers.common.message_converter import build_base_request_body

ANTHROPIC_DEFAULT_MAX_TOKENS = 81920


def build_request_body(request_data: Any) -> dict:
    """Build OpenAI-format request body from Anthropic request for Anthropic."""
    logger.debug(
        "ANTHROPIC_REQUEST: conversion start model={} msgs={}",
        getattr(request_data, "model", "?"),
        len(getattr(request_data, "messages", [])),
    )
    body = build_base_request_body(
        request_data,
        default_max_tokens=ANTHROPIC_DEFAULT_MAX_TOKENS,
    )

    logger.debug(
        "ANTHROPIC_REQUEST: conversion done model={} msgs={} tools={}",
        body.get("model"),
        len(body.get("messages", [])),
        len(body.get("tools", [])),
    )
    return body
