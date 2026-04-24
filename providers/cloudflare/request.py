"""Request builder for Cloudflare Workers AI provider."""

from typing import Any

from loguru import logger

from providers.common.message_converter import build_base_request_body

CLOUDFLARE_DEFAULT_MAX_TOKENS = 8192


def build_request_body(request_data: Any, *, thinking_enabled: bool) -> dict:
    """Build OpenAI-format request body from Anthropic request for Cloudflare."""
    logger.debug(
        "CLOUDFLARE_REQUEST: conversion start model={} msgs={}",
        getattr(request_data, "model", "?"),
        len(getattr(request_data, "messages", [])),
    )
    body = build_base_request_body(
        request_data,
        include_thinking=thinking_enabled,
        default_max_tokens=CLOUDFLARE_DEFAULT_MAX_TOKENS,
    )

    extra_body: dict[str, Any] = {}
    request_extra = getattr(request_data, "extra_body", None)
    if request_extra:
        extra_body.update(request_extra)
    if extra_body:
        body["extra_body"] = extra_body

    logger.debug(
        "CLOUDFLARE_REQUEST: conversion done model={} msgs={} tools={}",
        body.get("model"),
        len(body.get("messages", [])),
        len(body.get("tools", [])),
    )
    return body
