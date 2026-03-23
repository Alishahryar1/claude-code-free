"""Request builder for Custom OpenAI provider."""

from typing import Any

from loguru import logger

from providers.common.message_converter import build_base_request_body


def build_request_body(request_data: Any) -> dict:
    """Build OpenAI-format request body from Anthropic request for custom OpenAI API."""
    logger.debug(
        "CUSTOM_OPENAI_REQUEST: conversion start model={} msgs={}",
        getattr(request_data, "model", "?"),
        len(getattr(request_data, "messages", [])),
    )
    body = build_base_request_body(request_data)

    # Pass through any extra_body parameters for provider-specific customization
    extra_body = getattr(request_data, "extra_body", None)
    if extra_body:
        body["extra_body"] = extra_body

    logger.debug(
        "CUSTOM_OPENAI_REQUEST: conversion done model={} msgs={} tools={}",
        body.get("model"),
        len(body.get("messages", [])),
        len(body.get("tools", [])),
    )
    return body
