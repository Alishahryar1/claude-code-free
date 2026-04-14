"""Request builder for Modal provider."""

from typing import Any

from loguru import logger

from providers.common.message_converter import build_base_request_body


def build_request_body(request_data: Any) -> dict:
    """Build OpenAI-format request body from Anthropic request.

    Modal's GLV5 endpoint is OpenAI-compatible, so we use the standard
    conversion with no special handling needed.
    """
    logger.debug(
        "MODAL_REQUEST: conversion start model={} msgs={}",
        getattr(request_data, "model", "?"),
        len(getattr(request_data, "messages", [])),
    )

    body = build_base_request_body(request_data)

    # Ensure max_tokens is present (required by most endpoints)
    if "max_tokens" not in body:
        body["max_tokens"] = 4096

    logger.debug(
        "MODAL_REQUEST: conversion done model={} msgs={} tools={}",
        body.get("model"),
        len(body.get("messages", [])),
        len(body.get("tools", [])),
    )

    return body
