"""Request builder for Modal Research provider."""

from typing import Any

from loguru import logger

from providers.common.message_converter import build_base_request_body

MODAL_DEFAULT_MAX_TOKENS = 81920


def build_request_body(request_data: Any) -> dict:
    """Build OpenAI-format request body from Anthropic request for Modal.

    Modal Research uses a standard OpenAI-compatible API, so we just use
    the base converter without special handling.
    """
    logger.debug(
        "MODAL_REQUEST: conversion start model={} msgs={}",
        getattr(request_data, "model", "?"),
        len(getattr(request_data, "messages", [])),
    )

    body = build_base_request_body(
        request_data,
        default_max_tokens=MODAL_DEFAULT_MAX_TOKENS,
    )

    logger.debug(
        "MODAL_REQUEST: conversion done model={} msgs={} tools={}",
        body.get("model"),
        len(body.get("messages", [])),
        len(body.get("tools", [])),
    )
    return body
