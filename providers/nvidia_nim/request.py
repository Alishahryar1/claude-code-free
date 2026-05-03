"""Request builder for NVIDIA NIM provider."""

from collections.abc import Callable
from copy import deepcopy
from typing import Any

from loguru import logger

from config.nim import NimSettings
from core.anthropic import (
    ReasoningReplayMode,
    build_base_request_body,
    set_if_not_none,
)
from core.anthropic.conversion import OpenAIConversionError
from providers.exceptions import InvalidRequestError

_SCHEMA_VALUE_KEYS = frozenset(
    {
        "additionalProperties",
        "additionalItems",
        "unevaluatedProperties",
        "unevaluatedItems",
        "items",
        "contains",
        "propertyNames",
        "if",
        "then",
        "else",
        "not",
    }
)
_SCHEMA_LIST_KEYS = frozenset({"allOf", "anyOf", "oneOf", "prefixItems"})
_SCHEMA_MAP_KEYS = frozenset(
    {"properties", "patternProperties", "$defs", "definitions", "dependentSchemas"}
)


def _clone_strip_extra_body(
    body: dict[str, Any],
    strip: Callable[[dict[str, Any]], bool],
) -> dict[str, Any] | None:
    """Deep-clone ``body`` and remove fields via ``strip`` on ``extra_body`` only.

    Returns ``None`` when there is no ``extra_body`` dict or ``strip`` reports no change.
    """
    cloned_body = deepcopy(body)
    extra_body = cloned_body.get("extra_body")
    if not isinstance(extra_body, dict):
        return None
    if not strip(extra_body):
        return None
    if not extra_body:
        cloned_body.pop("extra_body", None)
    return cloned_body


def _strip_reasoning_budget_fields(extra_body: dict[str, Any]) -> bool:
    removed = extra_body.pop("reasoning_budget", None) is not None
    chat_template_kwargs = extra_body.get("chat_template_kwargs")
    if (
        isinstance(chat_template_kwargs, dict)
        and chat_template_kwargs.pop("reasoning_budget", None) is not None
    ):
        removed = True
    return removed


def _strip_chat_template_field(extra_body: dict[str, Any]) -> bool:
    return extra_body.pop("chat_template", None) is not None


def _strip_message_reasoning_content(body: dict[str, Any]) -> bool:
    removed = False
    messages = body.get("messages")
    if not isinstance(messages, list):
        return False
    for message in messages:
        if (
            isinstance(message, dict)
            and message.pop("reasoning_content", None) is not None
        ):
            removed = True
    return removed


def _sanitize_nim_schema_node(value: Any) -> tuple[bool, Any]:
    """Remove boolean JSON Schema subschemas that hosted NIM rejects."""
    if isinstance(value, bool):
        return False, None
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            if key in _SCHEMA_VALUE_KEYS:
                keep, sanitized_item = _sanitize_nim_schema_node(item)
                if keep:
                    sanitized[key] = sanitized_item
            elif key in _SCHEMA_LIST_KEYS and isinstance(item, list):
                sanitized_items: list[Any] = []
                for schema_item in item:
                    keep, sanitized_item = _sanitize_nim_schema_node(schema_item)
                    if keep:
                        sanitized_items.append(sanitized_item)
                if sanitized_items:
                    sanitized[key] = sanitized_items
            elif key in _SCHEMA_MAP_KEYS and isinstance(item, dict):
                sanitized_map: dict[str, Any] = {}
                for map_key, schema_item in item.items():
                    keep, sanitized_item = _sanitize_nim_schema_node(schema_item)
                    if keep:
                        sanitized_map[map_key] = sanitized_item
                sanitized[key] = sanitized_map
            else:
                sanitized[key] = item
        return True, sanitized
    if isinstance(value, list):
        sanitized_items = []
        for item in value:
            keep, sanitized_item = _sanitize_nim_schema_node(item)
            if keep:
                sanitized_items.append(sanitized_item)
        return True, sanitized_items
    return True, value


def _sanitize_nim_tool_schemas(body: dict[str, Any]) -> None:
    """Sanitize only tool parameter schemas, preserving tool calls/history."""
    tools = body.get("tools")
    if not isinstance(tools, list):
        return

    sanitized_tools: list[Any] = []
    for tool in tools:
        if not isinstance(tool, dict):
            sanitized_tools.append(tool)
            continue
        sanitized_tool = dict(tool)
        function = tool.get("function")
        if isinstance(function, dict):
            sanitized_function = dict(function)
            parameters = function.get("parameters")
            if isinstance(parameters, dict):
                _, sanitized_parameters = _sanitize_nim_schema_node(parameters)
                sanitized_function["parameters"] = sanitized_parameters
            sanitized_tool["function"] = sanitized_function
        sanitized_tools.append(sanitized_tool)

    body["tools"] = sanitized_tools


def _set_extra(
    extra_body: dict[str, Any], key: str, value: Any, ignore_value: Any = None
) -> None:
    if key in extra_body:
        return
    if value is None:
        return
    if ignore_value is not None and value == ignore_value:
        return
    extra_body[key] = value


def clone_body_without_reasoning_budget(body: dict[str, Any]) -> dict[str, Any] | None:
    """Clone a request body and strip only reasoning_budget fields."""
    return _clone_strip_extra_body(body, _strip_reasoning_budget_fields)


def clone_body_without_chat_template(body: dict[str, Any]) -> dict[str, Any] | None:
    """Clone a request body and strip only chat_template."""
    return _clone_strip_extra_body(body, _strip_chat_template_field)


def clone_body_without_reasoning_content(body: dict[str, Any]) -> dict[str, Any] | None:
    """Clone a request body and strip assistant message ``reasoning_content`` fields."""
    cloned_body = deepcopy(body)
    if not _strip_message_reasoning_content(cloned_body):
        return None
    return cloned_body


def build_request_body(
    request_data: Any, nim: NimSettings, *, thinking_enabled: bool
) -> dict:
    """Build OpenAI-format request body from Anthropic request."""
    logger.debug(
        "NIM_REQUEST: conversion start model={} msgs={}",
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

    _sanitize_nim_tool_schemas(body)

    # NIM-specific max_tokens: cap against nim.max_tokens
    max_tokens = body.get("max_tokens") or getattr(request_data, "max_tokens", None)
    if max_tokens is None:
        max_tokens = nim.max_tokens
    elif nim.max_tokens:
        max_tokens = min(max_tokens, nim.max_tokens)
    set_if_not_none(body, "max_tokens", max_tokens)

    # NIM-specific temperature/top_p: fall back to NIM defaults if request didn't set
    if body.get("temperature") is None and nim.temperature is not None:
        body["temperature"] = nim.temperature
    if body.get("top_p") is None and nim.top_p is not None:
        body["top_p"] = nim.top_p

    # NIM-specific stop sequences fallback
    if "stop" not in body and nim.stop:
        body["stop"] = nim.stop

    if nim.presence_penalty != 0.0:
        body["presence_penalty"] = nim.presence_penalty
    if nim.frequency_penalty != 0.0:
        body["frequency_penalty"] = nim.frequency_penalty
    if nim.seed is not None:
        body["seed"] = nim.seed

    body["parallel_tool_calls"] = nim.parallel_tool_calls

    # Handle non-standard parameters via extra_body
    extra_body: dict[str, Any] = {}
    request_extra = getattr(request_data, "extra_body", None)
    if request_extra:
        extra_body.update(request_extra)

    if thinking_enabled:
        chat_template_kwargs = extra_body.setdefault(
            "chat_template_kwargs", {"thinking": True, "enable_thinking": True}
        )
        if isinstance(chat_template_kwargs, dict):
            chat_template_kwargs.setdefault("reasoning_budget", max_tokens)

    req_top_k = getattr(request_data, "top_k", None)
    top_k = req_top_k if req_top_k is not None else nim.top_k
    _set_extra(extra_body, "top_k", top_k, ignore_value=-1)
    _set_extra(extra_body, "min_p", nim.min_p, ignore_value=0.0)
    _set_extra(
        extra_body, "repetition_penalty", nim.repetition_penalty, ignore_value=1.0
    )
    _set_extra(extra_body, "min_tokens", nim.min_tokens, ignore_value=0)
    _set_extra(extra_body, "chat_template", nim.chat_template)
    _set_extra(extra_body, "request_id", nim.request_id)
    _set_extra(extra_body, "ignore_eos", nim.ignore_eos)

    if extra_body:
        body["extra_body"] = extra_body

    logger.debug(
        "NIM_REQUEST: conversion done model={} msgs={} tools={}",
        body.get("model"),
        len(body.get("messages", [])),
        len(body.get("tools", [])),
    )
    return body
