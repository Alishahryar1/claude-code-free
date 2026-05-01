"""Request builder and DeepSeek native Anthropic compatibility sanitizer."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from loguru import logger

from config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from core.anthropic.native_messages_request import dump_raw_messages_request
from providers.exceptions import InvalidRequestError

# Block types not supported on DeepSeek partial Anthropic-compatible API.
_UNSUPPORTED_MESSAGE_BLOCK_TYPES = frozenset(
    {
        "image",
        "document",
        "server_tool_use",
        "web_search_tool_result",
        "web_fetch_tool_result",
    }
)


def _is_server_listed_tool(tool: Mapping[str, Any]) -> bool:
    """True for Anthropic web_search / web_fetch-style tool definitions (listed tools)."""
    name = (tool.get("name") or "").strip()
    if name in ("web_search", "web_fetch"):
        return True
    typ = tool.get("type")
    if isinstance(typ, str):
        return typ.startswith("web_search") or typ.startswith("web_fetch")
    return False


def _walk_block_list_for_unsupported(blocks: Any, *, where: str) -> None:
    if not isinstance(blocks, list):
        return
    for block in blocks:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype in _UNSUPPORTED_MESSAGE_BLOCK_TYPES:
            raise InvalidRequestError(
                f"DeepSeek native does not support {btype!r} blocks ({where})."
            )
        if btype == "tool_result" and "content" in block:
            _walk_block_list_for_unsupported(
                block["content"], where=f"{where} (tool_result content)"
            )


def _validate_deepseek_native_request_dict(data: dict[str, Any]) -> None:
    mcp = data.get("mcp_servers")
    if mcp:
        raise InvalidRequestError(
            "DeepSeek native does not support mcp_servers on requests."
        )

    for tool in data.get("tools") or ():
        if not isinstance(tool, dict):
            continue
        if _is_server_listed_tool(tool):
            raise InvalidRequestError(
                "DeepSeek native does not support listed Anthropic server tools "
                "(web_search / web_fetch). Remove them or use a different provider."
            )

    for i, message in enumerate(data.get("messages") or ()):
        if not isinstance(message, dict):
            continue
        c = message.get("content")
        if isinstance(c, list):
            _walk_block_list_for_unsupported(c, where=f"messages[{i}].content")
        if isinstance(c, str) and "<think>" in c:
            # Unusual, but block encoded redacted content — treat as unsafe for DeepSeek.
            pass

    system = data.get("system")
    if isinstance(system, list):
        _walk_block_list_for_unsupported(system, where="system")


def _has_tool_history_blocks(message: Mapping[str, Any]) -> bool:
    role = message.get("role")
    content = message.get("content")
    if not isinstance(content, list):
        return False

    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if role == "assistant" and btype == "tool_use":
            return True
        if role == "user" and btype == "tool_result":
            return True
    return False


def _has_replayable_thinking_before_tool_use(message: Mapping[str, Any]) -> bool:
    if message.get("role") != "assistant":
        return False
    content = message.get("content")
    if not isinstance(content, list):
        return False

    has_thinking = False
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "thinking" and isinstance(block.get("thinking"), str):
            has_thinking = bool(block["thinking"])
            continue
        if btype == "tool_use":
            return has_thinking
    return False


def _has_tool_history(data: dict[str, Any]) -> bool:
    for message in data.get("messages") or ():
        if isinstance(message, Mapping) and _has_tool_history_blocks(message):
            return True
    return False


def _has_replayable_tool_thinking(data: dict[str, Any]) -> bool:
    for message in data.get("messages") or ():
        if isinstance(message, Mapping) and _has_replayable_thinking_before_tool_use(
            message
        ):
            return True
    return False


def _remove_deepseek_thinking_hints(data: dict[str, Any]) -> None:
    """Remove request hints that can keep DeepSeek in thinking mode after fallback."""
    output_config = data.get("output_config")
    if isinstance(output_config, dict) and "effort" in output_config:
        cleaned_output_config = dict(output_config)
        cleaned_output_config.pop("effort", None)
        if cleaned_output_config:
            data["output_config"] = cleaned_output_config
        else:
            data.pop("output_config", None)

    context_management = data.get("context_management")
    if not isinstance(context_management, dict):
        return
    edits = context_management.get("edits")
    if not isinstance(edits, list):
        return
    filtered_edits = [
        edit
        for edit in edits
        if not (
            isinstance(edit, dict)
            and isinstance(edit.get("type"), str)
            and edit["type"].startswith("clear_thinking_")
        )
    ]
    if len(filtered_edits) == len(edits):
        return
    cleaned_context_management = dict(context_management)
    if filtered_edits:
        cleaned_context_management["edits"] = filtered_edits
        data["context_management"] = cleaned_context_management
    else:
        cleaned_context_management.pop("edits", None)
        if cleaned_context_management:
            data["context_management"] = cleaned_context_management
        else:
            data.pop("context_management", None)


def sanitize_deepseek_messages_for_native(
    messages: Any, *, thinking_enabled: bool
) -> Any:
    """Filter assistant content for DeepSeek: unsigned ``thinking`` is allowed; no ``redacted_thinking``."""
    if not isinstance(messages, list):
        return messages

    sanitized: list[Any] = []
    for message in messages:
        if not isinstance(message, dict):
            sanitized.append(message)
            continue
        if message.get("role") != "assistant":
            sanitized.append(message)
            continue
        content = message.get("content")
        if not isinstance(content, list):
            sanitized.append(message)
            continue

        if not thinking_enabled:
            filtered = [
                block
                for block in content
                if not (
                    isinstance(block, dict)
                    and block.get("type") in ("thinking", "redacted_thinking")
                )
            ]
        else:
            filtered = [
                block
                for block in content
                if not (
                    isinstance(block, dict) and block.get("type") == "redacted_thinking"
                )
            ]
        new_msg = dict(message)
        new_msg["content"] = filtered or ""
        sanitized.append(new_msg)
    return sanitized


def _strip_reasoning_content_when_native(messages: Any) -> Any:
    """``reasoning_content`` is OpenAI-helper metadata; not part of native Anthropic body."""
    if not isinstance(messages, list):
        return messages
    out: list[Any] = []
    for m in messages:
        if not isinstance(m, dict):
            out.append(m)
            continue
        msg = {k: v for k, v in m.items() if k != "reasoning_content"}
        out.append(msg)
    return out


def build_request_body(request_data: Any, *, thinking_enabled: bool) -> dict:
    """Build a DeepSeek ``/v1/messages`` JSON body (Anthropic format)."""
    logger.debug(
        "DEEPSEEK_REQUEST: native build model={} msgs={}",
        getattr(request_data, "model", "?"),
        len(getattr(request_data, "messages", [])),
    )

    data = dump_raw_messages_request(request_data)
    _validate_deepseek_native_request_dict(data)
    data.pop("extra_body", None)

    has_tool_history = _has_tool_history(data)
    has_replayable_tool_thinking = _has_replayable_tool_thinking(data)
    unsafe_tool_followup = has_tool_history and not has_replayable_tool_thinking
    effective_thinking_enabled = thinking_enabled and not unsafe_tool_followup
    if thinking_enabled:
        if unsafe_tool_followup:
            logger.debug(
                "DEEPSEEK_REQUEST: disabling thinking for tool follow-up without "
                "replayable thinking model={} msgs={} tools={}",
                data.get("model"),
                len(data.get("messages", [])),
                len(data.get("tools", [])),
            )
            _remove_deepseek_thinking_hints(data)
        elif has_tool_history:
            logger.debug(
                "DEEPSEEK_REQUEST: keeping thinking for tool follow-up with "
                "replayable thinking model={} msgs={} tools={}",
                data.get("model"),
                len(data.get("messages", [])),
                len(data.get("tools", [])),
            )
        elif data.get("tools") or data.get("tool_choice"):
            logger.debug(
                "DEEPSEEK_REQUEST: keeping thinking for initial tool request "
                "model={} msgs={} tools={}",
                data.get("model"),
                len(data.get("messages", [])),
                len(data.get("tools", [])),
            )

    thinking_cfg = data.pop("thinking", None)
    if effective_thinking_enabled and isinstance(thinking_cfg, dict):
        thinking_payload: dict[str, Any] = {"type": "enabled"}
        budget_tokens = thinking_cfg.get("budget_tokens")
        if isinstance(budget_tokens, int):
            thinking_payload["budget_tokens"] = budget_tokens
        data["thinking"] = thinking_payload

    if "messages" in data:
        data["messages"] = _strip_reasoning_content_when_native(
            sanitize_deepseek_messages_for_native(
                data["messages"],
                thinking_enabled=effective_thinking_enabled,
            )
        )
    if "max_tokens" not in data or data.get("max_tokens") is None:
        data["max_tokens"] = ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS

    data["stream"] = True

    logger.debug(
        "DEEPSEEK_REQUEST: build done model={} msgs={} tools={}",
        data.get("model"),
        len(data.get("messages", [])),
        len(data.get("tools", [])),
    )
    return data
