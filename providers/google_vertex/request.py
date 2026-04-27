"""Request builder for Google Vertex AI Express mode."""

from __future__ import annotations

from typing import Any

from loguru import logger

from config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from core.anthropic.native_messages_request import dump_raw_messages_request
from providers.exceptions import InvalidRequestError


_UNSUPPORTED_BLOCK_TYPES = frozenset(
    {
        "image",
        "document",
        "server_tool_use",
        "web_search_tool_result",
        "web_fetch_tool_result",
    }
)


def _convert_message_to_vertex_content(message: dict[str, Any]) -> dict[str, Any]:
    """Convert Anthropic message format to Vertex AI content format."""
    role = message.get("role", "user")
    if role == "assistant":
        role = "model"
    elif role == "system":
        role = "user"

    content = message.get("content")
    if isinstance(content, str):
        return {"role": role, "parts": [{"text": content}]}
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                block_type = block.get("type")
                if block_type == "text":
                    text = block.get("text", "")
                    if text:
                        parts.append({"text": text})
                elif block_type == "tool_use":
                    tool_input = block.get("input", {})
                    parts.append({
                        "functionCall": {
                            "name": block.get("name"),
                            "args": tool_input,
                        }
                    })
                elif block_type == "tool_result":
                    result_content = block.get("content", "")
                    if isinstance(result_content, str):
                        parts.append({"text": result_content})
        return {"role": role, "parts": parts}
    return {"role": role, "parts": [{"text": str(content)}]}


def _validate_vertex_request(data: dict[str, Any]) -> None:
    """Validate request for Vertex AI Express mode."""
    mcp = data.get("mcp_servers")
    if mcp:
        raise InvalidRequestError(
            "Vertex AI Express does not support mcp_servers on requests."
        )

    for tool in data.get("tools") or ():
        if isinstance(tool, dict):
            tool_type = tool.get("type")
            if isinstance(tool_type, str):
                if tool_type.startswith("web_search") or tool_type.startswith("web_fetch"):
                    raise InvalidRequestError(
                        "Vertex AI Express does not support listed Anthropic server tools "
                        "(web_search / web_fetch). Remove them or use a different provider."
                    )


def build_request_body(request_data: Any, *, thinking_enabled: bool) -> dict:
    """Build a Vertex AI Express JSON body."""
    logger.debug(
        "VERTEX_REQUEST: build model={} msgs={}",
        getattr(request_data, "model", "?"),
        len(getattr(request_data, "messages", [])),
    )

    data = dump_raw_messages_request(request_data)
    _validate_vertex_request(data)
    data.pop("extra_body", None)

    messages = data.pop("messages", [])
    system = data.pop("system", None)

    converted_messages = [_convert_message_to_vertex_content(m) for m in messages]
    if system:
        if isinstance(system, str):
            system_msg = {"role": "user", "parts": [{"text": system}]}
            converted_messages.insert(0, system_msg)
        elif isinstance(system, list):
            for sys_block in system:
                if isinstance(sys_block, dict) and sys_block.get("type") == "text":
                    text = sys_block.get("text", "")
                    if text:
                        system_msg = {"role": "user", "parts": [{"text": text}]}
                        converted_messages.insert(0, system_msg)

    data["contents"] = converted_messages

    generation_config: dict[str, Any] = {}
    if "max_tokens" in data and data.get("max_tokens") is not None:
        generation_config["maxOutputTokens"] = data.pop("max_tokens")
    else:
        generation_config["maxOutputTokens"] = ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS

    if "temperature" in data and data.get("temperature") is not None:
        generation_config["temperature"] = data.pop("temperature")
    if "top_p" in data and data.get("top_p") is not None:
        generation_config["topP"] = data.pop("top_p")
    if "top_k" in data and data.get("top_k") is not None:
        generation_config["topK"] = data.pop("top_k")
    if "stop_sequences" in data and data.get("stop_sequences"):
        generation_config["stopSequences"] = data.pop("stop_sequences")

    thinking_cfg = data.pop("thinking", None)
    if thinking_enabled and isinstance(thinking_cfg, dict):
        budget_tokens = thinking_cfg.get("budget_tokens")
        if isinstance(budget_tokens, int):
            generation_config["thinkingSettings"] = {"mode": "enabled", "budgetTokens": budget_tokens}

    if generation_config:
        data["generationConfig"] = generation_config

    logger.debug(
        "VERTEX_REQUEST: build done model={} contents={} tools={}",
        data.get("model"),
        len(data.get("contents", [])),
        len(data.get("tools", [])),
    )
    return data