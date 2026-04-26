"""Message and tool format converters."""

import json
from typing import Any

from pydantic import BaseModel

from .content import get_block_attr, get_block_type
from .utils import set_if_not_none


class OpenAIConversionError(Exception):
    """Raised when Anthropic content cannot be converted to OpenAI chat without data loss."""


def _openai_reject_native_only_top_level_fields(request_data: Any) -> None:
    """OpenAI chat providers may only convert known top-level request fields.

    First-class model fields (e.g. ``context_management``) are not forwarded to
    the OpenAI API but are allowed so clients do not hit spurious 400s.
    Unknown extra keys (``__pydantic_extra__``) are still rejected.
    """
    if not isinstance(request_data, BaseModel):
        return
    extra = getattr(request_data, "__pydantic_extra__", None)
    if not extra:
        return
    raise OpenAIConversionError(
        "OpenAI chat conversion does not support these top-level request fields: "
        f"{sorted(str(k) for k in extra)}. Use a native Anthropic transport provider."
    )


def _tool_name(tool: Any) -> str:
    return str(getattr(tool, "name", "") or "")


def _tool_input_schema(tool: Any) -> dict[str, Any]:
    schema = getattr(tool, "input_schema", None)
    if isinstance(schema, dict):
        return schema
    return {"type": "object", "properties": {}}


def _serialize_tool_result_content(tool_content: Any) -> str:
    """Serialize tool_result content for OpenAI ``role: tool`` messages (stable JSON for structured values)."""
    if tool_content is None:
        return ""
    if isinstance(tool_content, str):
        return tool_content
    if isinstance(tool_content, dict):
        return json.dumps(tool_content, ensure_ascii=False)
    if isinstance(tool_content, list):
        parts: list[str] = []
        for item in tool_content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif isinstance(item, dict):
                parts.append(json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(tool_content)


class AnthropicToOpenAIConverter:
    """Convert Anthropic message format to OpenAI-compatible format."""

    @staticmethod
    def convert_messages(
        messages: list[Any],
        *,
        include_thinking: bool = True,
        include_reasoning_content: bool = False,
    ) -> list[dict[str, Any]]:
        result = []

        for msg in messages:
            role = msg.role
            content = msg.content

            if isinstance(content, str):
                result.append({"role": role, "content": content})
            elif isinstance(content, list):
                if role == "assistant":
                    result.extend(
                        AnthropicToOpenAIConverter._convert_assistant_message(
                            content,
                            include_thinking=include_thinking,
                            include_reasoning_content=include_reasoning_content,
                        )
                    )
                elif role == "user":
                    result.extend(
                        AnthropicToOpenAIConverter._convert_user_message(content)
                    )
            else:
                result.append({"role": role, "content": str(content)})

        return result

    @staticmethod
    def _convert_assistant_message(
        content: list[Any],
        *,
        include_thinking: bool = True,
        include_reasoning_content: bool = False,
    ) -> list[dict[str, Any]]:
        content_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        seen_tool_use = False

        for block in content:
            block_type = get_block_type(block)

            if block_type == "text":
                if seen_tool_use:
                    raise OpenAIConversionError(
                        "OpenAI chat conversion does not support assistant text after "
                        "tool_use in the same message; split the transcript or use a "
                        "native Anthropic provider."
                    )
                content_parts.append(get_block_attr(block, "text", ""))
            elif block_type == "thinking":
                if not include_thinking:
                    continue
                if seen_tool_use:
                    raise OpenAIConversionError(
                        "OpenAI chat conversion does not support assistant thinking after "
                        "tool_use in the same message; split the transcript or use a "
                        "native Anthropic provider."
                    )
                thinking = get_block_attr(block, "thinking", "")
                content_parts.append(f"<think>\n{thinking}\n</think>")
                if include_reasoning_content:
                    thinking_parts.append(thinking)
            elif block_type == "redacted_thinking":
                # Opaque provider continuation data; do not materialize as model-visible text
                # or reasoning_content for OpenAI chat upstreams.
                continue
            elif block_type == "tool_use":
                seen_tool_use = True
                tool_input = get_block_attr(block, "input", {})
                tool_calls.append(
                    {
                        "id": get_block_attr(block, "id"),
                        "type": "function",
                        "function": {
                            "name": get_block_attr(block, "name"),
                            "arguments": json.dumps(tool_input)
                            if isinstance(tool_input, dict)
                            else str(tool_input),
                        },
                    }
                )
            elif block_type == "image":
                raise OpenAIConversionError(
                    "Assistant image blocks are not supported for OpenAI chat conversion."
                )
            elif block_type in (
                "server_tool_use",
                "web_search_tool_result",
                "web_fetch_tool_result",
            ):
                raise OpenAIConversionError(
                    "OpenAI chat conversion does not support Anthropic server tool blocks "
                    f"({block_type!r} in an assistant message). Use a native Anthropic transport provider."
                )

        content_str = "\n\n".join(content_parts)
        if not content_str and not tool_calls:
            content_str = " "

        msg: dict[str, Any] = {
            "role": "assistant",
            "content": content_str,
        }
        if tool_calls:
            msg["tool_calls"] = tool_calls
        if include_reasoning_content and thinking_parts:
            msg["reasoning_content"] = "\n".join(thinking_parts)

        return [msg]

    @staticmethod
    def _convert_user_message(content: list[Any]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        text_parts: list[str] = []

        def flush_text() -> None:
            if text_parts:
                result.append({"role": "user", "content": "\n".join(text_parts)})
                text_parts.clear()

        for block in content:
            block_type = get_block_type(block)

            if block_type == "text":
                text_parts.append(get_block_attr(block, "text", ""))
            elif block_type == "image":
                raise OpenAIConversionError(
                    "User message image blocks are not supported for OpenAI chat "
                    "conversion; use a vision-capable native Anthropic provider or "
                    "extend the converter."
                )
            elif block_type == "tool_result":
                flush_text()
                tool_content = get_block_attr(block, "content", "")
                serialized = _serialize_tool_result_content(tool_content)
                result.append(
                    {
                        "role": "tool",
                        "tool_call_id": get_block_attr(block, "tool_use_id"),
                        "content": serialized if serialized else "",
                    }
                )

        flush_text()
        return result

    @staticmethod
    def convert_tools(tools: list[Any]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": _tool_input_schema(tool),
                },
            }
            for tool in tools
        ]

    @staticmethod
    def convert_tool_choice(tool_choice: Any) -> Any:
        if not isinstance(tool_choice, dict):
            return tool_choice

        choice_type = tool_choice.get("type")
        if choice_type == "tool":
            name = tool_choice.get("name")
            if name:
                return {"type": "function", "function": {"name": name}}
        if choice_type == "any":
            return "required"
        if choice_type in {"auto", "none", "required"}:
            return choice_type
        if choice_type == "function" and isinstance(tool_choice.get("function"), dict):
            return tool_choice

        return tool_choice

    @staticmethod
    def convert_system_prompt(system: Any) -> dict[str, str] | None:
        if isinstance(system, str):
            return {"role": "system", "content": system}
        if isinstance(system, list):
            text_parts = [
                get_block_attr(block, "text", "")
                for block in system
                if get_block_type(block) == "text"
            ]
            if text_parts:
                return {"role": "system", "content": "\n\n".join(text_parts).strip()}
        return None


def build_base_request_body(
    request_data: Any,
    *,
    default_max_tokens: int | None = None,
    include_thinking: bool = True,
    include_reasoning_content: bool = False,
) -> dict[str, Any]:
    """Build the common parts of an OpenAI-format request body."""
    _openai_reject_native_only_top_level_fields(request_data)
    messages = AnthropicToOpenAIConverter.convert_messages(
        request_data.messages,
        include_thinking=include_thinking,
        include_reasoning_content=include_reasoning_content,
    )

    system = getattr(request_data, "system", None)
    if system:
        system_msg = AnthropicToOpenAIConverter.convert_system_prompt(system)
        if system_msg:
            messages.insert(0, system_msg)

    body: dict[str, Any] = {"model": request_data.model, "messages": messages}

    max_tokens = getattr(request_data, "max_tokens", None)
    set_if_not_none(body, "max_tokens", max_tokens or default_max_tokens)
    set_if_not_none(body, "temperature", getattr(request_data, "temperature", None))
    set_if_not_none(body, "top_p", getattr(request_data, "top_p", None))

    stop_sequences = getattr(request_data, "stop_sequences", None)
    if stop_sequences:
        body["stop"] = stop_sequences

    tools = getattr(request_data, "tools", None)
    if tools:
        body["tools"] = AnthropicToOpenAIConverter.convert_tools(tools)
    tool_choice = getattr(request_data, "tool_choice", None)
    if tool_choice:
        body["tool_choice"] = AnthropicToOpenAIConverter.convert_tool_choice(
            tool_choice
        )

    return body
