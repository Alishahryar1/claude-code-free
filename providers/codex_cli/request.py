"""Build plain-text prompts for the local Codex CLI provider."""

from __future__ import annotations

import json
from typing import Any


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)

    parts: list[str] = []
    for block in content:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            parts.append(getattr(block, "text", ""))
        elif block_type == "tool_result":
            tool_id = getattr(block, "tool_use_id", "")
            result = getattr(block, "content", "")
            parts.append(f"[tool_result {tool_id}] {result}")
        elif block_type == "tool_use":
            name = getattr(block, "name", "")
            tool_input = getattr(block, "input", {})
            parts.append(
                f"[tool_use requested by assistant: {name}] "
                f"{json.dumps(tool_input, ensure_ascii=False)}"
            )
        elif block_type == "thinking":
            parts.append(
                f"[assistant thinking omitted] {getattr(block, 'thinking', '')}"
            )
        elif block_type == "redacted_thinking":
            parts.append("[redacted thinking omitted]")
        elif block_type == "image":
            parts.append("[image input omitted: codex_cli adapter supports text only]")
        else:
            parts.append(str(block))
    return "\n".join(part for part in parts if part)


def _system_to_text(system: Any) -> str:
    if system is None:
        return ""
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        return "\n".join(
            getattr(block, "text", str(block)) for block in system if block is not None
        )
    return str(system)


def build_prompt(request: Any) -> str:
    """Convert an Anthropic Messages request into a single Codex prompt string."""
    sections: list[str] = []
    system_text = _system_to_text(getattr(request, "system", None)).strip()
    if system_text:
        sections.append(f"System instructions:\n{system_text}")

    tools = getattr(request, "tools", None) or []
    if tools:
        tool_names = ", ".join(getattr(tool, "name", "unknown") for tool in tools)
        sections.append(
            "Adapter note: Claude tool-use blocks are not supported by the "
            "codex_cli provider. Respond with plain text only; do not emit "
            f"structured tool calls. Requested tool names: {tool_names}."
        )

    messages = getattr(request, "messages", [])
    transcript: list[str] = []
    for message in messages:
        role = getattr(message, "role", "user")
        text = _content_to_text(getattr(message, "content", ""))
        transcript.append(f"{role}:\n{text}")
    if transcript:
        sections.append("Conversation:\n" + "\n\n".join(transcript))

    sections.append("Assistant: respond with plain text only.")
    return "\n\n".join(sections).strip()
