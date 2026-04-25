"""Response helpers for the local Codex CLI provider."""

from __future__ import annotations

import json
from typing import Any


def text_from_json_event(event: dict[str, Any]) -> str:
    """Extract user-visible text from one `codex exec --json` event."""
    if event.get("type") == "item.completed":
        item = event.get("item")
        if isinstance(item, dict) and item.get("type") == "agent_message":
            text = item.get("text")
            return text if isinstance(text, str) else ""
    return ""


def parse_jsonl_text(line: str) -> str:
    """Parse a Codex JSONL line and return assistant text, if any."""
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        return ""
    if not isinstance(event, dict):
        return ""
    return text_from_json_event(event)
