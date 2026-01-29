"""Logging utilities for compact, traceable request logging.

Provides fingerprinting and summary functions to reduce log file sizes
while maintaining full traceability through request IDs and content hashes.
"""

import hashlib
import json
import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# Separate debug file handler for full payloads
_debug_handler: Optional[RotatingFileHandler] = None


def _get_debug_handler() -> RotatingFileHandler:
    """Get or create the rotating debug file handler."""
    global _debug_handler
    if _debug_handler is None:
        _debug_handler = RotatingFileHandler(
            "server_debug.jsonl",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=3,
            encoding="utf-8",
        )
        _debug_handler.setLevel(logging.DEBUG)
    return _debug_handler


def generate_request_fingerprint(messages: List[Any]) -> str:
    """Generate unique short hash for message content.

    Creates a SHA256 hash of all message content, returning an 8-char prefix
    that's sufficient for correlation without full content logging.
    """
    content_parts = []
    for msg in messages:
        if hasattr(msg, "content"):
            content = msg.content
            if isinstance(content, str):
                content_parts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if hasattr(block, "text"):
                        content_parts.append(block.text)
                    elif hasattr(block, "type"):
                        content_parts.append(f"<{block.type}>")
        elif hasattr(msg, "role"):
            content_parts.append(msg.role)

    combined = "|".join(content_parts)
    hash_digest = hashlib.sha256(combined.encode("utf-8")).hexdigest()
    return f"fp_{hash_digest[:8]}"


def get_last_user_message_preview(messages: List[Any], max_len: int = 100) -> str:
    """Extract a preview of the last user message."""
    for msg in reversed(messages):
        if hasattr(msg, "role") and msg.role == "user":
            content = msg.content
            if isinstance(content, str):
                preview = content.replace("\n", " ").replace("\r", "")
                return preview[:max_len] + "..." if len(preview) > max_len else preview
            elif isinstance(content, list):
                text_parts = []
                for block in content:
                    if hasattr(block, "text"):
                        text_parts.append(block.text)
                if text_parts:
                    preview = " ".join(text_parts).replace("\n", " ")
                    return (
                        preview[:max_len] + "..." if len(preview) > max_len else preview
                    )
    return "(no user message)"


def get_tool_names(tools: Optional[List[Any]], max_count: int = 5) -> List[str]:
    """Extract tool names from tool list, limiting to max_count."""
    if not tools:
        return []
    names = []
    for tool in tools[:max_count]:
        if hasattr(tool, "name"):
            names.append(tool.name)
        elif isinstance(tool, dict) and "name" in tool:
            names.append(tool["name"])
    if len(tools) > max_count:
        names.append(f"+{len(tools) - max_count} more")
    return names


def build_request_summary(request_data: Any) -> Dict[str, Any]:
    """Build compact metadata dict for logging.

    Returns a dictionary with key metrics about the request without
    including the full content.
    """
    messages = getattr(request_data, "messages", [])
    tools = getattr(request_data, "tools", None)
    system = getattr(request_data, "system", None)
    thinking = getattr(request_data, "thinking", None)

    # Count message types
    user_count = sum(1 for m in messages if getattr(m, "role", None) == "user")
    assistant_count = sum(
        1 for m in messages if getattr(m, "role", None) == "assistant"
    )

    return {
        "fingerprint": generate_request_fingerprint(messages),
        "model": getattr(request_data, "model", "unknown"),
        "message_count": len(messages),
        "user_msgs": user_count,
        "assistant_msgs": assistant_count,
        "user_preview": get_last_user_message_preview(messages),
        "tool_count": len(tools) if tools else 0,
        "tool_names": get_tool_names(tools),
        "has_thinking": bool(thinking and getattr(thinking, "enabled", False)),
        "has_system": bool(system),
        "max_tokens": getattr(request_data, "max_tokens", 0),
    }


def log_full_payload(request_id: str, payload: Dict[str, Any]) -> None:
    """Write full payload to separate debug file for forensic analysis.

    Only writes if LOG_FULL_PAYLOADS env var is set to 'true'.
    """
    if os.getenv("LOG_FULL_PAYLOADS", "false").lower() != "true":
        return

    try:
        handler = _get_debug_handler()
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": request_id,
            "payload": payload,
        }
        handler.stream.write(json.dumps(record, default=str) + "\n")
        handler.stream.flush()
    except Exception as e:
        logger.warning(f"Failed to write debug payload: {e}")


def log_request_compact(
    logger_instance: logging.Logger,
    request_id: str,
    request_data: Any,
    prefix: str = "API_REQUEST",
) -> None:
    """Log a compact request summary with fingerprint for correlation.

    This is the main entry point for logging requests. It logs a single-line
    JSON summary to the main log and optionally writes full payload to debug file.
    """
    summary = build_request_summary(request_data)
    summary["request_id"] = request_id

    logger_instance.info(f"{prefix}: {json.dumps(summary)}")

    # Optionally write full payload to debug file
    if os.getenv("LOG_FULL_PAYLOADS", "false").lower() == "true":
        try:
            payload = (
                request_data.model_dump() if hasattr(request_data, "model_dump") else {}
            )
            log_full_payload(request_id, payload)
        except Exception as e:
            logger.debug(f"Could not dump request data: {e}")
