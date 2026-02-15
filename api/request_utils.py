"""Request utility functions for API route handlers.

Contains token counting and re-exports detection/command utilities.
"""

import json
import logging
from typing import List, Optional, Union

import tiktoken

from .models.anthropic import MessagesRequest
from .detection import (
    is_quota_check_request,
    is_title_generation_request,
    is_prefix_detection_request,
    is_suggestion_mode_request,
    is_filepath_extraction_request,
)
from .command_utils import extract_command_prefix, extract_filepaths_from_command

logger = logging.getLogger(__name__)
ENCODER = tiktoken.get_encoding("cl100k_base")

__all__ = [
    "is_quota_check_request",
    "is_title_generation_request",
    "is_prefix_detection_request",
    "is_suggestion_mode_request",
    "is_filepath_extraction_request",
    "extract_command_prefix",
    "extract_filepaths_from_command",
    "get_token_count",
]


def get_token_count(
    messages: List,
    system: Optional[Union[str, List]] = None,
    tools: Optional[List] = None,
) -> int:
    """Estimate token count for a request.

    Uses tiktoken cl100k_base encoding to estimate token usage.
    Includes system prompt, messages, tools, and per-message overhead.
    """
    total_tokens = 0

    if system:
        if isinstance(system, str):
            total_tokens += len(ENCODER.encode(system))
        elif isinstance(system, list):
            for block in system:
                if hasattr(block, "text"):
                    total_tokens += len(ENCODER.encode(block.text))

    for msg in messages:
        if isinstance(msg.content, str):
            total_tokens += len(ENCODER.encode(msg.content))
        elif isinstance(msg.content, list):
            for block in msg.content:
                b_type = getattr(block, "type", None)

                if b_type == "text":
                    total_tokens += len(ENCODER.encode(getattr(block, "text", "")))
                elif b_type == "thinking":
                    total_tokens += len(ENCODER.encode(getattr(block, "thinking", "")))
                elif b_type == "tool_use":
                    name = getattr(block, "name", "")
                    inp = getattr(block, "input", {})
                    total_tokens += len(ENCODER.encode(name))
                    total_tokens += len(ENCODER.encode(json.dumps(inp)))
                    total_tokens += 10
                elif b_type == "tool_result":
                    content = getattr(block, "content", "")
                    if isinstance(content, str):
                        total_tokens += len(ENCODER.encode(content))
                    else:
                        total_tokens += len(ENCODER.encode(json.dumps(content)))
                    total_tokens += 5

    if tools:
        for tool in tools:
            tool_str = (
                tool.name + (tool.description or "") + json.dumps(tool.input_schema)
            )
            total_tokens += len(ENCODER.encode(tool_str))

    total_tokens += len(messages) * 3
    if tools:
        total_tokens += len(tools) * 5

    return max(1, total_tokens)
