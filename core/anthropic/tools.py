"""Heuristic parser for text-emitted tool calls."""

import json
import re
import uuid
from enum import Enum
from typing import Any

from loguru import logger

_CONTROL_TOKEN_RE = re.compile(r"<\|[^|>]{1,80}\|>")
_CONTROL_TOKEN_START = "<|"
_CONTROL_TOKEN_END = "|>"


class ParserState(Enum):
    TEXT = 1
    MATCHING_FUNCTION = 2
    PARSING_PARAMETERS = 3


class HeuristicToolParser:
    """
    Stateful parser for raw text tool calls.

    Some OpenAI-compatible models emit tool calls as text rather than structured
    chunks. This parser converts the common ``● <function=...>`` form into
    Anthropic-style ``tool_use`` blocks.
    """

    _FUNC_START_PATTERN = re.compile(r"●\s*<function=([^>]+)>")
    _PARAM_PATTERN = re.compile(
        r"<parameter=([^>]+)>(.*?)(?:</parameter>|$)", re.DOTALL
    )
    _WEB_TOOL_JSON_PATTERN = re.compile(
        r"(?is)\b(?:use\s+)?(?P<tool>WebFetch|WebSearch)\b.*?(?P<json>\{.*?\})"
    )
    # Matches ● {...} where JSON has a "name" key and "arguments" or "parameters" key.
    # Covers llama-4-maverick (parameters), OpenAI-compat (arguments), any field order.
    _BULLET_JSON_PREFIX = re.compile(r"●\s*(\{)", re.DOTALL)
    # Matches <tool_call>...</tool_call> XML wrapper used by Llama 3.1+, Qwen, Mistral.
    _TOOL_CALL_XML_PATTERN = re.compile(
        r"<tool_call>\s*(?P<json>\{.*?\})\s*</tool_call>", re.DOTALL
    )

    def __init__(self):
        self._state = ParserState.TEXT
        self._buffer = ""
        self._current_tool_id = None
        self._current_function_name = None
        self._current_parameters = {}

    @staticmethod
    def _parse_tool_call_json(raw: str) -> dict[str, Any] | None:
        """
        Parse a JSON blob that may represent a tool call in any of these shapes:
          {"name": "X", "arguments": {...}}        — OpenAI / most NIM models
          {"name": "X", "parameters": {...}}       — llama-4-maverick via NIM
          {"type": "function", "name": "X", ...}  — verbose OpenAI variant
          {"function": {"name": "X", ...}}         — nested OpenAI delta form
        Returns {"name": str, "input": dict} or None if not a tool call.
        """
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if not isinstance(obj, dict):
            return None

        # Unwrap nested {"function": {...}} form
        if "function" in obj and isinstance(obj["function"], dict):
            obj = obj["function"]

        name = obj.get("name") or obj.get("function_name")
        if not name or not isinstance(name, str):
            return None

        params = obj.get("arguments") or obj.get("parameters") or obj.get("inputs") or {}
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except json.JSONDecodeError:
                params = {}
        if not isinstance(params, dict):
            params = {}

        return {"name": name, "input": params}

    def _extract_open_model_tool_calls(self) -> tuple[str, list[dict[str, Any]]]:
        """
        Detect tool calls emitted as text by open/NIM models:
          1. ● {...}  — bullet-prefixed JSON (llama-4-maverick, some NIM models)
          2. <tool_call>...</tool_call>  — XML wrapper (Llama 3.1+, Qwen, Mistral)
        """
        detected_tools: list[dict[str, Any]] = []
        spans_to_remove: list[tuple[int, int]] = []

        # --- XML wrapper form ---
        for match in self._TOOL_CALL_XML_PATTERN.finditer(self._buffer):
            parsed = self._parse_tool_call_json(match.group("json"))
            if parsed:
                detected_tools.append(
                    {
                        "type": "tool_use",
                        "id": f"toolu_heuristic_{uuid.uuid4().hex[:8]}",
                        **parsed,
                    }
                )
                spans_to_remove.append((match.start(), match.end()))
                logger.debug(
                    "Heuristic bypass: Detected <tool_call> XML tool call '{}'",
                    parsed["name"],
                )

        # --- Bullet-prefixed JSON form ---
        for match in self._BULLET_JSON_PREFIX.finditer(self._buffer):
            # Walk forward to find the matching closing brace
            start = match.start()
            brace_start = match.start(1)
            depth = 0
            end = None
            for i, ch in enumerate(self._buffer[brace_start:], brace_start):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            if end is None:
                continue  # JSON not complete yet — wait for more chunks
            raw_json = self._buffer[brace_start:end]
            parsed = self._parse_tool_call_json(raw_json)
            if parsed:
                detected_tools.append(
                    {
                        "type": "tool_use",
                        "id": f"toolu_heuristic_{uuid.uuid4().hex[:8]}",
                        **parsed,
                    }
                )
                spans_to_remove.append((start, end))
                logger.debug(
                    "Heuristic bypass: Detected ●-JSON tool call '{}'", parsed["name"]
                )

        if not detected_tools:
            return self._buffer, []

        # Remove matched spans in reverse order to preserve offsets
        remaining = self._buffer
        for s, e in sorted(spans_to_remove, reverse=True):
            remaining = remaining[:s] + remaining[e:]
        return remaining, detected_tools

    def _extract_web_tool_json_calls(self) -> tuple[str, list[dict[str, Any]]]:
        detected_tools: list[dict[str, Any]] = []

        for match in self._WEB_TOOL_JSON_PATTERN.finditer(self._buffer):
            try:
                tool_input = json.loads(match.group("json"))
            except json.JSONDecodeError:
                continue
            if not isinstance(tool_input, dict):
                continue

            tool_name = match.group("tool")
            if tool_name == "WebFetch" and "url" not in tool_input:
                continue
            if tool_name == "WebSearch" and "query" not in tool_input:
                continue

            detected_tools.append(
                {
                    "type": "tool_use",
                    "id": f"toolu_heuristic_{uuid.uuid4().hex[:8]}",
                    "name": tool_name,
                    "input": tool_input,
                }
            )
            logger.debug(
                "Heuristic bypass: Detected JSON-style tool call '{}'",
                tool_name,
            )

        if not detected_tools:
            return self._buffer, []

        return "", detected_tools

    def _strip_control_tokens(self, text: str) -> str:
        return _CONTROL_TOKEN_RE.sub("", text)

    def _split_incomplete_control_token_tail(self) -> str:
        start = self._buffer.rfind(_CONTROL_TOKEN_START)
        if start == -1:
            return ""
        end = self._buffer.find(_CONTROL_TOKEN_END, start)
        if end != -1:
            return ""

        prefix = self._buffer[:start]
        self._buffer = self._buffer[start:]
        return prefix

    def feed(self, text: str) -> tuple[str, list[dict[str, Any]]]:
        """Feed text and return safe text plus detected tool calls."""
        self._buffer += text
        self._buffer = self._strip_control_tokens(self._buffer)
        self._buffer, detected_tools = self._extract_open_model_tool_calls()
        if not detected_tools:
            self._buffer, detected_tools = self._extract_web_tool_json_calls()
        filtered_output_parts: list[str] = []

        while True:
            if self._state == ParserState.TEXT:
                if "●" in self._buffer:
                    idx = self._buffer.find("●")
                    filtered_output_parts.append(self._buffer[:idx])
                    self._buffer = self._buffer[idx:]
                    self._state = ParserState.MATCHING_FUNCTION
                else:
                    safe_prefix = self._split_incomplete_control_token_tail()
                    if safe_prefix:
                        filtered_output_parts.append(safe_prefix)
                        break

                    filtered_output_parts.append(self._buffer)
                    self._buffer = ""
                    break

            if self._state == ParserState.MATCHING_FUNCTION:
                match = self._FUNC_START_PATTERN.search(self._buffer)
                if match:
                    self._current_function_name = match.group(1).strip()
                    self._current_tool_id = f"toolu_heuristic_{uuid.uuid4().hex[:8]}"
                    self._current_parameters = {}
                    self._buffer = self._buffer[match.end() :]
                    self._state = ParserState.PARSING_PARAMETERS
                    logger.debug(
                        "Heuristic bypass: Detected start of tool call '{}'",
                        self._current_function_name,
                    )
                elif len(self._buffer) > 100:
                    filtered_output_parts.append(self._buffer[0])
                    self._buffer = self._buffer[1:]
                    self._state = ParserState.TEXT
                else:
                    break

            if self._state == ParserState.PARSING_PARAMETERS:
                finished_tool_call = False

                while True:
                    param_match = self._PARAM_PATTERN.search(self._buffer)
                    if param_match and "</parameter>" in param_match.group(0):
                        pre_match_text = self._buffer[: param_match.start()]
                        if pre_match_text:
                            filtered_output_parts.append(pre_match_text)

                        key = param_match.group(1).strip()
                        val = param_match.group(2).strip()
                        self._current_parameters[key] = val
                        self._buffer = self._buffer[param_match.end() :]
                    else:
                        break

                if "●" in self._buffer:
                    idx = self._buffer.find("●")
                    if idx > 0:
                        filtered_output_parts.append(self._buffer[:idx])
                        self._buffer = self._buffer[idx:]
                    finished_tool_call = True
                elif len(self._buffer) > 0 and not self._buffer.strip().startswith("<"):
                    if "<parameter=" not in self._buffer:
                        filtered_output_parts.append(self._buffer)
                        self._buffer = ""
                        finished_tool_call = True

                if finished_tool_call:
                    detected_tools.append(
                        {
                            "type": "tool_use",
                            "id": self._current_tool_id,
                            "name": self._current_function_name,
                            "input": self._current_parameters,
                        }
                    )
                    logger.debug(
                        "Heuristic bypass: Emitting tool call '{}' with {} params",
                        self._current_function_name,
                        len(self._current_parameters),
                    )
                    self._state = ParserState.TEXT
                else:
                    break

        return "".join(filtered_output_parts), detected_tools

    def flush(self) -> list[dict[str, Any]]:
        """Flush any remaining tool call in the buffer."""
        self._buffer = self._strip_control_tokens(self._buffer)
        detected_tools = []
        if self._state == ParserState.PARSING_PARAMETERS:
            partial_matches = re.finditer(
                r"<parameter=([^>]+)>(.*)$", self._buffer, re.DOTALL
            )
            for match in partial_matches:
                key = match.group(1).strip()
                val = match.group(2).strip()
                self._current_parameters[key] = val

            detected_tools.append(
                {
                    "type": "tool_use",
                    "id": self._current_tool_id,
                    "name": self._current_function_name,
                    "input": self._current_parameters,
                }
            )
            self._state = ParserState.TEXT
            self._buffer = ""

        return detected_tools
