"""Think tag parser for extracting reasoning content from responses."""

import re
from dataclasses import dataclass
from typing import Optional, Tuple, Iterator, Any
from enum import Enum


class ContentType(Enum):
    """Type of content chunk."""

    TEXT = "text"
    THINKING = "thinking"


@dataclass
class ContentChunk:
    """A chunk of parsed content."""

    type: ContentType
    content: str


class ThinkTagParser:
    """
    Streaming parser for <think>...</think> tags.

    Handles partial tags at chunk boundaries by buffering.
    """

    OPEN_TAG = "<think>"
    CLOSE_TAG = "</think>"
    OPEN_TAG_LEN = 7
    CLOSE_TAG_LEN = 8

    def __init__(self):
        self._buffer: str = ""
        self._in_think_tag: bool = False

    @property
    def in_think_mode(self) -> bool:
        """Whether currently inside a think tag."""
        return self._in_think_tag

    def feed(self, content: str) -> Iterator[ContentChunk]:
        """
        Feed content and yield parsed chunks.

        Handles partial tags by buffering content near potential tag boundaries.
        """
        self._buffer += content

        while self._buffer:
            if not self._in_think_tag:
                chunk = self._parse_outside_think()
                if chunk:
                    yield chunk
                else:
                    break
            else:
                chunk = self._parse_inside_think()
                if chunk:
                    yield chunk
                else:
                    break

    def _parse_outside_think(self) -> Optional[ContentChunk]:
        """Parse content outside think tags."""
        think_start = self._buffer.find(self.OPEN_TAG)

        if think_start == -1:
            # No tag found - check for partial at end
            if len(self._buffer) > self.OPEN_TAG_LEN - 1:
                # Check if buffer ends with start of a potential tag
                if self._buffer[-(self.OPEN_TAG_LEN - 1) :].startswith("<"):
                    emit = self._buffer[: -(self.OPEN_TAG_LEN - 1)]
                    self._buffer = self._buffer[-(self.OPEN_TAG_LEN - 1) :]
                else:
                    emit = self._buffer
                    self._buffer = ""
                if emit:
                    return ContentChunk(ContentType.TEXT, emit)
            return None
        else:
            # Found <think> tag
            pre_think = self._buffer[:think_start]
            self._buffer = self._buffer[think_start + self.OPEN_TAG_LEN :]
            self._in_think_tag = True
            if pre_think:
                return ContentChunk(ContentType.TEXT, pre_think)
            # Continue parsing inside think tag
            return self._parse_inside_think()

    def _parse_inside_think(self) -> Optional[ContentChunk]:
        """Parse content inside think tags."""
        think_end = self._buffer.find(self.CLOSE_TAG)

        if think_end == -1:
            # No closing tag - check for partial at end
            if len(self._buffer) > self.CLOSE_TAG_LEN - 1:
                # Check if buffer ends with start of a potential closing tag
                if self._buffer[-(self.CLOSE_TAG_LEN - 1) :].startswith("<"):
                    emit = self._buffer[: -(self.CLOSE_TAG_LEN - 1)]
                    self._buffer = self._buffer[-(self.CLOSE_TAG_LEN - 1) :]
                else:
                    emit = self._buffer
                    self._buffer = ""
                if emit:
                    return ContentChunk(ContentType.THINKING, emit)
            return None
        else:
            # Found </think> tag
            thinking_content = self._buffer[:think_end]
            self._buffer = self._buffer[think_end + self.CLOSE_TAG_LEN :]
            self._in_think_tag = False
            if thinking_content:
                return ContentChunk(ContentType.THINKING, thinking_content)
            # Continue parsing outside think tag
            return self._parse_outside_think()

    def flush(self) -> Optional[ContentChunk]:
        """Flush any remaining buffered content."""
        if self._buffer:
            chunk_type = (
                ContentType.THINKING if self._in_think_tag else ContentType.TEXT
            )
            content = self._buffer
            self._buffer = ""
            return ContentChunk(chunk_type, content)
        return None

    def reset(self):
        """Reset parser state."""
        self._buffer = ""
        self._in_think_tag = False


def extract_think_content(text: str) -> Tuple[Optional[str], str]:
    """
    Extract thinking content from text (non-streaming).

    Returns: (thinking_content, remaining_text)
    """
    think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    matches = think_pattern.findall(text)

    if matches:
        thinking = "\n".join(matches)
        remaining = think_pattern.sub("", text).strip()
        return thinking, remaining

    return None, text


def extract_reasoning_from_delta(delta: Any) -> Optional[str]:
    """
    Extract reasoning content from an OpenAI delta object.

    Checks both 'reasoning_content' and 'reasoning_details' fields.
    """
    if isinstance(delta, dict):
        reasoning = delta.get("reasoning_content")
        if reasoning:
            return reasoning

        reasoning_details = delta.get("reasoning_details")
        if reasoning_details and isinstance(reasoning_details, list):
            return "".join(
                item.get("text", "")
                for item in reasoning_details
                if isinstance(item, dict)
            )

    return None
