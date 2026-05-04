"""Simple context management - trim messages to configured limits."""

from config.settings import Settings
from core.anthropic.tokens import get_token_count


def _has_tool_use(message: object) -> bool:
    """Return True if an assistant message contains any tool_use blocks."""
    content = getattr(message, "content", None)
    if not isinstance(content, list):
        return False
    return any(
        (getattr(b, "type", None) or (b if not hasattr(b, "type") else None)) == "tool_use"
        or (isinstance(b, dict) and b.get("type") == "tool_use")
        for b in content
    )


def _has_tool_result(message: object) -> bool:
    """Return True if a user message contains any tool_result blocks."""
    content = getattr(message, "content", None)
    if not isinstance(content, list):
        return False
    return any(
        (getattr(b, "type", None) or (b if not hasattr(b, "type") else None)) == "tool_result"
        or (isinstance(b, dict) and b.get("type") == "tool_result")
        for b in content
    )


def _sanitize_seam(messages: list) -> list:
    """Drop leading messages until the conversation starts at a clean boundary.

    After trimming, the head of the list may be an assistant message whose
    tool_use blocks have no matching tool_result in the following user message
    (the result was in a removed section), or a user message that consists
    entirely of orphaned tool_results.  Either triggers a 400 from NIM/OpenAI.

    Strategy: drop from the front until:
    1. The first message is a user message.
    2. That user message does not consist solely of tool_result blocks
       (i.e., it is a genuine human turn, not a dangling tool response).
    """
    while messages:
        first = messages[0]
        role = getattr(first, "role", None) or (first.get("role") if isinstance(first, dict) else None)
        if role == "user" and not _has_tool_result(first):
            break
        messages = messages[1:]
    return messages


class ContextManager:
    """Manages conversation context with simple trimming."""

    def __init__(
        self,
        max_messages: int | None = None,
        max_tokens: int | None = None,
        min_messages: int = 20,
    ):
        """
        Initialize context manager.

        Args:
            max_messages: Maximum number of messages to keep (None = unlimited)
            max_tokens: Maximum token budget (None = unlimited)
            min_messages: Minimum messages to keep after trimming (prevents
                losing all context when thinking blocks inflate token counts)
        """
        self.max_messages = max_messages
        self.max_tokens = max_tokens
        self.min_messages = min_messages

    def trim_messages(
        self, messages: list, system: str | list | None = None, tools: list | None = None
    ) -> tuple[list, bool]:
        """
        Trim messages to fit within configured limits.

        Args:
            messages: List of message dicts
            system: Optional system prompt
            tools: Optional list of tool definitions

        Returns:
            (trimmed_messages, was_trimmed)
        """
        original_count = len(messages)
        trimmed = messages

        # Apply message count limit
        if self.max_messages and self.max_messages > 0:
            if len(trimmed) > self.max_messages:
                # Keep first few + last messages to maintain conversation flow
                keep_start = max(2, self.max_messages // 5)
                keep_end = self.max_messages - keep_start

                trimmed = trimmed[:keep_start] + trimmed[-keep_end:]

        # Apply token limit if set
        if self.max_tokens and self.max_tokens > 0:
            tokens = get_token_count(trimmed, system, tools)
            if tokens > self.max_tokens:
                # Reduce by removing oldest message pairs, but never drop below
                # min_messages — thinking blocks can inflate counts dramatically
                # and stripping everything leaves the model with no context.
                floor = max(4, self.min_messages)
                while len(trimmed) > floor and tokens > self.max_tokens * 0.9:
                    trimmed = trimmed[2:]  # Remove oldest user/assistant pair
                    tokens = get_token_count(trimmed, system, tools)

        # Sanitize the head of the trimmed list: drop any leading messages that
        # would form an invalid conversation structure (orphaned tool_use /
        # tool_result at the cut boundary), which causes a 400 from providers.
        trimmed = _sanitize_seam(list(trimmed))

        return trimmed, len(trimmed) < original_count


def get_context_manager(settings: Settings) -> ContextManager:
    """Create a ContextManager from settings."""
    return ContextManager(
        max_messages=settings.max_messages if settings.max_messages > 0 else None,
        max_tokens=settings.context_max_tokens if settings.context_max_tokens > 0 else None,
        min_messages=settings.context_min_messages,
    )
