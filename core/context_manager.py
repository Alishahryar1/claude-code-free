"""Simple context management - trim messages to configured limits."""

from config.settings import Settings
from core.anthropic.tokens import get_token_count


class ContextManager:
    """Manages conversation context with simple trimming."""

    def __init__(self, max_messages: int | None = None, max_tokens: int | None = None):
        """
        Initialize context manager.

        Args:
            max_messages: Maximum number of messages to keep (None = unlimited)
            max_tokens: Maximum token budget (None = unlimited)
        """
        self.max_messages = max_messages
        self.max_tokens = max_tokens

    def trim_messages(
        self, messages: list, system: str | list | None = None
    ) -> tuple[list, bool]:
        """
        Trim messages to fit within configured limits.

        Args:
            messages: List of message dicts
            system: Optional system prompt

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
            tokens = get_token_count(trimmed, system)
            if tokens > self.max_tokens:
                # Reduce by removing oldest message pairs
                while len(trimmed) > 4 and tokens > self.max_tokens * 0.9:
                    trimmed = trimmed[2:]  # Remove oldest user/assistant pair
                    tokens = get_token_count(trimmed, system)

        return trimmed, len(trimmed) < original_count


def get_context_manager(settings: Settings) -> ContextManager:
    """Create a ContextManager from settings."""
    return ContextManager(
        max_messages=settings.max_messages if settings.max_messages > 0 else None,
        max_tokens=settings.context_max_tokens if settings.context_max_tokens > 0 else None,
    )
