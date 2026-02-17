"""Backward-compatible re-export. Use messaging.platforms.discord for new code."""

from .platforms.discord import (
    DiscordPlatform,
    DISCORD_AVAILABLE,
    DISCORD_MESSAGE_LIMIT,
)

__all__ = ["DiscordPlatform", "DISCORD_AVAILABLE", "DISCORD_MESSAGE_LIMIT"]
