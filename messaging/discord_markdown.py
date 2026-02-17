"""Backward-compatible re-export. Use messaging.rendering.discord_markdown for new code."""

from .rendering.discord_markdown import (
    escape_discord,
    escape_discord_code,
    discord_bold,
    discord_code_inline,
    format_status,
    format_status_discord,
    render_markdown_to_discord,
)

__all__ = [
    "escape_discord",
    "escape_discord_code",
    "discord_bold",
    "discord_code_inline",
    "format_status",
    "format_status_discord",
    "render_markdown_to_discord",
]
