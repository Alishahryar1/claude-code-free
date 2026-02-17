"""Markdown rendering utilities for messaging platforms."""

from .discord_markdown import (
    escape_discord,
    escape_discord_code,
    discord_bold,
    discord_code_inline,
    format_status as format_status_discord_fn,
    format_status_discord,
    render_markdown_to_discord,
)
from .telegram_markdown import (
    escape_md_v2,
    escape_md_v2_code,
    escape_md_v2_link_url,
    mdv2_bold,
    mdv2_code_inline,
    format_status as format_status_telegram_fn,
    render_markdown_to_mdv2,
)

__all__ = [
    "escape_discord",
    "escape_discord_code",
    "discord_bold",
    "discord_code_inline",
    "format_status_discord_fn",
    "format_status_discord",
    "render_markdown_to_discord",
    "escape_md_v2",
    "escape_md_v2_code",
    "escape_md_v2_link_url",
    "mdv2_bold",
    "mdv2_code_inline",
    "format_status_telegram_fn",
    "render_markdown_to_mdv2",
]
