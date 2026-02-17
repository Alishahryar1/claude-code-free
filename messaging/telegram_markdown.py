"""Backward-compatible re-export. Use messaging.rendering.telegram_markdown for new code."""

from .rendering.telegram_markdown import (
    escape_md_v2,
    escape_md_v2_code,
    escape_md_v2_link_url,
    mdv2_bold,
    mdv2_code_inline,
    format_status,
    render_markdown_to_mdv2,
)

__all__ = [
    "escape_md_v2",
    "escape_md_v2_code",
    "escape_md_v2_link_url",
    "mdv2_bold",
    "mdv2_code_inline",
    "format_status",
    "render_markdown_to_mdv2",
]
