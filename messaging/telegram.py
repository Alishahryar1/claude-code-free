"""Backward-compatible re-export. Use messaging.platforms.telegram for new code."""

from .platforms.telegram import (
    TelegramPlatform,
    TELEGRAM_AVAILABLE,
)

__all__ = ["TelegramPlatform", "TELEGRAM_AVAILABLE"]
