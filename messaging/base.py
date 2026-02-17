"""Backward-compatible re-export. Use messaging.platforms.base for new code."""

from .platforms.base import (
    MessagingPlatform,
    SessionManagerInterface,
    CLISession,
)

__all__ = ["MessagingPlatform", "SessionManagerInterface", "CLISession"]
