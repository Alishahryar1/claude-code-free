"""Messaging platform adapters (Telegram, Discord, etc.)."""

from .base import MessagingPlatform, SessionManagerInterface, CLISession
from .factory import create_messaging_platform

__all__ = [
    "MessagingPlatform",
    "SessionManagerInterface",
    "CLISession",
    "create_messaging_platform",
]
