"""Backward-compatible re-export. Use messaging.platforms.factory for new code."""

from .platforms.factory import create_messaging_platform

__all__ = ["create_messaging_platform"]
