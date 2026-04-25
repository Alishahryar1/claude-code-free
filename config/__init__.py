"""Configuration management."""

# Apply compatibility patches as early as possible
import core.compatibility  # noqa: F401

from .settings import Settings, get_settings

__all__ = ["Settings", "get_settings"]
