"""API layer for Claude Code Proxy."""

# Apply compatibility patches as early as possible
import core.compatibility  # noqa: F401

from .app import app, create_app
from .models import (
    MessagesRequest,
    MessagesResponse,
    TokenCountRequest,
    TokenCountResponse,
)

__all__ = [
    "MessagesRequest",
    "MessagesResponse",
    "TokenCountRequest",
    "TokenCountResponse",
    "app",
    "create_app",
]
