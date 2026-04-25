"""Providers package - implement your own provider by extending BaseProvider.

Concrete adapters (e.g. ``NvidiaNimProvider``) live in subpackages; import them
from ``providers.nvidia_nim`` etc. to avoid loading every adapter when the
``providers`` package is imported.
"""

# Apply compatibility patches as early as possible
import core.compatibility  # noqa: F401

from .base import BaseProvider, ProviderConfig
from .exceptions import (
    APIError,
    AuthenticationError,
    InvalidRequestError,
    OverloadedError,
    ProviderError,
    RateLimitError,
    UnknownProviderTypeError,
)

__all__ = [
    "APIError",
    "AuthenticationError",
    "BaseProvider",
    "InvalidRequestError",
    "OverloadedError",
    "ProviderConfig",
    "ProviderError",
    "RateLimitError",
    "UnknownProviderTypeError",
]
