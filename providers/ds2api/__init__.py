"""DS2API provider - local DeepSeek via Docker (OpenAI-compatible)."""

from providers.defaults import DS2API_DEFAULT_BASE

from .client import DS2APIProvider

__all__ = [
    "DS2API_DEFAULT_BASE",
    "DS2APIProvider",
]
