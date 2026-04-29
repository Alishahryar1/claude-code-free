"""OpenRouter provider - Anthropic-compatible native transport."""

from providers.defaults import OPENROUTER_DEFAULT_BASE

from .client import OpenRouterProvider

__all__ = ["OPENROUTER_DEFAULT_BASE", "OpenRouterProvider"]
