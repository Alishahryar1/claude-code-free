"""DeepSeek provider exports."""

from providers.defaults import DEEPSEEK_ANTHROPIC_DEFAULT_BASE, DEEPSEEK_DEFAULT_BASE

from .client import DeepSeekProvider

__all__ = [
    "DEEPSEEK_ANTHROPIC_DEFAULT_BASE",
    "DEEPSEEK_DEFAULT_BASE",
    "DeepSeekProvider",
]
