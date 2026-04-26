"""Re-exports default upstream base URLs from the config provider catalog."""

from config.provider_catalog import (
    DEEPSEEK_DEFAULT_BASE,
    LLAMACPP_DEFAULT_BASE,
    LMSTUDIO_DEFAULT_BASE,
    NVIDIA_NIM_DEFAULT_BASE,
    OLLAMA_DEFAULT_BASE,
    OPENROUTER_DEFAULT_BASE,
)

__all__ = (
    "DEEPSEEK_DEFAULT_BASE",
    "LLAMACPP_DEFAULT_BASE",
    "LMSTUDIO_DEFAULT_BASE",
    "NVIDIA_NIM_DEFAULT_BASE",
    "OLLAMA_DEFAULT_BASE",
    "OPENROUTER_DEFAULT_BASE",
)
