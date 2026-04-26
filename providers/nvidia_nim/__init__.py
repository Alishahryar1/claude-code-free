"""NVIDIA NIM provider package."""

from providers.defaults import NVIDIA_NIM_DEFAULT_BASE

from .client import NvidiaNimProvider

__all__ = ["NVIDIA_NIM_DEFAULT_BASE", "NvidiaNimProvider"]
