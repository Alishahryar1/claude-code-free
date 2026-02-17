"""Backward-compatible re-export. Use messaging.trees.processor for new code."""

from .trees.processor import TreeQueueProcessor

__all__ = ["TreeQueueProcessor"]
