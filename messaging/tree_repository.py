"""Backward-compatible re-export. Use messaging.trees.repository for new code."""

from .trees.repository import TreeRepository

__all__ = ["TreeRepository"]
