"""Backward-compatible re-export. Use messaging.trees.data for new code."""

from .trees.data import MessageTree, MessageNode, MessageState

__all__ = ["MessageTree", "MessageNode", "MessageState"]
