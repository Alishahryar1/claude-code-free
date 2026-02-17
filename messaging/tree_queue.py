"""Backward-compatible re-export. Use messaging.trees.queue_manager for new code."""

from .trees.queue_manager import (
    TreeQueueManager,
    MessageTree,
    MessageNode,
    MessageState,
)

__all__ = ["TreeQueueManager", "MessageTree", "MessageNode", "MessageState"]
