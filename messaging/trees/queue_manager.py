"""Tree-based message queue: index, async node processor, and public manager API."""

import asyncio
from collections.abc import Awaitable, Callable

from loguru import logger

from config.settings import get_settings
from core.anthropic import get_user_facing_error_message

from ..models import IncomingMessage
from ..safe_diagnostics import format_exception_for_log
from .data import MessageNode, MessageState, MessageTree


class TreeRepository:
    """
    In-memory index of trees and node-to-root mappings.

    Used only by :class:`TreeQueueManager`; kept as a named type for tests.
    """

    def __init__(self) -> None:
        self._trees: dict[str, MessageTree] = {}  # root_id -> tree
        self._node_to_tree: dict[str, str] = {}  # node_id -> root_id

    def get_tree(self, root_id: str) -> MessageTree | None:
        """Get a tree by its root ID."""
        return self._trees.get(root_id)

    def get_tree_for_node(self, node_id: str) -> MessageTree | None:
        """Get the tree containing a given node."""
        root_id = self._node_to_tree.get(node_id)
        if not root_id:
            return None
        return self._trees.get(root_id)

    def get_node(self, node_id: str) -> MessageNode | None:
        """Get a node from any tree."""
        tree = self.get_tree_for_node(node_id)
        return tree.get_node(node_id) if tree else None

    def add_tree(self, root_id: str, tree: MessageTree) -> None:
        """Add a new tree to the repository."""
        self._trees[root_id] = tree
        self._node_to_tree[root_id] = root_id
        logger.debug("TREE_REPO: add_tree root_id={}", root_id)

    def register_node(self, node_id: str, root_id: str) -> None:
        """Register a node ID to a tree."""
        self._node_to_tree[node_id] = root_id
        logger.debug("TREE_REPO: register_node node_id={} root_id={}", node_id, root_id)

    def has_node(self, node_id: str) -> bool:
        """Check if a node is registered in any tree."""
        return node_id in self._node_to_tree

    def tree_count(self) -> int:
        """Get the number of trees in the repository."""
        return len(self._trees)

    def is_tree_busy(self, root_id: str) -> bool:
        """Check if a tree is currently processing."""
        tree = self._trees.get(root_id)
        return tree.is_processing if tree else False

    def is_node_tree_busy(self, node_id: str) -> bool:
        """Check if the tree containing a node is busy."""
        tree = self.get_tree_for_node(node_id)
        return tree.is_processing if tree else False

    def get_queue_size(self, node_id: str) -> int:
        """Get queue size for the tree containing a node."""
        tree = self.get_tree_for_node(node_id)
        return tree.get_queue_size() if tree else 0

    def resolve_parent_node_id(self, msg_id: str) -> str | None:
        """
        Resolve a message ID to the actual parent node ID.

        Handles the case where msg_id is a status message ID
        (which maps to the tree but isn't an actual node).

        Returns:
            The node_id to use as parent, or None if not found
        """
        tree = self.get_tree_for_node(msg_id)
        if not tree:
            return None

        if tree.has_node(msg_id):
            return msg_id

        node = tree.find_node_by_status_message(msg_id)
        if node:
            return node.node_id

        return None

    def get_pending_children(self, node_id: str) -> list[MessageNode]:
        """
        Get all pending child nodes (recursively) of a given node.

        Used for error propagation - when a node fails, its pending
        children should also be marked as failed.
        """
        tree = self.get_tree_for_node(node_id)
        if not tree:
            return []

        pending: list[MessageNode] = []
        stack = [node_id]

        while stack:
            current_id = stack.pop()
            node = tree.get_node(current_id)
            if not node:
                continue
            for child_id in node.children_ids:
                child = tree.get_node(child_id)
                if child and child.state == MessageState.PENDING:
                    pending.append(child)
                    stack.append(child_id)

        return pending

    def all_trees(self) -> list[MessageTree]:
        """Get all trees in the repository."""
        return list(self._trees.values())

    def tree_ids(self) -> list[str]:
        """Get all tree root IDs."""
        return list(self._trees.keys())

    def unregister_nodes(self, node_ids: list[str]) -> None:
        """Remove node IDs from the node-to-tree mapping."""
        for nid in node_ids:
            self._node_to_tree.pop(nid, None)

    def remove_tree(self, root_id: str) -> MessageTree | None:
        """
        Remove a tree and all its node mappings from the repository.

        Returns:
            The removed tree, or None if not found.
        """
        tree = self._trees.pop(root_id, None)
        if not tree:
            return None
        for node in tree.all_nodes():
            self._node_to_tree.pop(node.node_id, None)
        logger.debug("TREE_REPO: remove_tree root_id={}", root_id)
        return tree

    def get_message_ids_for_chat(self, platform: str, chat_id: str) -> set[str]:
        """Get all message IDs (incoming + status) for a given platform/chat."""
        msg_ids: set[str] = set()
        for tree in self._trees.values():
            for node in tree.all_nodes():
                if str(node.incoming.platform) == str(platform) and str(
                    node.incoming.chat_id
                ) == str(chat_id):
                    if node.incoming.message_id is not None:
                        msg_ids.add(str(node.incoming.message_id))
                    if node.status_message_id:
                        msg_ids.add(str(node.status_message_id))
        return msg_ids

    def to_dict(self) -> dict:
        """Serialize all trees."""
        return {
            "trees": {rid: tree.to_dict() for rid, tree in self._trees.items()},
            "node_to_tree": self._node_to_tree.copy(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> TreeRepository:
        """Deserialize from dictionary."""
        repo = cls()
        for root_id, tree_data in data.get("trees", {}).items():
            repo._trees[root_id] = MessageTree.from_dict(tree_data)
        repo._node_to_tree = data.get("node_to_tree", {})
        return repo


class TreeQueueProcessor:
    """
    Per-tree async queue processing (one manager owns one processor instance).
    """

    def __init__(
        self,
        queue_update_callback: Callable[[MessageTree], Awaitable[None]] | None = None,
        node_started_callback: Callable[[MessageTree, str], Awaitable[None]]
        | None = None,
    ) -> None:
        self._queue_update_callback = queue_update_callback
        self._node_started_callback = node_started_callback

    def set_queue_update_callback(
        self,
        queue_update_callback: Callable[[MessageTree], Awaitable[None]] | None,
    ) -> None:
        """Update the callback used to refresh queue positions."""
        self._queue_update_callback = queue_update_callback

    def set_node_started_callback(
        self,
        node_started_callback: Callable[[MessageTree, str], Awaitable[None]] | None,
    ) -> None:
        """Update the callback used when a queued node starts processing."""
        self._node_started_callback = node_started_callback

    async def _notify_queue_updated(self, tree: MessageTree) -> None:
        """Invoke queue update callback if set."""
        if not self._queue_update_callback:
            return
        try:
            await self._queue_update_callback(tree)
        except Exception as e:
            d = get_settings().log_messaging_error_details
            logger.warning(
                "Queue update callback failed: {}",
                format_exception_for_log(e, log_full_message=d),
            )

    async def _notify_node_started(self, tree: MessageTree, node_id: str) -> None:
        """Invoke node started callback if set."""
        if not self._node_started_callback:
            return
        try:
            await self._node_started_callback(tree, node_id)
        except Exception as e:
            d = get_settings().log_messaging_error_details
            logger.warning(
                "Node started callback failed: {}",
                format_exception_for_log(e, log_full_message=d),
            )

    async def process_node(
        self,
        tree: MessageTree,
        node: MessageNode,
        processor: Callable[[str, MessageNode], Awaitable[None]],
    ) -> None:
        """Process a single node and then check the queue."""
        if node.state == MessageState.ERROR:
            logger.info(
                f"Skipping node {node.node_id} as it is already in state {node.state}"
            )
            await self._process_next(tree, processor)
            return

        try:
            await processor(node.node_id, node)
        except asyncio.CancelledError:
            logger.info(f"Task for node {node.node_id} was cancelled")
            raise
        except Exception as e:
            d = get_settings().log_messaging_error_details
            logger.error(
                "Error processing node {}: {}",
                node.node_id,
                format_exception_for_log(e, log_full_message=d),
            )
            await tree.update_state(
                node.node_id,
                MessageState.ERROR,
                error_message=get_user_facing_error_message(e),
            )
        finally:
            async with tree.with_lock():
                tree.clear_current_node()
            await self._process_next(tree, processor)

    async def _process_next(
        self,
        tree: MessageTree,
        processor: Callable[[str, MessageNode], Awaitable[None]],
    ) -> None:
        """Process the next message in queue, if any."""
        next_node_id = None
        async with tree.with_lock():
            next_node_id = await tree.dequeue()

            if not next_node_id:
                tree.set_processing_state(None, False)
                logger.debug(f"Tree {tree.root_id} queue empty, marking as free")
                return

            tree.set_processing_state(next_node_id, True)
            logger.info(f"Processing next queued node {next_node_id}")

            node = tree.get_node(next_node_id)
            if node:
                tree.set_current_task(
                    asyncio.create_task(self.process_node(tree, node, processor))
                )

        if next_node_id:
            await self._notify_node_started(tree, next_node_id)
            await self._notify_queue_updated(tree)

    async def enqueue_and_start(
        self,
        tree: MessageTree,
        node_id: str,
        processor: Callable[[str, MessageNode], Awaitable[None]],
    ) -> bool:
        """
        Enqueue a node or start processing immediately.

        Returns:
            True if queued, False if processing immediately
        """
        async with tree.with_lock():
            if tree.is_processing:
                tree.put_queue_unlocked(node_id)
                queue_size = tree.get_queue_size()
                logger.info(f"Queued node {node_id}, position {queue_size}")
                return True
            else:
                tree.set_processing_state(node_id, True)

                node = tree.get_node(node_id)
                if node:
                    tree.set_current_task(
                        asyncio.create_task(self.process_node(tree, node, processor))
                    )
                return False

    def cancel_current(self, tree: MessageTree) -> bool:
        """Cancel the currently running task in a tree."""
        return tree.cancel_current_task()


class TreeQueueManager:
    """
    Manages multiple message trees: index + async processing.

    Each new conversation creates a new tree.
    Replies to existing messages add nodes to existing trees.
    """

    def __init__(
        self,
        queue_update_callback: Callable[[MessageTree], Awaitable[None]] | None = None,
        node_started_callback: Callable[[MessageTree, str], Awaitable[None]]
        | None = None,
        _repository: TreeRepository | None = None,
    ) -> None:
        self._repository = _repository or TreeRepository()
        self._processor = TreeQueueProcessor(
            queue_update_callback=queue_update_callback,
            node_started_callback=node_started_callback,
        )
        self._lock = asyncio.Lock()

        logger.info("TreeQueueManager initialized")

    async def create_tree(
        self,
        node_id: str,
        incoming: IncomingMessage,
        status_message_id: str,
    ) -> MessageTree:
        """
        Create a new tree with a root node.

        Args:
            node_id: ID for the root node
            incoming: The incoming message
            status_message_id: Bot's status message ID

        Returns:
            The created MessageTree
        """
        async with self._lock:
            root_node = MessageNode(
                node_id=node_id,
                incoming=incoming,
                status_message_id=status_message_id,
                state=MessageState.PENDING,
            )

            tree = MessageTree(root_node)
            self._repository.add_tree(node_id, tree)

            logger.info(f"Created new tree with root {node_id}")
            return tree

    async def add_to_tree(
        self,
        parent_node_id: str,
        node_id: str,
        incoming: IncomingMessage,
        status_message_id: str,
    ) -> tuple[MessageTree, MessageNode]:
        """
        Add a reply as a child node to an existing tree.

        Args:
            parent_node_id: ID of the parent message
            node_id: ID for the new node
            incoming: The incoming reply message
            status_message_id: Bot's status message ID

        Returns:
            Tuple of (tree, new_node)
        """
        async with self._lock:
            if not self._repository.has_node(parent_node_id):
                raise ValueError(f"Parent node {parent_node_id} not found in any tree")

            tree = self._repository.get_tree_for_node(parent_node_id)
            if not tree:
                raise ValueError(f"Parent node {parent_node_id} not found in any tree")

        node = await tree.add_node(
            node_id=node_id,
            incoming=incoming,
            status_message_id=status_message_id,
            parent_id=parent_node_id,
        )

        async with self._lock:
            self._repository.register_node(node_id, tree.root_id)

        logger.info(f"Added node {node_id} to tree {tree.root_id}")
        return tree, node

    def get_tree(self, root_id: str) -> MessageTree | None:
        """Get a tree by its root ID."""
        return self._repository.get_tree(root_id)

    def get_tree_for_node(self, node_id: str) -> MessageTree | None:
        """Get the tree containing a given node."""
        return self._repository.get_tree_for_node(node_id)

    def get_node(self, node_id: str) -> MessageNode | None:
        """Get a node from any tree."""
        return self._repository.get_node(node_id)

    def resolve_parent_node_id(self, msg_id: str) -> str | None:
        """Resolve a message ID to the actual parent node ID."""
        return self._repository.resolve_parent_node_id(msg_id)

    def is_tree_busy(self, root_id: str) -> bool:
        """Check if a tree is currently processing."""
        return self._repository.is_tree_busy(root_id)

    def is_node_tree_busy(self, node_id: str) -> bool:
        """Check if the tree containing a node is busy."""
        return self._repository.is_node_tree_busy(node_id)

    async def enqueue(
        self,
        node_id: str,
        processor: Callable[[str, MessageNode], Awaitable[None]],
    ) -> bool:
        """
        Enqueue a node for processing.

        If the tree is not busy, processing starts immediately.
        If busy, the message is queued.

        Args:
            node_id: Node to process
            processor: Async function to process the node

        Returns:
            True if queued, False if processing immediately
        """
        tree = self._repository.get_tree_for_node(node_id)
        if not tree:
            logger.error(f"No tree found for node {node_id}")
            return False

        return await self._processor.enqueue_and_start(tree, node_id, processor)

    def get_queue_size(self, node_id: str) -> int:
        """Get queue size for the tree containing a node."""
        return self._repository.get_queue_size(node_id)

    def get_pending_children(self, node_id: str) -> list[MessageNode]:
        """Get all pending child nodes (recursively) of a given node."""
        return self._repository.get_pending_children(node_id)

    async def mark_node_error(
        self,
        node_id: str,
        error_message: str,
        propagate_to_children: bool = True,
    ) -> list[MessageNode]:
        """
        Mark a node as ERROR and optionally propagate to pending children.

        Args:
            node_id: The node to mark as error
            error_message: Error description
            propagate_to_children: If True, also mark pending children as error

        Returns:
            List of all nodes marked as error (including children)
        """
        tree = self._repository.get_tree_for_node(node_id)
        if not tree:
            return []

        affected = []
        node = tree.get_node(node_id)
        if node:
            await tree.update_state(
                node_id, MessageState.ERROR, error_message=error_message
            )
            affected.append(node)

        if propagate_to_children:
            pending_children = self._repository.get_pending_children(node_id)
            for child in pending_children:
                await tree.update_state(
                    child.node_id,
                    MessageState.ERROR,
                    error_message=f"Parent failed: {error_message}",
                )
                affected.append(child)

        return affected

    async def cancel_tree(self, root_id: str) -> list[MessageNode]:
        """
        Cancel all queued and in-progress messages in a tree.

        Updates node states to ERROR and returns list of affected nodes
        that were actually active or in the current processing queue.
        """
        tree = self._repository.get_tree(root_id)
        if not tree:
            return []

        cancelled_nodes = []

        cleanup_count = 0
        async with tree.with_lock():
            if tree.cancel_current_task():
                current_id = tree.current_node_id
                if current_id:
                    node = tree.get_node(current_id)
                    if node and node.state not in (
                        MessageState.COMPLETED,
                        MessageState.ERROR,
                    ):
                        tree.set_node_error_sync(node, "Cancelled by user")
                        cancelled_nodes.append(node)

            queue_nodes = tree.drain_queue_and_mark_cancelled()
            cancelled_nodes.extend(queue_nodes)
            cancelled_ids = {n.node_id for n in cancelled_nodes}

            for node in tree.all_nodes():
                if (
                    node.state in (MessageState.PENDING, MessageState.IN_PROGRESS)
                    and node.node_id not in cancelled_ids
                ):
                    tree.set_node_error_sync(node, "Stale task cleaned up")
                    cleanup_count += 1

            tree.reset_processing_state()

        if cancelled_nodes:
            logger.info(
                f"Cancelled {len(cancelled_nodes)} active nodes in tree {root_id}"
            )
        if cleanup_count:
            logger.info(f"Cleaned up {cleanup_count} stale nodes in tree {root_id}")

        return cancelled_nodes

    async def cancel_node(self, node_id: str) -> list[MessageNode]:
        """
        Cancel a single node (queued or in-progress) without affecting other nodes.

        Returns:
            List containing the cancelled node if it was cancellable, else empty list.
        """
        tree = self._repository.get_tree_for_node(node_id)
        if not tree:
            return []

        async with tree.with_lock():
            node = tree.get_node(node_id)
            if not node:
                return []

            if node.state in (MessageState.COMPLETED, MessageState.ERROR):
                return []

            if tree.is_current_node(node_id):
                self._processor.cancel_current(tree)

            try:
                tree.remove_from_queue(node_id)
            except Exception:
                logger.debug(
                    "Failed to remove node from queue; will rely on state=ERROR"
                )

            tree.set_node_error_sync(node, "Cancelled by user")

            return [node]

    async def cancel_all(self) -> list[MessageNode]:
        """Cancel all messages in all trees."""
        async with self._lock:
            root_ids = list(self._repository.tree_ids())
            all_cancelled: list[MessageNode] = []
            for root_id in root_ids:
                all_cancelled.extend(await self.cancel_tree(root_id))
            return all_cancelled

    def cleanup_stale_nodes(self) -> int:
        """
        Mark any PENDING or IN_PROGRESS nodes in all trees as ERROR.
        Used on startup to reconcile restored state.
        """
        count = 0
        for tree in self._repository.all_trees():
            for node in tree.all_nodes():
                if node.state in (MessageState.PENDING, MessageState.IN_PROGRESS):
                    tree.set_node_error_sync(node, "Lost during server restart")
                    count += 1
        if count:
            logger.info(f"Cleaned up {count} stale nodes during startup")
        return count

    def get_tree_count(self) -> int:
        """Get the number of active message trees."""
        return self._repository.tree_count()

    def set_queue_update_callback(
        self,
        queue_update_callback: Callable[[MessageTree], Awaitable[None]] | None,
    ) -> None:
        """Set callback for queue position updates."""
        self._processor.set_queue_update_callback(queue_update_callback)

    def set_node_started_callback(
        self,
        node_started_callback: Callable[[MessageTree, str], Awaitable[None]] | None,
    ) -> None:
        """Set callback for when a queued node starts processing."""
        self._processor.set_node_started_callback(node_started_callback)

    def register_node(self, node_id: str, root_id: str) -> None:
        """Register a node ID to a tree (for external mapping)."""
        self._repository.register_node(node_id, root_id)

    async def cancel_branch(self, branch_root_id: str) -> list[MessageNode]:
        """
        Cancel all PENDING/IN_PROGRESS nodes in the subtree (branch_root + descendants).
        """
        tree = self._repository.get_tree_for_node(branch_root_id)
        if not tree:
            return []

        branch_ids = set(tree.get_descendants(branch_root_id))
        cancelled: list[MessageNode] = []

        async with tree.with_lock():
            for nid in branch_ids:
                node = tree.get_node(nid)
                if not node or node.state in (
                    MessageState.COMPLETED,
                    MessageState.ERROR,
                ):
                    continue

                if tree.is_current_node(nid):
                    self._processor.cancel_current(tree)
                    tree.set_node_error_sync(node, "Cancelled by user")
                    cancelled.append(node)
                else:
                    tree.remove_from_queue(nid)
                    tree.set_node_error_sync(node, "Cancelled by user")
                    cancelled.append(node)

        if cancelled:
            logger.info(f"Cancelled {len(cancelled)} nodes in branch {branch_root_id}")
        return cancelled

    async def remove_branch(
        self, branch_root_id: str
    ) -> tuple[list[MessageNode], str, bool]:
        """
        Remove a branch (subtree) from the tree.

        If branch_root is the tree root, removes the entire tree.

        Returns:
            (removed_nodes, root_id, removed_entire_tree)
        """
        tree = self._repository.get_tree_for_node(branch_root_id)
        if not tree:
            return ([], "", False)

        root_id = tree.root_id

        if branch_root_id == root_id:
            cancelled = await self.cancel_tree(root_id)
            removed_tree = self._repository.remove_tree(root_id)
            if removed_tree:
                return (removed_tree.all_nodes(), root_id, True)
            return (cancelled, root_id, True)

        async with tree.with_lock():
            removed = tree.remove_branch(branch_root_id)

        self._repository.unregister_nodes([n.node_id for n in removed])
        return (removed, root_id, False)

    def get_message_ids_for_chat(self, platform: str, chat_id: str) -> set[str]:
        """Get all message IDs for a given platform/chat."""
        return self._repository.get_message_ids_for_chat(platform, chat_id)

    def to_dict(self) -> dict:
        """Serialize all trees."""
        return self._repository.to_dict()

    @classmethod
    def from_dict(
        cls,
        data: dict,
        queue_update_callback: Callable[[MessageTree], Awaitable[None]] | None = None,
        node_started_callback: Callable[[MessageTree, str], Awaitable[None]]
        | None = None,
    ) -> TreeQueueManager:
        """Deserialize from dictionary."""
        return cls(
            queue_update_callback=queue_update_callback,
            node_started_callback=node_started_callback,
            _repository=TreeRepository.from_dict(data),
        )


__all__ = [
    "TreeQueueManager",
    "TreeQueueProcessor",
    "TreeRepository",
]
