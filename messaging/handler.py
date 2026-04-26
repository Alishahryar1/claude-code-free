"""
Claude Message Handler

Platform-agnostic Claude interaction logic.
Handles the core workflow of processing user messages via Claude CLI.
Uses tree-based queuing for message ordering.
"""

import asyncio

from loguru import logger

from core.anthropic import format_user_error_preview, get_user_facing_error_message

from .cli_event_constants import STATUS_MESSAGE_PREFIXES
from .command_dispatcher import (
    dispatch_command,
    message_kind_for_command,
    parse_command_base,
)
from .event_parser import parse_cli_event
from .models import IncomingMessage
from .node_event_pipeline import handle_session_info_event, process_parsed_cli_event
from .platforms.base import MessagingPlatform, SessionManagerInterface
from .rendering.profiles import build_rendering_profile
from .safe_diagnostics import format_exception_for_log
from .session import SessionStore
from .transcript import RenderCtx, TranscriptBuffer
from .trees.queue_manager import (
    MessageNode,
    MessageState,
    MessageTree,
    TreeQueueManager,
)
from .ui_updates import ThrottledTranscriptEditor


class ClaudeMessageHandler:
    """
    Platform-agnostic handler for Claude interactions.

    Uses a tree-based message queue where:
    - New messages create a tree root
    - Replies become children of the message being replied to
    - Each node has state: PENDING, IN_PROGRESS, COMPLETED, ERROR
    - Per-tree queue ensures ordered processing
    """

    def __init__(
        self,
        platform: MessagingPlatform,
        cli_manager: SessionManagerInterface,
        session_store: SessionStore,
        *,
        debug_platform_edits: bool = False,
        debug_subagent_stack: bool = False,
        log_raw_messaging_content: bool = False,
        log_raw_cli_diagnostics: bool = False,
        log_messaging_error_details: bool = False,
    ):
        self.platform = platform
        self.cli_manager = cli_manager
        self.session_store = session_store
        self._debug_platform_edits = debug_platform_edits
        self._debug_subagent_stack = debug_subagent_stack
        self._log_raw_messaging_content = log_raw_messaging_content
        self._log_raw_cli_diagnostics = log_raw_cli_diagnostics
        self._log_messaging_error_details = log_messaging_error_details
        self._tree_queue = TreeQueueManager(
            queue_update_callback=self.update_queue_positions,
            node_started_callback=self.mark_node_processing,
        )
        self._rendering_profile = build_rendering_profile(platform.name)

    def format_status(self, emoji: str, label: str, suffix: str | None = None) -> str:
        return self._rendering_profile.format_status(emoji, label, suffix)

    def _parse_mode(self) -> str | None:
        return self._rendering_profile.parse_mode

    def get_render_ctx(self) -> RenderCtx:
        return self._rendering_profile.render_ctx

    def _get_limit_chars(self) -> int:
        return self._rendering_profile.limit_chars

    @property
    def tree_queue(self) -> TreeQueueManager:
        """Accessor for the current tree queue manager."""
        return self._tree_queue

    def replace_tree_queue(self, tree_queue: TreeQueueManager) -> None:
        """Replace tree queue manager via explicit API."""
        self._tree_queue = tree_queue
        self._tree_queue.set_queue_update_callback(self.update_queue_positions)
        self._tree_queue.set_node_started_callback(self.mark_node_processing)

    async def handle_message(self, incoming: IncomingMessage) -> None:
        """
        Main entry point for handling an incoming message.

        Determines if this is a new conversation or reply,
        creates/extends the message tree, and queues for processing.
        """
        raw = incoming.text or ""
        if self._log_raw_messaging_content:
            text_preview = raw[:80]
            if len(raw) > 80:
                text_preview += "..."
            logger.info(
                "HANDLER_ENTRY: chat_id={} message_id={} reply_to={} text_preview={!r}",
                incoming.chat_id,
                incoming.message_id,
                incoming.reply_to_message_id,
                text_preview,
            )
        else:
            logger.info(
                "HANDLER_ENTRY: chat_id={} message_id={} reply_to={} text_len={}",
                incoming.chat_id,
                incoming.message_id,
                incoming.reply_to_message_id,
                len(raw),
            )

        with logger.contextualize(
            chat_id=incoming.chat_id, node_id=incoming.message_id
        ):
            await self._handle_message_impl(incoming)

    async def _handle_message_impl(self, incoming: IncomingMessage) -> None:
        """Implementation of handle_message with context bound."""
        cmd_base = parse_command_base(incoming.text)

        # Record incoming message ID for best-effort UI clearing (/clear), even if
        # we later ignore this message (status/command/etc).
        try:
            if incoming.message_id is not None:
                self.session_store.record_message_id(
                    incoming.platform,
                    incoming.chat_id,
                    str(incoming.message_id),
                    direction="in",
                    kind=message_kind_for_command(cmd_base),
                )
        except Exception as e:
            logger.debug(
                "Failed to record incoming message_id: {}",
                format_exception_for_log(
                    e, log_full_message=self._log_messaging_error_details
                ),
            )

        if await dispatch_command(self, incoming, cmd_base):
            return

        # Filter out status messages (our own messages)
        text = incoming.text or ""
        if any(text.startswith(p) for p in STATUS_MESSAGE_PREFIXES):
            return

        # Check if this is a reply to an existing node in a tree
        parent_node_id = None
        tree = None

        if incoming.is_reply() and incoming.reply_to_message_id:
            # Look up if the replied-to message is in any tree (could be a node or status message)
            reply_id = incoming.reply_to_message_id
            tree = self.tree_queue.get_tree_for_node(reply_id)
            if tree:
                # Resolve to actual node ID (handles status message replies)
                parent_node_id = self.tree_queue.resolve_parent_node_id(reply_id)
                if parent_node_id:
                    logger.info(f"Found tree for reply, parent node: {parent_node_id}")
                else:
                    logger.warning(
                        f"Reply to {incoming.reply_to_message_id} found tree but no valid parent node"
                    )
                    tree = None  # Treat as new conversation

        # Generate node ID
        node_id = incoming.message_id

        # Use pre-sent status (e.g. voice note) or send new
        status_text = self._get_initial_status(tree, parent_node_id)
        if incoming.status_message_id:
            status_msg_id = incoming.status_message_id
            await self.platform.queue_edit_message(
                incoming.chat_id,
                status_msg_id,
                status_text,
                parse_mode=self._parse_mode(),
                fire_and_forget=False,
            )
        else:
            status_msg_id = await self.platform.queue_send_message(
                incoming.chat_id,
                status_text,
                reply_to=incoming.message_id,
                fire_and_forget=False,
                message_thread_id=incoming.message_thread_id,
            )
        self.record_outgoing_message(
            incoming.platform, incoming.chat_id, status_msg_id, "status"
        )

        # Create or extend tree
        if parent_node_id and tree and status_msg_id:
            # Reply to existing node - add as child
            tree, _node = await self.tree_queue.add_to_tree(
                parent_node_id=parent_node_id,
                node_id=node_id,
                incoming=incoming,
                status_message_id=status_msg_id,
            )
            # Register status message as a node too for reply chains
            self.tree_queue.register_node(status_msg_id, tree.root_id)
            self.session_store.register_node(status_msg_id, tree.root_id)
            self.session_store.register_node(node_id, tree.root_id)
        elif status_msg_id:
            # New conversation - create new tree
            tree = await self.tree_queue.create_tree(
                node_id=node_id,
                incoming=incoming,
                status_message_id=status_msg_id,
            )
            # Register status message
            self.tree_queue.register_node(status_msg_id, tree.root_id)
            self.session_store.register_node(node_id, tree.root_id)
            self.session_store.register_node(status_msg_id, tree.root_id)

        # Persist tree
        if tree:
            self.session_store.save_tree(tree.root_id, tree.to_dict())

        # Enqueue for processing
        was_queued = await self.tree_queue.enqueue(
            node_id=node_id,
            processor=self._process_node,
        )

        if was_queued and status_msg_id:
            # Update status to show queue position
            queue_size = self.tree_queue.get_queue_size(node_id)
            await self.platform.queue_edit_message(
                incoming.chat_id,
                status_msg_id,
                self.format_status(
                    "📋", "Queued", f"(position {queue_size}) - waiting..."
                ),
                parse_mode=self._parse_mode(),
            )

    async def update_queue_positions(self, tree: MessageTree) -> None:
        """Refresh queued status messages after a dequeue."""
        try:
            queued_ids = await tree.get_queue_snapshot()
        except Exception as e:
            logger.warning(
                "Failed to read queue snapshot: {}",
                format_exception_for_log(
                    e, log_full_message=self._log_messaging_error_details
                ),
            )
            return

        if not queued_ids:
            return

        position = 0
        for node_id in queued_ids:
            node = tree.get_node(node_id)
            if not node or node.state != MessageState.PENDING:
                continue
            position += 1
            self.platform.fire_and_forget(
                self.platform.queue_edit_message(
                    node.incoming.chat_id,
                    node.status_message_id,
                    self.format_status(
                        "📋", "Queued", f"(position {position}) - waiting..."
                    ),
                    parse_mode=self._parse_mode(),
                )
            )

    async def mark_node_processing(self, tree: MessageTree, node_id: str) -> None:
        """Update the dequeued node's status to processing immediately."""
        node = tree.get_node(node_id)
        if not node or node.state == MessageState.ERROR:
            return
        self.platform.fire_and_forget(
            self.platform.queue_edit_message(
                node.incoming.chat_id,
                node.status_message_id,
                self.format_status("🔄", "Processing..."),
                parse_mode=self._parse_mode(),
            )
        )

    def _create_transcript_and_render_ctx(
        self,
    ) -> tuple[TranscriptBuffer, RenderCtx]:
        """Create transcript buffer and render context for node processing."""
        transcript = TranscriptBuffer(
            show_tool_results=False,
            debug_subagent_stack=self._debug_subagent_stack,
        )
        return transcript, self.get_render_ctx()

    async def _process_node(
        self,
        node_id: str,
        node: MessageNode,
    ) -> None:
        """Core task processor - handles a single Claude CLI interaction."""
        incoming = node.incoming
        status_msg_id = node.status_message_id
        chat_id = incoming.chat_id

        with logger.contextualize(node_id=node_id, chat_id=chat_id):
            await self._process_node_impl(node_id, node, chat_id, status_msg_id)

    async def _process_node_impl(
        self,
        node_id: str,
        node: MessageNode,
        chat_id: str,
        status_msg_id: str,
    ) -> None:
        """Internal implementation of _process_node with context bound."""
        incoming = node.incoming

        tree = self.tree_queue.get_tree_for_node(node_id)
        if tree:
            await tree.update_state(node_id, MessageState.IN_PROGRESS)

        transcript, render_ctx = self._create_transcript_and_render_ctx()

        had_transcript_events = False
        captured_session_id = None
        temp_session_id = None
        last_status: str | None = None

        parent_session_id = None
        if tree and node.parent_id:
            parent_session_id = tree.get_parent_session_id(node_id)
            if parent_session_id:
                logger.info(f"Will fork from parent session: {parent_session_id}")

        editor = ThrottledTranscriptEditor(
            platform=self.platform,
            parse_mode=self._parse_mode(),
            get_limit_chars=self._get_limit_chars,
            transcript=transcript,
            render_ctx=render_ctx,
            node_id=node_id,
            chat_id=chat_id,
            status_msg_id=status_msg_id,
            debug_platform_edits=self._debug_platform_edits,
            log_messaging_error_details=self._log_messaging_error_details,
        )

        async def update_ui(status: str | None = None, force: bool = False) -> None:
            await editor.update(status, force=force)

        try:
            try:
                (
                    cli_session,
                    session_or_temp_id,
                    is_new,
                ) = await self.cli_manager.get_or_create_session(session_id=None)
                if is_new:
                    temp_session_id = session_or_temp_id
                else:
                    captured_session_id = session_or_temp_id
            except RuntimeError as e:
                error_message = get_user_facing_error_message(e)
                transcript.apply({"type": "error", "message": error_message})
                await update_ui(
                    self.format_status("⏳", "Session limit reached"),
                    force=True,
                )
                if tree:
                    await tree.update_state(
                        node_id,
                        MessageState.ERROR,
                        error_message=error_message,
                    )
                return

            logger.info(f"HANDLER: Starting CLI task processing for node {node_id}")
            event_count = 0
            async for event_data in cli_session.start_task(
                incoming.text,
                session_id=parent_session_id,
                fork_session=bool(parent_session_id),
            ):
                if not isinstance(event_data, dict):
                    logger.warning(
                        f"HANDLER: Non-dict event received: {type(event_data)}"
                    )
                    continue
                event_count += 1
                if event_count % 10 == 0:
                    logger.debug(f"HANDLER: Processed {event_count} events so far")

                (
                    captured_session_id,
                    temp_session_id,
                ) = await handle_session_info_event(
                    event_data,
                    tree,
                    node_id,
                    captured_session_id,
                    temp_session_id,
                    cli_manager=self.cli_manager,
                    session_store=self.session_store,
                )
                if event_data.get("type") == "session_info":
                    continue

                parsed_list = parse_cli_event(
                    event_data, log_raw_cli=self._log_raw_cli_diagnostics
                )
                logger.debug(f"HANDLER: Parsed {len(parsed_list)} events from CLI")

                for parsed in parsed_list:
                    (
                        last_status,
                        had_transcript_events,
                    ) = await process_parsed_cli_event(
                        parsed,
                        transcript,
                        update_ui,
                        last_status,
                        had_transcript_events,
                        tree,
                        node_id,
                        captured_session_id,
                        session_store=self.session_store,
                        format_status=self.format_status,
                        propagate_error_to_children=self._propagate_error_to_children,
                        log_messaging_error_details=self._log_messaging_error_details,
                    )

        except asyncio.CancelledError:
            logger.warning(f"HANDLER: Task cancelled for node {node_id}")
            cancel_reason = None
            if isinstance(node.context, dict):
                cancel_reason = node.context.get("cancel_reason")

            if cancel_reason == "stop":
                await update_ui(self.format_status("⏹", "Stopped."), force=True)
            else:
                transcript.apply({"type": "error", "message": "Task was cancelled"})
                await update_ui(self.format_status("❌", "Cancelled"), force=True)

            # Do not propagate cancellation to children; a reply-scoped "/stop"
            # should only stop the targeted task.
            if tree:
                await tree.update_state(
                    node_id, MessageState.ERROR, error_message="Cancelled by user"
                )
        except Exception as e:
            logger.error(
                "HANDLER: Task failed with exception: {}",
                format_exception_for_log(
                    e, log_full_message=self._log_messaging_error_details
                ),
            )
            error_msg = format_user_error_preview(e)
            transcript.apply({"type": "error", "message": error_msg})
            await update_ui(self.format_status("💥", "Task Failed"), force=True)
            if tree:
                await self._propagate_error_to_children(
                    node_id, error_msg, "Parent task failed"
                )
        finally:
            logger.info(f"HANDLER: _process_node completed for node {node_id}")
            # Free the session-manager slot. Session IDs are persisted in the tree and
            # can be resumed later by ID; we don't need to keep a CLISession instance
            # around after this node completes.
            try:
                if captured_session_id:
                    await self.cli_manager.remove_session(captured_session_id)
                elif temp_session_id:
                    await self.cli_manager.remove_session(temp_session_id)
            except Exception as e:
                logger.debug(
                    "Failed to remove session for node {}: {}",
                    node_id,
                    format_exception_for_log(
                        e, log_full_message=self._log_messaging_error_details
                    ),
                )

    async def _propagate_error_to_children(
        self,
        node_id: str,
        error_msg: str,
        child_status_text: str,
    ) -> None:
        """Mark node as error and propagate to pending children with UI updates."""
        affected = await self.tree_queue.mark_node_error(
            node_id, error_msg, propagate_to_children=True
        )
        # Update status messages for all affected children (skip first = current node)
        for child in affected[1:]:
            self.platform.fire_and_forget(
                self.platform.queue_edit_message(
                    child.incoming.chat_id,
                    child.status_message_id,
                    self.format_status("❌", "Cancelled:", child_status_text),
                    parse_mode=self._parse_mode(),
                )
            )

    def _get_initial_status(
        self,
        tree: object | None,
        parent_node_id: str | None,
    ) -> str:
        """Get initial status message text."""
        if tree and parent_node_id:
            # Reply to existing tree
            if self.tree_queue.is_node_tree_busy(parent_node_id):
                queue_size = self.tree_queue.get_queue_size(parent_node_id) + 1
                return self.format_status(
                    "📋", "Queued", f"(position {queue_size}) - waiting..."
                )
            return self.format_status("🔄", "Continuing conversation...")

        # New conversation
        return self.format_status("⏳", "Launching new Claude CLI instance...")

    async def stop_all_tasks(self) -> int:
        """
        Stop all pending and in-progress tasks.

        Order of operations:
        1. Cancel tree queue tasks (uses internal locking)
        2. Stop CLI sessions
        3. Update UI for all affected nodes
        """
        # 1. Cancel tree queue tasks using the public async method
        logger.info("Cancelling tree queue tasks...")
        cancelled_nodes = await self.tree_queue.cancel_all()
        logger.info(f"Cancelled {len(cancelled_nodes)} nodes")

        # 2. Stop CLI sessions - this kills subprocesses and ensures everything is dead
        logger.info("Stopping all CLI sessions...")
        await self.cli_manager.stop_all()

        # 3. Update UI and persist state for all cancelled nodes
        self.update_cancelled_nodes_ui(cancelled_nodes)

        return len(cancelled_nodes)

    async def stop_task(self, node_id: str) -> int:
        """
        Stop a single queued or in-progress task node.

        Used when the user replies "/stop" to a specific status/user message.
        """
        tree = self.tree_queue.get_tree_for_node(node_id)
        if tree:
            node = tree.get_node(node_id)
            if node and node.state not in (MessageState.COMPLETED, MessageState.ERROR):
                # Used by _process_node cancellation path to render "Stopped."
                node.set_context({"cancel_reason": "stop"})

        cancelled_nodes = await self.tree_queue.cancel_node(node_id)
        self.update_cancelled_nodes_ui(cancelled_nodes)
        return len(cancelled_nodes)

    def record_outgoing_message(
        self,
        platform: str,
        chat_id: str,
        msg_id: str | None,
        kind: str,
    ) -> None:
        """Record outgoing message ID for /clear. Best-effort, never raises."""
        if not msg_id:
            return
        try:
            self.session_store.record_message_id(
                platform, chat_id, str(msg_id), direction="out", kind=kind
            )
        except Exception as e:
            logger.debug(
                "Failed to record message_id: {}",
                format_exception_for_log(
                    e, log_full_message=self._log_messaging_error_details
                ),
            )

    def update_cancelled_nodes_ui(self, nodes: list[MessageNode]) -> None:
        """Update status messages and persist tree state for cancelled nodes."""
        trees_to_save: dict[str, MessageTree] = {}
        for node in nodes:
            self.platform.fire_and_forget(
                self.platform.queue_edit_message(
                    node.incoming.chat_id,
                    node.status_message_id,
                    self.format_status("⏹", "Stopped."),
                    parse_mode=self._parse_mode(),
                )
            )
            tree = self.tree_queue.get_tree_for_node(node.node_id)
            if tree:
                trees_to_save[tree.root_id] = tree
        for root_id, tree in trees_to_save.items():
            self.session_store.save_tree(root_id, tree.to_dict())
