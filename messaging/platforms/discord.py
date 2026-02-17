"""
Discord Platform Adapter

Implements MessagingPlatform for Discord using discord.py.
"""

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Callable, Awaitable, Optional, Any, Set, cast

from loguru import logger

from .base import MessagingPlatform
from ..models import IncomingMessage
from ..rendering.discord_markdown import format_status_discord

AUDIO_EXTENSIONS = (".ogg", ".mp4", ".mp3", ".wav", ".m4a")

_discord_module: Any = None
try:
    import discord as _discord_import

    _discord_module = _discord_import
    DISCORD_AVAILABLE = True
except ImportError:
    DISCORD_AVAILABLE = False

DISCORD_MESSAGE_LIMIT = 2000


def _get_discord() -> Any:
    """Return the discord module. Raises if not available."""
    if not DISCORD_AVAILABLE or _discord_module is None:
        raise ImportError(
            "discord.py is required. Install with: pip install discord.py"
        )
    return _discord_module


def _parse_allowed_channels(raw: Optional[str]) -> Set[str]:
    """Parse comma-separated channel IDs into a set of strings."""
    if not raw or not raw.strip():
        return set()
    return {s.strip() for s in raw.split(",") if s.strip()}


if DISCORD_AVAILABLE and _discord_module is not None:
    _discord = _discord_module

    class _DiscordClient(_discord.Client):
        """Internal Discord client that forwards events to DiscordPlatform."""

        def __init__(
            self,
            platform: "DiscordPlatform",
            intents: _discord.Intents,
        ) -> None:
            super().__init__(intents=intents)
            self._platform = platform

        async def on_ready(self) -> None:
            """Called when the bot is ready."""
            self._platform._connected = True
            logger.info("Discord platform connected")

        async def on_message(self, message: Any) -> None:
            """Handle incoming Discord messages."""
            await self._platform._on_discord_message(message)
else:
    _DiscordClient = None


class DiscordPlatform(MessagingPlatform):
    """
    Discord messaging platform adapter.

    Uses discord.py for Discord access.
    Requires a Bot Token from Discord Developer Portal and message_content intent.
    """

    name = "discord"

    def __init__(
        self,
        bot_token: Optional[str] = None,
        allowed_channel_ids: Optional[str] = None,
    ):
        if not DISCORD_AVAILABLE:
            raise ImportError(
                "discord.py is required. Install with: pip install discord.py"
            )

        self.bot_token = bot_token or os.getenv("DISCORD_BOT_TOKEN")
        raw_channels = allowed_channel_ids or os.getenv("ALLOWED_DISCORD_CHANNELS")
        self.allowed_channel_ids = _parse_allowed_channels(raw_channels)

        if not self.bot_token:
            logger.warning("DISCORD_BOT_TOKEN not set")

        discord = _get_discord()
        intents = discord.Intents.default()
        intents.message_content = True

        assert _DiscordClient is not None
        self._client = _DiscordClient(self, intents)
        self._message_handler: Optional[
            Callable[[IncomingMessage], Awaitable[None]]
        ] = None
        self._connected = False
        self._limiter: Optional[Any] = None
        self._start_task: Optional[asyncio.Task] = None

    def _get_audio_attachment(self, message: Any) -> Any | None:
        """Return first audio attachment, or None."""
        for att in message.attachments:
            ct = (att.content_type or "").lower()
            fn = (att.filename or "").lower()
            if ct.startswith("audio/") or any(
                fn.endswith(ext) for ext in AUDIO_EXTENSIONS
            ):
                return att
        return None

    async def _handle_voice_note(
        self, message: Any, attachment: Any, channel_id: str
    ) -> bool:
        """Handle voice/audio attachment. Returns True if handled."""
        from config.settings import get_settings

        settings = get_settings()
        if not settings.voice_note_enabled:
            await message.reply("Voice notes are disabled.")
            return True

        if not self._message_handler:
            return False

        status_msg_id = await self.queue_send_message(
            channel_id,
            format_status_discord("Transcribing voice note..."),
            reply_to=str(message.id),
            fire_and_forget=False,
        )

        user_id = str(message.author.id)
        message_id = str(message.id)
        reply_to = (
            str(message.reference.message_id)
            if message.reference and message.reference.message_id
            else None
        )

        ext = ".ogg"
        fn = (attachment.filename or "").lower()
        for e in AUDIO_EXTENSIONS:
            if fn.endswith(e):
                ext = e
                break
        ct = attachment.content_type or "audio/ogg"
        if "mp4" in ct or "m4a" in fn:
            ext = ".m4a" if "m4a" in fn else ".mp4"
        elif "mp3" in ct or fn.endswith(".mp3"):
            ext = ".mp3"

        tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
        tmp_path = Path(tmp.name)
        tmp.close()

        try:
            await attachment.save(str(tmp_path))

            from ..transcription import transcribe_audio

            transcribed = await asyncio.to_thread(
                transcribe_audio,
                tmp_path,
                ct,
                whisper_model=settings.whisper_model,
                whisper_device=settings.whisper_device,
            )

            incoming = IncomingMessage(
                text=transcribed,
                chat_id=channel_id,
                user_id=user_id,
                message_id=message_id,
                platform="discord",
                reply_to_message_id=reply_to,
                username=message.author.display_name,
                raw_event=message,
                status_message_id=status_msg_id,
            )

            logger.info(
                "DISCORD_VOICE: chat_id=%s message_id=%s transcribed=%r",
                channel_id,
                message_id,
                (transcribed[:80] + "..." if len(transcribed) > 80 else transcribed),
            )

            await self._message_handler(incoming)
            return True
        except ValueError as e:
            await message.reply(str(e)[:200])
            return True
        except ImportError as e:
            await message.reply(str(e)[:200])
            return True
        except Exception as e:
            logger.error(f"Voice transcription failed: {e}")
            await message.reply(
                "Could not transcribe voice note. Please try again or send text."
            )
            return True
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass

    async def _on_discord_message(self, message: Any) -> None:
        """Handle incoming Discord messages."""
        if message.author.bot:
            return

        channel_id = str(message.channel.id)

        if not self.allowed_channel_ids or channel_id not in self.allowed_channel_ids:
            return

        # Handle voice/audio attachments when message has no text content
        if not message.content:
            audio_att = self._get_audio_attachment(message)
            if audio_att:
                await self._handle_voice_note(message, audio_att, channel_id)
                return
            return

        user_id = str(message.author.id)
        message_id = str(message.id)
        reply_to = (
            str(message.reference.message_id)
            if message.reference and message.reference.message_id
            else None
        )

        text_preview = (message.content or "")[:80]
        if len(message.content or "") > 80:
            text_preview += "..."
        logger.info(
            "DISCORD_MSG: chat_id=%s message_id=%s reply_to=%s text_preview=%r",
            channel_id,
            message_id,
            reply_to,
            text_preview,
        )

        if not self._message_handler:
            return

        incoming = IncomingMessage(
            text=message.content,
            chat_id=channel_id,
            user_id=user_id,
            message_id=message_id,
            platform="discord",
            reply_to_message_id=reply_to,
            username=message.author.display_name,
            raw_event=message,
        )

        try:
            await self._message_handler(incoming)
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            try:
                await self.send_message(
                    channel_id,
                    format_status_discord("Error:", str(e)[:200]),
                    reply_to=message_id,
                )
            except Exception:
                pass

    def _truncate(self, text: str, limit: int = DISCORD_MESSAGE_LIMIT) -> str:
        """Truncate text to Discord's message limit."""
        if len(text) <= limit:
            return text
        return text[: limit - 3] + "..."

    async def start(self) -> None:
        """Initialize and connect to Discord."""
        if not self.bot_token:
            raise ValueError("DISCORD_BOT_TOKEN is required")

        from ..limiter import MessagingRateLimiter

        self._limiter = await MessagingRateLimiter.get_instance()

        self._start_task = asyncio.create_task(
            self._client.start(self.bot_token),
            name="discord-client-start",
        )

        max_wait = 30
        waited = 0
        while not self._connected and waited < max_wait:
            await asyncio.sleep(0.5)
            waited += 0.5

        if not self._connected:
            raise RuntimeError("Discord client failed to connect within timeout")

        logger.info("Discord platform started")

    async def stop(self) -> None:
        """Stop the bot."""
        if self._client.is_closed():
            self._connected = False
            return

        await self._client.close()
        if self._start_task and not self._start_task.done():
            try:
                await asyncio.wait_for(self._start_task, timeout=5.0)
            except asyncio.TimeoutError, asyncio.CancelledError:
                self._start_task.cancel()
                try:
                    await self._start_task
                except asyncio.CancelledError:
                    pass

        self._connected = False
        logger.info("Discord platform stopped")

    async def send_message(
        self,
        chat_id: str,
        text: str,
        reply_to: Optional[str] = None,
        parse_mode: Optional[str] = None,
    ) -> str:
        """Send a message to a channel."""
        channel = self._client.get_channel(int(chat_id))
        if not channel or not hasattr(channel, "send"):
            raise RuntimeError(f"Channel {chat_id} not found")

        text = self._truncate(text)
        channel = cast(Any, channel)

        discord = _get_discord()
        if reply_to:
            ref = discord.MessageReference(
                message_id=int(reply_to),
                channel_id=int(chat_id),
            )
            msg = await channel.send(content=text, reference=ref)
        else:
            msg = await channel.send(content=text)

        return str(msg.id)

    async def edit_message(
        self,
        chat_id: str,
        message_id: str,
        text: str,
        parse_mode: Optional[str] = None,
    ) -> None:
        """Edit an existing message."""
        channel = self._client.get_channel(int(chat_id))
        if not channel or not hasattr(channel, "fetch_message"):
            raise RuntimeError(f"Channel {chat_id} not found")

        discord = _get_discord()
        channel = cast(Any, channel)
        try:
            msg = await channel.fetch_message(int(message_id))
        except discord.NotFound:
            return

        text = self._truncate(text)
        await msg.edit(content=text)

    async def delete_message(
        self,
        chat_id: str,
        message_id: str,
    ) -> None:
        """Delete a message from a channel."""
        channel = self._client.get_channel(int(chat_id))
        if not channel or not hasattr(channel, "fetch_message"):
            return

        discord = _get_discord()
        channel = cast(Any, channel)
        try:
            msg = await channel.fetch_message(int(message_id))
            await msg.delete()
        except discord.NotFound, discord.Forbidden:
            pass

    async def delete_messages(self, chat_id: str, message_ids: list[str]) -> None:
        """Delete multiple messages (best-effort)."""
        for mid in message_ids:
            await self.delete_message(chat_id, mid)

    async def queue_send_message(
        self,
        chat_id: str,
        text: str,
        reply_to: Optional[str] = None,
        parse_mode: Optional[str] = None,
        fire_and_forget: bool = True,
    ) -> Optional[str]:
        """Enqueue a message to be sent."""
        if not self._limiter:
            return await self.send_message(chat_id, text, reply_to, parse_mode)

        async def _send():
            return await self.send_message(chat_id, text, reply_to, parse_mode)

        if fire_and_forget:
            self._limiter.fire_and_forget(_send)
            return None
        return await self._limiter.enqueue(_send)

    async def queue_edit_message(
        self,
        chat_id: str,
        message_id: str,
        text: str,
        parse_mode: Optional[str] = None,
        fire_and_forget: bool = True,
    ) -> None:
        """Enqueue a message edit."""
        if not self._limiter:
            await self.edit_message(chat_id, message_id, text, parse_mode)
            return

        async def _edit():
            await self.edit_message(chat_id, message_id, text, parse_mode)

        dedup_key = f"edit:{chat_id}:{message_id}"
        if fire_and_forget:
            self._limiter.fire_and_forget(_edit, dedup_key=dedup_key)
        else:
            await self._limiter.enqueue(_edit, dedup_key=dedup_key)

    async def queue_delete_message(
        self,
        chat_id: str,
        message_id: str,
        fire_and_forget: bool = True,
    ) -> None:
        """Enqueue a message delete."""
        if not self._limiter:
            await self.delete_message(chat_id, message_id)
            return

        async def _delete():
            await self.delete_message(chat_id, message_id)

        dedup_key = f"del:{chat_id}:{message_id}"
        if fire_and_forget:
            self._limiter.fire_and_forget(_delete, dedup_key=dedup_key)
        else:
            await self._limiter.enqueue(_delete, dedup_key=dedup_key)

    async def queue_delete_messages(
        self,
        chat_id: str,
        message_ids: list[str],
        fire_and_forget: bool = True,
    ) -> None:
        """Enqueue a bulk delete."""
        if not message_ids:
            return

        if not self._limiter:
            await self.delete_messages(chat_id, message_ids)
            return

        async def _bulk():
            await self.delete_messages(chat_id, message_ids)

        dedup_key = f"del_bulk:{chat_id}:{hash(tuple(message_ids))}"
        if fire_and_forget:
            self._limiter.fire_and_forget(_bulk, dedup_key=dedup_key)
        else:
            await self._limiter.enqueue(_bulk, dedup_key=dedup_key)

    def fire_and_forget(self, task: Awaitable[Any]) -> None:
        """Execute a coroutine without awaiting it."""
        if asyncio.iscoroutine(task):
            asyncio.create_task(task)
        else:
            asyncio.ensure_future(task)

    def on_message(
        self,
        handler: Callable[[IncomingMessage], Awaitable[None]],
    ) -> None:
        """Register a message handler callback."""
        self._message_handler = handler

    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._connected
