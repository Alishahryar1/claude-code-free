"""Tests for Discord platform adapter."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from messaging.discord import (
    DiscordPlatform,
    _parse_allowed_channels,
    DISCORD_AVAILABLE,
)


class TestParseAllowedChannels:
    """Tests for _parse_allowed_channels helper."""

    def test_empty_string_returns_empty_set(self):
        assert _parse_allowed_channels("") == set()
        assert _parse_allowed_channels(None) == set()

    def test_single_channel(self):
        assert _parse_allowed_channels("123456789") == {"123456789"}

    def test_comma_separated(self):
        assert _parse_allowed_channels("111,222,333") == {"111", "222", "333"}

    def test_strips_whitespace(self):
        assert _parse_allowed_channels(" 111 , 222 ") == {"111", "222"}

    def test_empty_parts_ignored(self):
        assert _parse_allowed_channels("111,,222,") == {"111", "222"}


@pytest.mark.skipif(not DISCORD_AVAILABLE, reason="discord.py not installed")
class TestDiscordPlatform:
    """Tests for DiscordPlatform (requires discord.py)."""

    def test_init_with_token(self):
        platform = DiscordPlatform(
            bot_token="test_token",
            allowed_channel_ids="123,456",
        )
        assert platform.bot_token == "test_token"
        assert platform.allowed_channel_ids == {"123", "456"}

    def test_init_without_allowed_channels(self):
        with patch.dict("os.environ", {"ALLOWED_DISCORD_CHANNELS": ""}, clear=False):
            platform = DiscordPlatform(bot_token="token", allowed_channel_ids="")
        assert platform.allowed_channel_ids == set()

    def test_empty_allowed_channels_rejects_all_messages(self):
        """When allowed_channel_ids is empty, no channels are allowed (secure default)."""
        with patch.dict("os.environ", {"ALLOWED_DISCORD_CHANNELS": ""}, clear=False):
            platform = DiscordPlatform(bot_token="token", allowed_channel_ids="")
        assert platform.allowed_channel_ids == set()
        # Empty set means: not self.allowed_channel_ids is True -> reject

    def test_truncate_long_message(self):
        platform = DiscordPlatform(bot_token="token")
        long_text = "x" * 2500
        truncated = platform._truncate(long_text)
        assert len(truncated) == 2000
        assert truncated.endswith("...")

    def test_truncate_short_message_unchanged(self):
        platform = DiscordPlatform(bot_token="token")
        short = "hello"
        assert platform._truncate(short) == short

    @pytest.mark.asyncio
    async def test_send_message_returns_message_id(self):
        platform = DiscordPlatform(bot_token="token")
        mock_msg = MagicMock()
        mock_msg.id = 999
        mock_channel = AsyncMock()
        mock_channel.send = AsyncMock(return_value=mock_msg)
        platform._connected = True
        with patch.object(
            platform._client, "get_channel", MagicMock(return_value=mock_channel)
        ):
            msg_id = await platform.send_message("123", "Hello")
        assert msg_id == "999"

    @pytest.mark.asyncio
    async def test_edit_message(self):
        platform = DiscordPlatform(bot_token="token")
        mock_msg = AsyncMock()
        mock_channel = AsyncMock()
        mock_channel.fetch_message = AsyncMock(return_value=mock_msg)
        platform._connected = True
        with patch.object(
            platform._client, "get_channel", MagicMock(return_value=mock_channel)
        ):
            await platform.edit_message("123", "456", "Updated text")
        mock_msg.edit.assert_called_once_with(content="Updated text")
