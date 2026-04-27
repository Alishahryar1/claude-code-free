from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from loguru import logger

from messaging.platforms.telegram import (
    TelegramPlatform,
    _parse_allowed_user_ids,
)


@pytest.fixture
def telegram_platform():
    with patch("messaging.platforms.telegram.TELEGRAM_AVAILABLE", True):
        platform = TelegramPlatform(bot_token="test_token", allowed_user_id="12345")
        return platform


def test_telegram_platform_init_no_token():
    with patch.dict("os.environ", {}, clear=True):
        platform = TelegramPlatform(bot_token=None)
        assert platform.bot_token is None


@pytest.mark.asyncio
async def test_telegram_platform_start_success(telegram_platform):
    with patch("telegram.ext.Application.builder") as mock_builder:
        mock_app = MagicMock()
        mock_app.initialize = AsyncMock()
        mock_app.start = AsyncMock()
        mock_app.updater.start_polling = AsyncMock()

        mock_builder.return_value.token.return_value.request.return_value.build.return_value = mock_app

        # Mock MessagingRateLimiter
        with patch("messaging.limiter.MessagingRateLimiter.get_instance", AsyncMock()):
            await telegram_platform.start()

            assert telegram_platform._connected is True
            mock_app.initialize.assert_called_once()
            mock_app.start.assert_called_once()


@pytest.mark.asyncio
async def test_telegram_platform_send_message_success(telegram_platform):
    mock_bot = AsyncMock()
    mock_msg = MagicMock()
    mock_msg.message_id = 999
    mock_bot.send_message.return_value = mock_msg

    telegram_platform._application = MagicMock()
    telegram_platform._application.bot = mock_bot

    msg_id = await telegram_platform.send_message("chat_1", "hello")

    assert msg_id == "999"
    mock_bot.send_message.assert_called_once_with(
        chat_id="chat_1",
        text="hello",
        reply_to_message_id=None,
        parse_mode="MarkdownV2",
    )


@pytest.mark.asyncio
async def test_telegram_platform_edit_message_success(telegram_platform):
    mock_bot = AsyncMock()
    telegram_platform._application = MagicMock()
    telegram_platform._application.bot = mock_bot

    await telegram_platform.edit_message("chat_1", "999", "new text")

    mock_bot.edit_message_text.assert_called_once_with(
        chat_id="chat_1", message_id=999, text="new text", parse_mode="MarkdownV2"
    )


@pytest.mark.asyncio
async def test_telegram_platform_queue_send_message(telegram_platform):
    mock_limiter = AsyncMock()
    telegram_platform._limiter = mock_limiter

    await telegram_platform.queue_send_message("chat_1", "hello", fire_and_forget=False)

    mock_limiter.enqueue.assert_called_once()


@pytest.mark.asyncio
async def test_on_telegram_message_authorized(telegram_platform):
    handler = AsyncMock()
    telegram_platform.on_message(handler)

    mock_update = MagicMock()
    mock_update.message.text = "hello"
    mock_update.message.message_id = 1
    mock_update.effective_user.id = 12345
    mock_update.effective_chat.id = 6789
    mock_update.message.reply_to_message = None

    await telegram_platform._on_telegram_message(mock_update, MagicMock())

    handler.assert_called_once()
    incoming = handler.call_args[0][0]
    assert incoming.text == "hello"
    assert incoming.user_id == "12345"


@pytest.mark.asyncio
async def test_on_telegram_message_unauthorized(telegram_platform):
    handler = AsyncMock()
    telegram_platform.on_message(handler)

    mock_update = MagicMock()
    mock_update.message.text = "hello"
    mock_update.effective_user.id = 99999  # Unauthorized

    await telegram_platform._on_telegram_message(mock_update, MagicMock())

    handler.assert_not_called()


# ---------------------------------------------------------------------------
# Multi-user allow-list (issue #178)
# ---------------------------------------------------------------------------


def test_parse_allowed_user_ids_none():
    assert _parse_allowed_user_ids(None) == set()


def test_parse_allowed_user_ids_empty_string():
    assert _parse_allowed_user_ids("") == set()


def test_parse_allowed_user_ids_whitespace_only():
    assert _parse_allowed_user_ids("   ") == set()


def test_parse_allowed_user_ids_single():
    assert _parse_allowed_user_ids("12345") == {"12345"}


def test_parse_allowed_user_ids_single_with_padding():
    assert _parse_allowed_user_ids("  12345  ") == {"12345"}


def test_parse_allowed_user_ids_multiple():
    assert _parse_allowed_user_ids("123,456,789") == {"123", "456", "789"}


def test_parse_allowed_user_ids_strips_inner_whitespace():
    assert _parse_allowed_user_ids("123, 456 , 789") == {"123", "456", "789"}


def test_parse_allowed_user_ids_drops_empty_segments():
    # Trailing comma / double commas / lone whitespace segments are noise,
    # not a wildcard — drop them rather than turning them into "" entries
    # that would never match a real user id but would still leave the set
    # truthy.
    assert _parse_allowed_user_ids("123,,456,") == {"123", "456"}
    assert _parse_allowed_user_ids(",  , 123 , ,") == {"123"}


def test_parse_allowed_user_ids_dedups():
    assert _parse_allowed_user_ids("123,123,456") == {"123", "456"}


def test_telegram_platform_init_parses_multi():
    with patch("messaging.platforms.telegram.TELEGRAM_AVAILABLE", True):
        platform = TelegramPlatform(
            bot_token="test_token", allowed_user_id="123,456,789"
        )
    assert platform.allowed_user_ids == {"123", "456", "789"}


def test_telegram_platform_init_parses_single_back_compat():
    with patch("messaging.platforms.telegram.TELEGRAM_AVAILABLE", True):
        platform = TelegramPlatform(bot_token="test_token", allowed_user_id="12345")
    assert platform.allowed_user_ids == {"12345"}


def test_telegram_platform_init_no_allow_list():
    with patch("messaging.platforms.telegram.TELEGRAM_AVAILABLE", True):
        platform = TelegramPlatform(bot_token="test_token", allowed_user_id=None)
    assert platform.allowed_user_ids == set()


@pytest.fixture
def telegram_platform_multi():
    with patch("messaging.platforms.telegram.TELEGRAM_AVAILABLE", True):
        return TelegramPlatform(bot_token="test_token", allowed_user_id="111,222,333")


@pytest.mark.asyncio
async def test_on_telegram_message_each_authorized_id_passes(
    telegram_platform_multi,
):
    handler = AsyncMock()
    telegram_platform_multi.on_message(handler)

    for uid in (111, 222, 333):
        handler.reset_mock()
        mock_update = MagicMock()
        mock_update.message.text = "hello"
        mock_update.message.message_id = 1
        mock_update.effective_user.id = uid
        mock_update.effective_chat.id = 6789
        mock_update.message.reply_to_message = None

        await telegram_platform_multi._on_telegram_message(mock_update, MagicMock())

        handler.assert_called_once()


@pytest.mark.asyncio
async def test_on_telegram_message_outside_allow_list_blocked(
    telegram_platform_multi,
):
    handler = AsyncMock()
    telegram_platform_multi.on_message(handler)

    mock_update = MagicMock()
    mock_update.message.text = "hello"
    mock_update.effective_user.id = 444  # not in {111, 222, 333}
    mock_update.effective_chat.id = 6789

    await telegram_platform_multi._on_telegram_message(mock_update, MagicMock())

    handler.assert_not_called()


@pytest.mark.asyncio
async def test_on_telegram_message_substring_id_does_not_authorize():
    """`"11"` must not authorize when allow-list is `"111,222"` — a prior
    `str.startswith` or `in` against the raw string would have allowed this."""
    with patch("messaging.platforms.telegram.TELEGRAM_AVAILABLE", True):
        platform = TelegramPlatform(bot_token="test_token", allowed_user_id="111,222")
    handler = AsyncMock()
    platform.on_message(handler)

    mock_update = MagicMock()
    mock_update.message.text = "hello"
    mock_update.effective_user.id = 11
    mock_update.effective_chat.id = 6789

    await platform._on_telegram_message(mock_update, MagicMock())

    handler.assert_not_called()


@pytest.mark.asyncio
async def test_on_telegram_voice_each_authorized_id_passes(
    telegram_platform_multi,
):
    """Auth gate on the voice path mirrors the text path. The downstream
    transcription pipeline is heavy to stub, so we drive each allowed id
    through with no message handler bound — the gate's only observable
    pre-handler effect is whether the early-return warning is logged.
    Allowed ids must NOT trigger 'Unauthorized voice access attempt'."""
    telegram_platform_multi._message_handler = None  # short-circuit after gate

    captured: list[str] = []
    sink_id = logger.add(lambda msg: captured.append(str(msg)), level="WARNING")
    try:
        for uid in (111, 222, 333):
            mock_update = MagicMock()
            mock_update.message.voice = MagicMock()
            mock_update.effective_user.id = uid
            mock_update.effective_chat.id = 6789

            await telegram_platform_multi._on_telegram_voice(mock_update, MagicMock())
    finally:
        logger.remove(sink_id)

    assert not any("Unauthorized voice access attempt" in line for line in captured)


@pytest.mark.asyncio
async def test_on_telegram_voice_outside_allow_list_blocked(
    telegram_platform_multi,
):
    telegram_platform_multi._message_handler = None  # in case the gate is bypassed

    captured: list[str] = []
    sink_id = logger.add(lambda msg: captured.append(str(msg)), level="WARNING")
    try:
        mock_update = MagicMock()
        mock_update.message.voice = MagicMock()
        mock_update.effective_user.id = 444  # not in allow-list
        mock_update.effective_chat.id = 6789

        await telegram_platform_multi._on_telegram_voice(mock_update, MagicMock())
    finally:
        logger.remove(sink_id)

    assert any(
        "Unauthorized voice access attempt from 444" in line for line in captured
    )


def _patched_app_builder():
    """Helper: a `telegram.ext.Application.builder` mock whose returned app
    looks alive enough to drive `TelegramPlatform.start()` to completion."""
    mock_app = MagicMock()
    mock_app.initialize = AsyncMock()
    mock_app.start = AsyncMock()
    mock_app.updater.start_polling = AsyncMock()
    builder = patch("telegram.ext.Application.builder")
    return builder, mock_app


@pytest.mark.asyncio
async def test_startup_notification_broadcast_to_all_allowed_users():
    """The startup ping should reach every allowed user, not just the first."""
    with patch("messaging.platforms.telegram.TELEGRAM_AVAILABLE", True):
        platform = TelegramPlatform(
            bot_token="test_token", allowed_user_id="111,222,333"
        )

    fake_send = AsyncMock(return_value="1")
    builder, mock_app = _patched_app_builder()
    with (
        builder as mock_builder,
        patch("messaging.limiter.MessagingRateLimiter.get_instance", AsyncMock()),
        patch.object(platform, "send_message", fake_send),
    ):
        mock_builder.return_value.token.return_value.request.return_value.build.return_value = mock_app
        await platform.start()

    sent_targets = [call.args[0] for call in fake_send.await_args_list]
    assert sorted(sent_targets) == ["111", "222", "333"]


@pytest.mark.asyncio
async def test_startup_notification_one_target_failure_does_not_silence_others():
    """A single user blocking the bot must not stop the rest from getting pinged."""
    with patch("messaging.platforms.telegram.TELEGRAM_AVAILABLE", True):
        platform = TelegramPlatform(
            bot_token="test_token", allowed_user_id="111,222,333"
        )

    delivered: list[str] = []

    async def fake_send(target, *_args, **_kwargs):
        if target == "222":
            raise RuntimeError("user blocked the bot")
        delivered.append(target)
        return "1"

    builder, mock_app = _patched_app_builder()
    with (
        builder as mock_builder,
        patch("messaging.limiter.MessagingRateLimiter.get_instance", AsyncMock()),
        patch.object(platform, "send_message", fake_send),
    ):
        mock_builder.return_value.token.return_value.request.return_value.build.return_value = mock_app
        await platform.start()

    assert sorted(delivered) == ["111", "333"]


@pytest.mark.asyncio
async def test_startup_notification_skipped_when_no_allow_list():
    """No allow-list ⇒ no broadcast; bot still starts cleanly."""
    with patch("messaging.platforms.telegram.TELEGRAM_AVAILABLE", True):
        platform = TelegramPlatform(bot_token="test_token", allowed_user_id=None)

    fake_send = AsyncMock(return_value="1")
    builder, mock_app = _patched_app_builder()
    with (
        builder as mock_builder,
        patch("messaging.limiter.MessagingRateLimiter.get_instance", AsyncMock()),
        patch.object(platform, "send_message", fake_send),
    ):
        mock_builder.return_value.token.return_value.request.return_value.build.return_value = mock_app
        await platform.start()

    fake_send.assert_not_awaited()
    assert platform._connected is True
