import asyncio
from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def test_telegram_platform_init_raises_when_dependency_missing():
    from messaging import telegram as telegram_mod

    with patch.object(telegram_mod, "TELEGRAM_AVAILABLE", False):
        from messaging.telegram import TelegramPlatform

        with pytest.raises(ImportError):
            TelegramPlatform(bot_token="x")


@pytest.mark.asyncio
async def test_telegram_platform_start_requires_token():
    with (
        patch.dict("os.environ", {}, clear=True),
        patch("messaging.telegram.TELEGRAM_AVAILABLE", True),
    ):
        from messaging.telegram import TelegramPlatform

        platform = TelegramPlatform(bot_token=None)
        with pytest.raises(ValueError):
            await platform.start()


@pytest.mark.asyncio
async def test_telegram_platform_stop_no_application_is_noop():
    with patch("messaging.telegram.TELEGRAM_AVAILABLE", True):
        from messaging.telegram import TelegramPlatform

        platform = TelegramPlatform(bot_token="t")
        platform._application = None
        platform._connected = True
        await platform.stop()
        assert platform.is_connected is False


@pytest.mark.asyncio
async def test_with_retry_returns_none_when_message_not_modified_network_error():
    with patch("messaging.telegram.TELEGRAM_AVAILABLE", True):
        from messaging.telegram import TelegramPlatform, NetworkError

        platform = TelegramPlatform(bot_token="t")

        async def _f():
            raise NetworkError("Message is not modified")

        assert await platform._with_retry(_f) is None


@pytest.mark.asyncio
async def test_with_retry_retries_network_error_then_succeeds(monkeypatch):
    with patch("messaging.telegram.TELEGRAM_AVAILABLE", True):
        from messaging.telegram import TelegramPlatform, NetworkError

        platform = TelegramPlatform(bot_token="t")

        monkeypatch.setattr(asyncio, "sleep", AsyncMock())

        calls = {"n": 0}

        async def _f():
            calls["n"] += 1
            if calls["n"] == 1:
                raise NetworkError("temporary")
            return "ok"

        assert await platform._with_retry(_f) == "ok"
        assert calls["n"] == 2


@pytest.mark.asyncio
async def test_with_retry_honors_retry_after_timedelta(monkeypatch):
    with patch("messaging.telegram.TELEGRAM_AVAILABLE", True):
        from messaging.telegram import TelegramPlatform, RetryAfter

        platform = TelegramPlatform(bot_token="t")

        monkeypatch.setattr(asyncio, "sleep", AsyncMock())

        calls = {"n": 0}

        async def _f():
            calls["n"] += 1
            if calls["n"] == 1:
                raise RetryAfter(retry_after=timedelta(seconds=0.01))
            return "ok"

        assert await platform._with_retry(_f) == "ok"
        assert calls["n"] == 2


@pytest.mark.asyncio
async def test_with_retry_drops_parse_mode_on_markdown_entity_error():
    with patch("messaging.telegram.TELEGRAM_AVAILABLE", True):
        from messaging.telegram import TelegramPlatform, TelegramError

        platform = TelegramPlatform(bot_token="t")

        calls = []

        async def _f(parse_mode=None):
            calls.append(parse_mode)
            if len(calls) == 1:
                raise TelegramError("Can't parse entities: bad markdown")
            return "ok"

        assert await platform._with_retry(_f, parse_mode="MarkdownV2") == "ok"
        assert calls == ["MarkdownV2", None]


@pytest.mark.asyncio
async def test_queue_send_message_without_limiter_calls_send_message():
    with patch("messaging.telegram.TELEGRAM_AVAILABLE", True):
        from messaging.telegram import TelegramPlatform

        platform = TelegramPlatform(bot_token="t")
        platform._limiter = None
        with patch.object(
            platform, "send_message", new_callable=AsyncMock
        ) as mock_send:
            mock_send.return_value = "1"
            assert await platform.queue_send_message("c", "t") == "1"
            mock_send.assert_awaited_once()


@pytest.mark.asyncio
async def test_queue_edit_message_without_limiter_calls_edit_message():
    with patch("messaging.telegram.TELEGRAM_AVAILABLE", True):
        from messaging.telegram import TelegramPlatform

        platform = TelegramPlatform(bot_token="t")
        platform._limiter = None
        with patch.object(
            platform, "edit_message", new_callable=AsyncMock
        ) as mock_edit:
            await platform.queue_edit_message("c", "1", "t")
            mock_edit.assert_awaited_once()


def test_fire_and_forget_non_coroutine_uses_ensure_future(monkeypatch):
    with patch("messaging.telegram.TELEGRAM_AVAILABLE", True):
        from messaging.telegram import TelegramPlatform

        platform = TelegramPlatform(bot_token="t")

        ef = MagicMock()
        monkeypatch.setattr(asyncio, "ensure_future", ef)

        platform.fire_and_forget(MagicMock())
        ef.assert_called_once()


@pytest.mark.asyncio
async def test_on_start_command_replies_and_forwards():
    with patch("messaging.telegram.TELEGRAM_AVAILABLE", True):
        from messaging.telegram import TelegramPlatform

        platform = TelegramPlatform(bot_token="t")
        with patch.object(
            platform, "_on_telegram_message", new_callable=AsyncMock
        ) as mock_msg:
            update = MagicMock()
            update.message.reply_text = AsyncMock()

            await platform._on_start_command(update, MagicMock())
            update.message.reply_text.assert_awaited_once()
            mock_msg.assert_awaited_once()


@pytest.mark.asyncio
async def test_on_telegram_message_handler_error_sends_error_message():
    with patch("messaging.telegram.TELEGRAM_AVAILABLE", True):
        from messaging.telegram import TelegramPlatform

        platform = TelegramPlatform(bot_token="t", allowed_user_id="123")
        with patch.object(
            platform, "send_message", new_callable=AsyncMock
        ) as mock_send:

            async def _boom(_incoming):
                raise RuntimeError("bad")

            platform.on_message(_boom)

            update = MagicMock()
            update.message.text = "hello"
            update.message.message_id = 7
            update.message.reply_to_message = None
            update.effective_user.id = 123
            update.effective_chat.id = 456

            await platform._on_telegram_message(update, MagicMock())
            mock_send.assert_awaited_once()


@pytest.mark.asyncio
async def test_telegram_start_retries_on_network_error(monkeypatch):
    with patch("messaging.telegram.TELEGRAM_AVAILABLE", True):
        from messaging.telegram import TelegramPlatform, NetworkError

        platform = TelegramPlatform(bot_token="token", allowed_user_id=None)

        monkeypatch.setattr(asyncio, "sleep", AsyncMock())

        with (
            patch("telegram.ext.Application.builder") as mock_builder,
            patch("messaging.limiter.MessagingRateLimiter.get_instance", AsyncMock()),
        ):
            mock_app = MagicMock()
            mock_app.initialize = AsyncMock(side_effect=[NetworkError("no"), None])
            mock_app.start = AsyncMock()
            mock_app.updater = None

            mock_builder.return_value.token.return_value.request.return_value.build.return_value = mock_app

            await platform.start()
            assert platform.is_connected is True
