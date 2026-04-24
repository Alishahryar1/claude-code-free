"""Tests for api/fallback.py — provider fallback orchestration."""

from unittest.mock import MagicMock, patch

import httpx
import openai
import pytest

from api.fallback import is_fallback_trigger, stream_with_fallback
from providers.exceptions import (
    APIError,
    AuthenticationError,
    InvalidRequestError,
    OverloadedError,
    RateLimitError,
)

# ---------------------------------------------------------------------------
# is_fallback_trigger tests
# ---------------------------------------------------------------------------


class TestIsFallbackTrigger:
    """Test which errors trigger provider fallback."""

    def test_rate_limit_error(self):
        assert is_fallback_trigger(RateLimitError("rate limited")) is True

    def test_openai_rate_limit_error(self):
        err = openai.RateLimitError(
            message="rate limited",
            response=MagicMock(status_code=429),
            body=None,
        )
        assert is_fallback_trigger(err) is True

    def test_overloaded_error(self):
        assert is_fallback_trigger(OverloadedError("overloaded")) is True

    def test_openai_internal_server_error(self):
        err = openai.InternalServerError(
            message="internal error",
            response=MagicMock(status_code=500),
            body=None,
        )
        assert is_fallback_trigger(err) is True

    def test_api_error_500(self):
        assert is_fallback_trigger(APIError("fail", status_code=500)) is True

    def test_api_error_502(self):
        assert is_fallback_trigger(APIError("fail", status_code=502)) is True

    def test_api_error_400_no_trigger(self):
        assert is_fallback_trigger(APIError("fail", status_code=400)) is False

    def test_connect_error(self):
        assert is_fallback_trigger(httpx.ConnectError("refused")) is True

    def test_connect_timeout(self):
        assert is_fallback_trigger(httpx.ConnectTimeout("timeout")) is True

    def test_read_timeout(self):
        assert is_fallback_trigger(httpx.ReadTimeout("timeout")) is True

    def test_timeout_error(self):
        assert is_fallback_trigger(TimeoutError("timeout")) is True

    def test_auth_error_no_trigger(self):
        assert is_fallback_trigger(AuthenticationError("bad key")) is False

    def test_openai_auth_error_no_trigger(self):
        err = openai.AuthenticationError(
            message="bad key",
            response=MagicMock(status_code=401),
            body=None,
        )
        assert is_fallback_trigger(err) is False

    def test_invalid_request_no_trigger(self):
        assert is_fallback_trigger(InvalidRequestError("bad request")) is False

    def test_openai_bad_request_no_trigger(self):
        err = openai.BadRequestError(
            message="bad request",
            response=MagicMock(status_code=400),
            body=None,
        )
        assert is_fallback_trigger(err) is False

    def test_httpx_status_error_500(self):
        response = MagicMock()
        response.status_code = 500
        err = httpx.HTTPStatusError(
            "server error", request=MagicMock(), response=response
        )
        assert is_fallback_trigger(err) is True

    def test_httpx_status_error_429(self):
        response = MagicMock()
        response.status_code = 429
        err = httpx.HTTPStatusError(
            "rate limited", request=MagicMock(), response=response
        )
        assert is_fallback_trigger(err) is True

    def test_httpx_status_error_400_no_trigger(self):
        response = MagicMock()
        response.status_code = 400
        err = httpx.HTTPStatusError(
            "bad request", request=MagicMock(), response=response
        )
        assert is_fallback_trigger(err) is False

    def test_generic_exception_no_trigger(self):
        assert is_fallback_trigger(ValueError("something")) is False

    def test_generic_openai_api_error_500(self):
        mock_resp = MagicMock(status_code=500)
        err = openai.APIStatusError(
            message="fail",
            response=mock_resp,
            body=None,
        )
        assert is_fallback_trigger(err) is True

    def test_generic_openai_api_error_429(self):
        mock_resp = MagicMock(status_code=429)
        err = openai.APIStatusError(
            message="fail",
            response=mock_resp,
            body=None,
        )
        assert is_fallback_trigger(err) is True


# ---------------------------------------------------------------------------
# stream_with_fallback tests
# ---------------------------------------------------------------------------


async def _collect(gen):
    """Collect all items from an async generator."""
    return [item async for item in gen]


def _make_mock_request():
    """Create a minimal mock MessagesRequest."""
    req = MagicMock()
    req.model = "test-model"
    req.original_model = "claude-sonnet-4-20250514"
    return req


def _make_provider_that_yields(events):
    """Create a mock provider whose stream_response yields the given events."""
    provider = MagicMock()

    async def _stream(*args, **kwargs):
        for event in events:
            yield event

    provider.stream_response = _stream
    return provider


def _make_provider_that_raises(error):
    """Create a mock provider whose stream_response raises the given error."""
    provider = MagicMock()

    async def _stream(*args, **kwargs):
        raise error
        yield  # make it a generator

    provider.stream_response = _stream
    return provider


class TestStreamWithFallback:
    """Test the stream_with_fallback orchestrator."""

    @pytest.mark.asyncio
    async def test_primary_succeeds_no_fallback_invoked(self):
        primary = _make_provider_that_yields(["event1", "event2"])
        request = _make_mock_request()

        events = await _collect(
            stream_with_fallback(primary, request, 100, "req_123", fallback_model=None)
        )
        assert events == ["event1", "event2"]

    @pytest.mark.asyncio
    async def test_primary_succeeds_fallback_configured_but_not_used(self):
        primary = _make_provider_that_yields(["event1", "event2"])
        request = _make_mock_request()

        events = await _collect(
            stream_with_fallback(
                primary,
                request,
                100,
                "req_123",
                fallback_model="open_router/fallback-model",
            )
        )
        assert events == ["event1", "event2"]

    @pytest.mark.asyncio
    async def test_primary_fails_no_fallback_configured(self):
        primary = _make_provider_that_raises(httpx.ConnectError("refused"))
        request = _make_mock_request()

        with pytest.raises(httpx.ConnectError):
            await _collect(
                stream_with_fallback(
                    primary, request, 100, "req_123", fallback_model=None
                )
            )

    @pytest.mark.asyncio
    async def test_primary_fails_non_trigger_error_no_fallback(self):
        primary = _make_provider_that_raises(AuthenticationError("bad key"))
        request = _make_mock_request()

        with pytest.raises(AuthenticationError):
            await _collect(
                stream_with_fallback(
                    primary,
                    request,
                    100,
                    "req_123",
                    fallback_model="open_router/fallback-model",
                )
            )

    @pytest.mark.asyncio
    @patch("api.fallback.get_provider_for_type")
    async def test_primary_fails_fallback_succeeds(self, mock_get_provider):
        primary = _make_provider_that_raises(httpx.ConnectError("refused"))
        fallback = _make_provider_that_yields(["fb_event1", "fb_event2"])
        mock_get_provider.return_value = fallback

        request = _make_mock_request()

        events = await _collect(
            stream_with_fallback(
                primary,
                request,
                100,
                "req_123",
                fallback_model="open_router/fallback-model",
            )
        )
        assert events == ["fb_event1", "fb_event2"]
        mock_get_provider.assert_called_once_with("open_router")

    @pytest.mark.asyncio
    @patch("api.fallback.get_provider_for_type")
    async def test_primary_fails_fallback_also_fails(self, mock_get_provider):
        primary = _make_provider_that_raises(httpx.ConnectError("refused"))
        fallback = _make_provider_that_raises(httpx.ReadTimeout("timeout"))
        mock_get_provider.return_value = fallback

        request = _make_mock_request()

        with pytest.raises(httpx.ReadTimeout):
            await _collect(
                stream_with_fallback(
                    primary,
                    request,
                    100,
                    "req_123",
                    fallback_model="open_router/fallback-model",
                )
            )

    @pytest.mark.asyncio
    @patch("api.fallback.get_provider_for_type")
    async def test_model_restored_after_fallback(self, mock_get_provider):
        primary = _make_provider_that_raises(httpx.ConnectError("refused"))
        fallback = _make_provider_that_yields(["fb_event"])
        mock_get_provider.return_value = fallback

        request = _make_mock_request()
        original_model = request.model

        await _collect(
            stream_with_fallback(
                primary,
                request,
                100,
                "req_123",
                fallback_model="open_router/fallback-model",
            )
        )
        assert request.model == original_model

    @pytest.mark.asyncio
    @patch("api.fallback.get_provider_for_type")
    async def test_model_restored_after_fallback_failure(self, mock_get_provider):
        primary = _make_provider_that_raises(httpx.ConnectError("refused"))
        fallback = _make_provider_that_raises(httpx.ConnectError("also refused"))
        mock_get_provider.return_value = fallback

        request = _make_mock_request()
        original_model = request.model

        with pytest.raises(httpx.ConnectError):
            await _collect(
                stream_with_fallback(
                    primary,
                    request,
                    100,
                    "req_123",
                    fallback_model="open_router/fallback-model",
                )
            )
        assert request.model == original_model

    @pytest.mark.asyncio
    @patch("api.fallback.get_provider_for_type")
    async def test_fallback_model_name_set_during_fallback(self, mock_get_provider):
        """Verify the request.model is set to the fallback model name during streaming."""
        captured_model = None

        provider = MagicMock()

        async def _capture_stream(request, *args, **kwargs):
            nonlocal captured_model
            captured_model = request.model
            yield "event"

        provider.stream_response = _capture_stream

        primary = _make_provider_that_raises(httpx.ConnectError("refused"))
        mock_get_provider.return_value = provider

        request = _make_mock_request()
        await _collect(
            stream_with_fallback(
                primary,
                request,
                100,
                "req_123",
                fallback_model="open_router/my-fallback/model",
            )
        )
        assert captured_model == "my-fallback/model"

    @pytest.mark.asyncio
    @patch("api.fallback.get_provider_for_type")
    async def test_rate_limit_triggers_fallback(self, mock_get_provider):
        primary = _make_provider_that_raises(RateLimitError("429"))
        fallback = _make_provider_that_yields(["fb_event"])
        mock_get_provider.return_value = fallback

        request = _make_mock_request()
        events = await _collect(
            stream_with_fallback(
                primary,
                request,
                100,
                "req_123",
                fallback_model="nvidia_nim/backup-model",
            )
        )
        assert events == ["fb_event"]
        mock_get_provider.assert_called_once_with("nvidia_nim")
