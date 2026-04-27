"""Tests for session-level context usage tracking and SSE usage parsing."""

from __future__ import annotations

from collections.abc import AsyncIterator
from unittest.mock import MagicMock, patch

import pytest

from api.context_usage import (
    SessionContextUsageTracker,
    extract_usage_input_tokens_from_sse_event,
)
from api.services import ClaudeProxyService
from config.settings import Settings


def test_extract_usage_input_tokens_from_message_start_event() -> None:
    event = (
        "event: message_start\n"
        'data: {"type":"message_start","message":{"usage":{"input_tokens":156000,"output_tokens":1}}}\n\n'
    )

    assert extract_usage_input_tokens_from_sse_event(event) == 156000


def test_extract_usage_input_tokens_from_message_delta_event() -> None:
    event = (
        "event: message_delta\n"
        'data: {"type":"message_delta","usage":{"input_tokens":2000,"output_tokens":100}}\n\n'
    )

    assert extract_usage_input_tokens_from_sse_event(event) == 2000


def test_tracker_logs_context_bar_and_warns_on_threshold_crossing() -> None:
    tracker = SessionContextUsageTracker(warn_threshold=0.75)

    with (
        patch("api.context_usage.logger.info") as info_log,
        patch("api.context_usage.logger.warning") as warn_log,
    ):
        tracker.record_usage(model_name="z-ai/glm-5.1", input_tokens=100_000)
        tracker.record_usage(model_name="z-ai/glm-5.1", input_tokens=60_000)

    assert info_log.call_count == 2
    context_log_template = str(info_log.call_args_list[-1].args[0])
    assert context_log_template.startswith("[Context: ")
    assert "{}%" in context_log_template
    assert "{}/{} tokens" in context_log_template
    assert warn_log.call_count == 1
    assert "Run /compact soon." in str(warn_log.call_args.args[0])


def test_tracker_skips_unknown_model_without_logging() -> None:
    tracker = SessionContextUsageTracker(warn_threshold=0.75)
    with patch("api.context_usage.logger.info") as info_log:
        result = tracker.record_usage(
            model_name="custom/unknown-model", input_tokens=500
        )

    assert result is None
    info_log.assert_not_called()


@pytest.mark.asyncio
async def test_service_stream_tracking_reads_usage_from_sse_event() -> None:
    settings = Settings()
    tracker = MagicMock(spec=SessionContextUsageTracker)

    service = ClaudeProxyService(
        settings,
        provider_getter=lambda _: MagicMock(),
        context_usage_tracker=tracker,
    )

    async def fake_stream() -> AsyncIterator[str]:
        yield (
            "event: message_start\n"
            'data: {"type":"message_start","message":{"usage":{"input_tokens":1234,"output_tokens":1}}}\n\n'
        )
        yield 'event: message_stop\ndata: {"type":"message_stop"}\n\n'

    chunks = [
        chunk
        async for chunk in service._stream_with_context_usage_tracking(
            fake_stream(), model_name="z-ai/glm-5.1"
        )
    ]

    assert len(chunks) == 2
    tracker.record_usage.assert_called_once_with(
        model_name="z-ai/glm-5.1",
        input_tokens=1234,
    )
