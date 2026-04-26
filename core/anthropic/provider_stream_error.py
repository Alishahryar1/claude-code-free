"""Canonical Anthropic-style SSE sequence for provider-side streaming errors."""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from typing import Any

from core.anthropic.sse import SSEBuilder


def iter_provider_stream_error_sse_events(
    *,
    request: Any,
    input_tokens: int,
    error_message: str,
    sent_any_event: bool,
    log_raw_sse_events: bool,
    message_id: str | None = None,
) -> Iterator[str]:
    """Yield message_start (if needed), a text block with the error, then message_delta/stop."""
    mid = message_id or f"msg_{uuid.uuid4()}"
    model = getattr(request, "model", "") or ""
    sse = SSEBuilder(
        mid,
        model,
        input_tokens,
        log_raw_events=log_raw_sse_events,
    )
    if not sent_any_event:
        yield sse.message_start()
    yield from sse.emit_error(error_message)
    yield sse.message_delta("end_turn", 1)
    yield sse.message_stop()
