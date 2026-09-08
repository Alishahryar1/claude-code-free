"""Native relay validation never rewrites identities or completes a truncated stream."""

import json
from collections.abc import AsyncIterator
from typing import cast

import pytest

from free_claude_code.core.anthropic.native import NativeMessagesError
from free_claude_code.core.anthropic.native_stream import (
    NativeMessagesRelay,
)
from free_claude_code.core.anthropic.sse_aggregation import (
    aggregate_anthropic_sse_to_message,
)
from free_claude_code.core.anthropic.stream_contracts import parse_sse_lines
from free_claude_code.core.json_types import JsonObject, JsonValue

_START: JsonObject = {
    "type": "message_start",
    "message": {
        "id": "upstream-id",
        "type": "message",
        "model": "upstream",
        "role": "assistant",
        "content": [],
        "usage": {"input_tokens": 3, "output_tokens": 0},
        "native_extension": "kept",
    },
}


@pytest.mark.asyncio
@pytest.mark.parametrize("initial", [None, []])
async def test_native_citations_survive_block_completion_and_fragmented_aggregation(
    initial: JsonValue,
) -> None:
    citation: JsonObject = {
        "type": "char_location",
        "cited_text": "source",
        "document_index": 0,
        "start_char_index": 0,
        "end_char_index": 6,
        "document_title": "doc",
    }
    events: list[JsonObject] = [
        _START,
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": "answer", "citations": initial},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "citations_delta", "citation": citation},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 2},
        },
        {"type": "message_stop"},
    ]
    relay = NativeMessagesRelay(public_model="public")

    async def stream() -> AsyncIterator[str]:
        for event in events:
            chunk = relay.feed(cast(str, event["type"]), event)
            for start in range(0, len(chunk), 3):
                yield chunk[start : start + 3]

    message, error, complete = await aggregate_anthropic_sse_to_message(stream())
    assert complete and error is None
    assert message["content"][0]["citations"] == [citation]


def test_relay_preserves_native_identity_and_extensions_without_mutating_input() -> (
    None
):
    relay = NativeMessagesRelay(public_model="public")
    result = parse_sse_lines(relay.feed("message_start", _START).splitlines())[0].data
    assert result["message"]["model"] == "public"
    assert result["message"]["id"] == "upstream-id"
    assert result["message"]["native_extension"] == "kept"
    assert cast(JsonObject, _START["message"])["model"] == "upstream"
    block: JsonObject = {
        "type": "content_block_start",
        "index": 5,
        "content_block": {"type": "text", "text": "hello", "citations": []},
    }
    assert (
        parse_sse_lines(relay.feed("content_block_start", block).splitlines())[0].data
        == block
    )
    assert not relay.completed


@pytest.mark.parametrize(
    "block",
    [
        {"type": "text", "text": "partial"},
        {"type": "tool_use", "id": "call-a", "name": "lookup", "input": {}},
        {"type": "server_tool_use", "id": "call-a", "name": "lookup", "input": {}},
    ],
)
def test_native_stop_requires_every_started_block_to_close(block: JsonObject) -> None:
    relay = NativeMessagesRelay(public_model="public")
    events: list[JsonObject] = [
        _START,
        {"type": "content_block_start", "index": 3, "content_block": block},
        {
            "type": "content_block_start",
            "index": 9,
            "content_block": {"type": "text", "text": "done"},
        },
        {"type": "content_block_stop", "index": 9},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
    ]
    for event in events:
        relay.feed(cast(str, event["type"]), event)
    with pytest.raises(NativeMessagesError):
        relay.feed("message_stop", {"type": "message_stop"})
    assert not relay.completed


def test_relay_rejects_prestart_reasoning_before_collecting_it() -> None:
    relay = NativeMessagesRelay(public_model="public")
    with pytest.raises(NativeMessagesError):
        relay.feed(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 3,
                "content_block": {"type": "thinking", "thinking": "unstarted"},
            },
        )
    # Rejected input must not leave an open reasoning block in the relay.
    relay.feed("message_start", _START)
    relay.feed(
        "message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn"}}
    )
    relay.feed("message_stop", {"type": "message_stop"})
    assert relay.completed


@pytest.mark.parametrize("event", [_START, {"type": "ping"}, {"type": "message_stop"}])
def test_completed_relay_cannot_emit_more_events(event: JsonObject) -> None:
    relay = NativeMessagesRelay(public_model="public")
    relay.feed("message_start", _START)
    relay.feed(
        "message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn"}}
    )
    relay.feed("message_stop", {"type": "message_stop"})
    with pytest.raises(NativeMessagesError):
        relay.feed(cast(str, event["type"]), event)


@pytest.mark.asyncio
async def test_fragmented_native_relay_preserves_nonstreaming_thinking_and_usage() -> (
    None
):
    relay = NativeMessagesRelay(public_model="public")
    events: list[JsonObject] = [
        _START,
        {
            "type": "content_block_start",
            "index": 2,
            "content_block": {"type": "thinking", "thinking": ""},
        },
        {
            "type": "content_block_delta",
            "index": 2,
            "delta": {"type": "signature_delta", "signature": "sig-"},
        },
        {
            "type": "content_block_delta",
            "index": 2,
            "delta": {"type": "signature_delta", "signature": "end"},
        },
        {"type": "content_block_stop", "index": 2},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 8},
        },
        {"type": "message_stop"},
    ]

    async def stream() -> AsyncIterator[str]:
        for event in events:
            chunk = relay.feed(cast(str, event["type"]), event)
            for start in range(0, len(chunk), 5):
                yield chunk[start : start + 5]

    message, error, complete = await aggregate_anthropic_sse_to_message(stream())
    assert relay.completed and complete and error is None
    assert message["id"] == "upstream-id" and message["model"] == "public"
    assert message["content"] == [
        {"type": "thinking", "thinking": "", "signature": "sig-end"}
    ]
    assert message["usage"] == {"input_tokens": 3, "output_tokens": 8}


def _tool_events(
    kind: str,
    fragments: list[str],
    *,
    initial: JsonObject | None = None,
    close: bool = True,
) -> list[JsonObject]:
    return cast(
        list[JsonObject],
        [
            _START,
            {
                "type": "content_block_start",
                "index": 4,
                "content_block": {
                    "type": kind,
                    "id": "call-native",
                    "name": "lookup",
                    "input": initial or {},
                },
            },
            *[
                {
                    "type": "content_block_delta",
                    "index": 4,
                    "delta": {"type": "input_json_delta", "partial_json": fragment},
                }
                for fragment in fragments
            ],
            *([{"type": "content_block_stop", "index": 4}] if close else []),
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 7},
            },
            {"type": "message_stop"},
        ],
    )


async def _wire(events: list[JsonObject]) -> AsyncIterator[str]:
    for event in events:
        yield f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"


@pytest.mark.asyncio
async def test_native_server_tool_arguments_are_forwarded_and_aggregated() -> None:
    events = _tool_events("server_tool_use", ['{"query":', '"weather"}'])
    relay = NativeMessagesRelay(public_model="public")
    output = [
        parse_sse_lines(relay.feed(cast(str, event["type"]), event).splitlines())[
            0
        ].data
        for event in events
    ]
    assert output[1:] == events[1:]
    message, error, complete = await aggregate_anthropic_sse_to_message(_wire(output))
    assert complete and error is None
    assert message["content"] == [
        {
            "type": "server_tool_use",
            "id": "call-native",
            "name": "lookup",
            "input": {"query": "weather"},
        }
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["tool_use", "server_tool_use"])
@pytest.mark.parametrize("fragment", ['{"query":', "[]", '{"x":NaN}', '{"x":Infinity}'])
async def test_aggregation_rejects_invalid_complete_tool_arguments(
    kind: str, fragment: str
) -> None:
    _, error, complete = await aggregate_anthropic_sse_to_message(
        _wire(_tool_events(kind, [fragment]))
    )
    assert error is not None and not complete


@pytest.mark.asyncio
@pytest.mark.parametrize("close", [False, True])
async def test_aggregation_rejects_unclosed_or_mixed_tool_input(close: bool) -> None:
    _, error, complete = await aggregate_anthropic_sse_to_message(
        _wire(
            _tool_events(
                "tool_use", ['{"query":"new"}'], initial={"query": "old"}, close=close
            )
        )
    )
    assert error is not None and not complete
