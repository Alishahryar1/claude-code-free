"""Native Messages relay and original reasoning capture for history replay."""

import json
from collections.abc import Mapping
from dataclasses import replace

from free_claude_code.core.history_replay import (
    ReplayOrigin,
    ReplayRecord,
    encode_replay,
)
from free_claude_code.core.json_types import JsonObject, JsonValue

from .native import NativeMessagesError


class NativeReasoningBlocks:
    """Collect only the original reasoning needed to write complete replay records."""

    def __init__(self) -> None:
        self._blocks: dict[int, JsonObject] = {}
        self._fragments: dict[int, dict[str, list[str]]] = {}

    def feed(
        self, event_type: str, payload: Mapping[str, JsonValue]
    ) -> JsonObject | None:
        if event_type == "message_stop" and self._blocks:
            raise NativeMessagesError("Messages ended with incomplete reasoning.")
        index = payload.get("index")
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            return None
        if event_type == "content_block_start":
            block = payload.get("content_block")
            if index in self._blocks:
                raise NativeMessagesError("Messages replaced incomplete reasoning.")
            if isinstance(block, Mapping) and block.get("type") in {
                "thinking",
                "redacted_thinking",
            }:
                self._blocks[index] = dict(block)
                self._fragments[index] = {}
            return None
        block = self._blocks.get(index)
        if block is None:
            return None
        if event_type == "content_block_delta":
            delta = payload.get("delta")
            if isinstance(delta, Mapping):
                key = {
                    "thinking_delta": "thinking",
                    "signature_delta": "signature",
                }.get(str(delta.get("type")))
                if key is not None:
                    value = delta.get(key)
                    if block["type"] != "thinking" or not isinstance(value, str):
                        raise NativeMessagesError("Invalid native reasoning fragment.")
                    self._fragments[index].setdefault(key, []).append(value)
        if event_type != "content_block_stop":
            return None
        for key, parts in self._fragments.pop(index).items():
            initial = block.get(key, "")
            if not isinstance(initial, str):
                raise NativeMessagesError("Invalid native reasoning field.")
            block[key] = initial + "".join(parts)
        if block["type"] == "thinking":
            if not isinstance(block.get("thinking"), str) or not (
                isinstance(block.get("signature"), str) and block["signature"]
            ):
                raise NativeMessagesError(
                    "Completed native thinking requires its signature."
                )
        elif not (isinstance(block.get("data"), str) and block["data"]):
            raise NativeMessagesError("Redacted native thinking requires its data.")
        return self._blocks.pop(index)


class NativeMessagesRelay:
    """Preserve upstream IDs, indexes, fields and event order for Messages clients."""

    def __init__(
        self, *, public_model: str, replay_origin: ReplayOrigin | None = None
    ) -> None:
        self._public_model = public_model
        self._replay_origin = replay_origin
        self._reasoning = NativeReasoningBlocks()
        self._started = False
        self._open_blocks: set[int] = set()
        self._stop_reason: str | None = None
        self.completed = False

    def feed(self, event_type: str, payload: Mapping[str, JsonValue]) -> str:
        if self.completed:
            raise NativeMessagesError("Native event arrived after message completion.")
        if event_type == "message_start":
            if self._started:
                raise NativeMessagesError("Duplicate Messages start.")
        elif event_type != "ping" and not self._started:
            raise NativeMessagesError("Messages event arrived before message start.")
        completed = self._reasoning.feed(event_type, payload)
        if event_type in {"content_block_start", "content_block_stop"}:
            index = payload.get("index")
            if not isinstance(index, int) or isinstance(index, bool) or index < 0:
                raise NativeMessagesError("Messages content requires a block index.")
            if event_type == "content_block_start":
                if index in self._open_blocks:
                    raise NativeMessagesError("Messages replaced incomplete content.")
                self._open_blocks.add(index)
            else:
                if index not in self._open_blocks:
                    raise NativeMessagesError(
                        "Messages closed content that never started."
                    )
                self._open_blocks.remove(index)
        elif event_type == "message_delta":
            delta = payload.get("delta")
            reason = delta.get("stop_reason") if isinstance(delta, Mapping) else None
            if isinstance(reason, str) and reason:
                self._stop_reason = reason
        elif event_type == "message_stop":
            if self._open_blocks or self._stop_reason is None:
                raise NativeMessagesError("Messages ended before content completed.")
            self.completed = True
        body = dict(payload)
        if event_type == "message_start":
            message = body.get("message")
            if not isinstance(message, dict):
                raise NativeMessagesError("Messages start must contain a message.")
            self._started = True
            body["message"] = {**message, "model": self._public_model}
            if (
                self._replay_origin is not None
                and isinstance(message.get("model"), str)
                and message["model"]
            ):
                self._replay_origin = replace(
                    self._replay_origin, model=message["model"]
                )
        prefix = ""
        if self._replay_origin is not None:
            block = body.get("content_block")
            if event_type == "content_block_start" and isinstance(block, dict):
                if block.get("type") == "thinking":
                    body["content_block"] = {**block, "signature": ""}
                elif block.get("type") == "redacted_thinking":
                    body["content_block"] = {
                        **block,
                        "data": encode_replay(ReplayRecord(self._replay_origin, block)),
                    }
            delta = body.get("delta")
            if (
                event_type == "content_block_delta"
                and isinstance(delta, dict)
                and delta.get("type") == "signature_delta"
            ):
                return ""
            if (
                completed is not None
                and completed.get("type") == "thinking"
                and completed.get("signature")
            ):
                signature = encode_replay(ReplayRecord(self._replay_origin, completed))
                prefix = _event(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": body["index"],
                        "delta": {"type": "signature_delta", "signature": signature},
                    },
                )
        return prefix + _event(event_type, body)


def _event(kind: str, payload: JsonObject) -> str:
    return f"event: {kind}\ndata: {json.dumps(payload, ensure_ascii=False, allow_nan=False)}\n\n"
