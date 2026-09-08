"""Native Messages attempts preserve protocol, commitment and resource ownership."""

import asyncio
import json
from collections.abc import AsyncIterator, Callable

import httpx2
import pytest
from anthropic import AsyncAnthropic

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.openai_responses import OpenAIResponsesRequest
from free_claude_code.core.reasoning import ReasoningPolicy
from free_claude_code.providers.admission import ProviderAdmissionController
from free_claude_code.providers.anthropic_messages.transport import (
    AnthropicMessagesTransport,
)
from free_claude_code.providers.endpoint import HttpEndpoint, RequestEndpoint
from free_claude_code.providers.http import maybe_await_aclose
from tests.providers.support import immediate_admission


@pytest.mark.asyncio
@pytest.mark.parametrize("stream_error", [False, True])
@pytest.mark.parametrize(
    "error",
    [
        {
            "type": "invalid_request_error",
            "message": "Reasoning is mandatory and cannot be disabled.",
        },
        {
            "type": "invalid_request_error",
            "param": "thinking.type",
            "message": "Value 'disabled' is not supported",
        },
        {
            "type": "invalid_request_error",
            "loc": ["body", "thinking", "type"],
            "message": "Value 'disabled' is not supported",
        },
    ],
    ids=["mandatory", "param", "loc"],
)
async def test_tolerant_classifier_corrects_required_reasoning(stream_error, error):
    bodies = []

    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        if body.get("thinking", {}).get("type") == "disabled":
            if stream_error:
                return httpx2.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    stream=Wire([_sse({"type": "error", "error": error})]),
                )
            return httpx2.Response(400, json={"error": error})
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=Wire([_sse(*_events("<severity>0</severity>"))]),
        )

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as client:
        output = [
            event
            async for event in _transport(client).stream_messages(
                MessagesRequest.model_validate(
                    {
                        "model": "route",
                        "messages": [{"role": "user", "content": "classify"}],
                        "max_tokens": 64,
                        "thinking": {"type": "disabled"},
                    }
                ),
                endpoint_context=Endpoint(),
                reasoning=ReasoningPolicy.prefer_off(),
            )
        ]
    assert "<severity>0</severity>" in "".join(output)
    assert len(bodies) == 2
    assert bodies[0]["max_tokens"] == 8192
    assert "thinking" not in bodies[1]


class Endpoint:
    def __init__(self) -> None:
        self.refreshes: list[bool] = []
        self.token = "original"
        self.base_url = "https://native.invalid/v1/"

    async def endpoint(self, *, force_refresh: bool = False) -> HttpEndpoint:
        self.refreshes.append(force_refresh)
        return HttpEndpoint(
            self.base_url,
            {
                "Authorization": "Bearer fresh"
                if force_refresh
                else f"Bearer {self.token}",
                "Anthropic-Version": "2023-06-01",
            },
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("started", [False, True])
@pytest.mark.parametrize("kind", ["authentication_error", "permission_error"])
async def test_precommit_sse_auth_failure_refreshes_once(
    started: bool, kind: str
) -> None:
    calls = 0

    def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal calls
        calls += 1
        events = (
            _events()
            if calls > 1
            else [
                *(_events()[:1] if started else []),
                {"type": "error", "error": {"type": kind}},
            ]
        )
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=Wire([_sse(*events)]),
        )

    endpoint = Endpoint()
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as client:
        result = parse_sse_text(
            "".join(
                [
                    event
                    async for event in _stream(
                        _transport(client), endpoint, responses=True
                    )
                ]
            )
        )
    assert calls == 2 and endpoint.refreshes == [False, True]
    assert sum(event.event == "response.created" for event in result) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("separator", ["\u2028", "\u2029", "\u0085"])
async def test_unicode_line_characters_remain_inside_sse_text(separator: str) -> None:
    text = f"left{separator}right"
    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(
            lambda _: httpx2.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=Wire([_sse(*_events(text))]),
            )
        )
    ) as client:
        endpoint = Endpoint()
        output = parse_sse_text(
            "".join(
                [
                    event
                    async for event in _stream(
                        _transport(client), endpoint, responses=True
                    )
                ]
            )
        )
    assert endpoint.refreshes == [False]
    assert output[-1].data["response"]["output"][0]["content"][0]["text"] == text


class Wire(httpx2.AsyncByteStream):
    def __init__(
        self,
        chunks: list[bytes | Exception],
        *,
        closed: Callable[[], None] | None = None,
        close_gate: asyncio.Event | None = None,
    ) -> None:
        self.chunks = chunks
        self.closed = False
        self._on_close = closed
        self._close_gate = close_gate

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self.chunks:
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk

    async def aclose(self) -> None:
        if self._close_gate is not None:
            await self._close_gate.wait()
        self.closed = True
        if self._on_close is not None:
            self._on_close()


def _sse(*events: JsonObject) -> bytes:
    return "".join(
        f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
        for event in events
    ).encode()


def _events(text: str = "hello", stop: str = "end_turn") -> list[JsonObject]:
    return [
        {
            "type": "message_start",
            "message": {
                "id": "upstream-id",
                "type": "message",
                "model": "native",
                "role": "assistant",
                "content": [],
                "usage": {"input_tokens": 3, "output_tokens": 0},
            },
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": text},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop},
            "usage": {"output_tokens": 2},
        },
        {"type": "message_stop"},
    ]


def _transport(
    client: httpx2.AsyncClient, admission: ProviderAdmissionController | None = None
) -> AnthropicMessagesTransport:
    return AnthropicMessagesTransport(
        client=AsyncAnthropic(
            api_key="",
            base_url="https://unused.invalid",
            http_client=client,
            max_retries=0,
            timeout=3,
        ),
        endpoint_transport=client._transport,
        admission=admission or immediate_admission(max_attempts=2),
        provider_name="TEST",
        replay_scope="test/messages",
        read_timeout_s=3,
    )


def _stream(
    transport: AnthropicMessagesTransport, endpoint: Endpoint, *, responses: bool
) -> AsyncIterator[str]:
    if responses:
        return transport.stream_responses(
            OpenAIResponsesRequest(model="native", input="hi"),
            endpoint_context=endpoint,
            response_model="public",
        )
    return transport.stream_messages(
        MessagesRequest(
            model="native",
            messages=[{"role": "user", "content": "hi"}],
            betas=["test-beta"],
        ),
        endpoint_context=endpoint,
        response_model="public",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("responses", [False, True])
@pytest.mark.parametrize("versioned_base", [False, True])
async def test_fragmented_messages_http_keeps_native_path_and_public_identity(
    responses: bool,
    versioned_base: bool,
) -> None:
    raw = _sse(*_events())
    wire = Wire([raw[index : index + 7] for index in range(0, len(raw), 7)])
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(
            200, headers={"content-type": "text/event-stream"}, stream=wire
        )

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as client:
        endpoint = Endpoint()
        endpoint.base_url = (
            "https://native.invalid/v1/" if versioned_base else "https://native.invalid"
        )
        output = parse_sse_text(
            "".join(
                [
                    event
                    async for event in _stream(
                        _transport(client), endpoint, responses=responses
                    )
                ]
            )
        )
        assert not client.is_closed
    assert wire.closed and len(requests) == 1
    assert requests[0].url.path == "/v1/messages"
    assert requests[0].headers["Authorization"] == "Bearer original"
    assert requests[0].headers["anthropic-version"] == "2023-06-01"
    assert requests[0].headers.get_list("anthropic-version") == ["2023-06-01"]
    body = json.loads(requests[0].content)
    assert body["stream"] is True and body["model"] == "native"
    assert body["max_tokens"] > 0
    if responses:
        assert output[-1].data["response"]["model"] == "public"
        assert output[-1].event == "response.completed"
    else:
        assert requests[0].headers["anthropic-beta"] == "test-beta"
        assert output[0].data["message"]["id"] == "upstream-id"
        assert output[0].data["message"]["model"] == "public"
        assert output[-1].event == "message_stop"


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 403])
async def test_unauthorized_refresh_uses_same_budget_and_closes_before_next_request(
    status: int,
) -> None:
    endpoint = Endpoint()
    wires: list[Wire] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        if wires:
            assert wires[0].closed
            assert request.headers["Authorization"] == "Bearer fresh"
        wire = Wire([b'{"error":"expired"}' if not wires else _sse(*_events())])
        wires.append(wire)
        return httpx2.Response(
            status if len(wires) == 1 else 200,
            headers={"content-type": "text/event-stream"},
            stream=wire,
        )

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as client:
        assert [
            item async for item in _stream(_transport(client), endpoint, responses=True)
        ]
    assert endpoint.refreshes == [False, True] and all(wire.closed for wire in wires)


@pytest.mark.asyncio
async def test_repeated_unauthorized_cannot_create_unbounded_auth_retry() -> None:
    calls = 0

    def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal calls
        calls += 1
        return httpx2.Response(401, json={"error": {"type": "authentication_error"}})

    endpoint = Endpoint()
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as client:
        with pytest.raises(ExecutionFailure) as caught:
            _ = [
                event
                async for event in _stream(_transport(client), endpoint, responses=True)
            ]
    assert caught.value.kind is FailureKind.AUTHENTICATION
    assert calls == 2 and endpoint.refreshes == [False, True]


@pytest.mark.asyncio
@pytest.mark.parametrize("responses", [False, True])
@pytest.mark.parametrize(
    "first",
    [
        b"event: content_block_delta\ndata: {bad}\n\n",
        _sse(*_events()[:3]),
        b'event: error\ndata: {"type":"error","error":{"type":"overloaded_error"}}\n\n',
        pytest.param(_sse(*_events()[:3], *_events()[4:]), id="unclosed-block"),
        pytest.param(_sse(*_events()[:4], _events()[-1]), id="missing-stop-reason"),
        pytest.param(_sse(*_events()[-2:]), id="missing-message-start"),
    ],
)
async def test_early_malformed_truncated_and_overloaded_attempts_retry_invisibly(
    first: bytes,
    responses: bool,
) -> None:
    calls = 0
    wires: list[Wire] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal calls
        calls += 1
        wire = Wire([first if calls == 1 else _sse(*_events("final"))])
        wires.append(wire)
        return httpx2.Response(
            200, headers={"content-type": "text/event-stream"}, stream=wire
        )

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as client:
        result = "".join(
            [
                event
                async for event in _stream(
                    _transport(client), Endpoint(), responses=responses
                )
            ]
        )
    events = parse_sse_text(result)
    assert calls == 2 and all(wire.closed for wire in wires)
    start = "response.created" if responses else "message_start"
    stop = "response.completed" if responses else "message_stop"
    assert sum(event.event == start for event in events) == 1
    assert events[-1].event == stop
    assert "final" in result and "hello" not in result


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize(
    "kind,field",
    [
        ("message_start", "message"),
        ("text", "text"),
        ("text_delta", "text"),
        ("thinking_delta", "thinking"),
        ("tool_use", "name"),
        ("tool_use", "id"),
    ],
)
async def test_malformed_converted_fields_retry_without_leaking_first_attempt(
    kind: str, field: str, missing: bool
) -> None:
    first = _events()[:3]
    malformed: JsonObject
    if kind == "message_start":
        first = first[:1]
        malformed = first[0]
    elif kind in {"text", "tool_use"}:
        first = first[:2]
        malformed = (
            {"type": "text", "text": ""}
            if kind == "text"
            else {"type": "tool_use", "id": "call-a", "name": "lookup", "input": {}}
        )
        first[1]["content_block"] = malformed
    else:
        first[1]["content_block"] = {
            "type": "thinking" if kind == "thinking_delta" else "text",
            field: "",
        }
        malformed = {"type": kind, field: ""}
        first[2]["delta"] = malformed
    if missing:
        malformed.pop(field)
    else:
        malformed[field] = None
    wires: list[Wire] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        wire = Wire([_sse(*first) if not wires else _sse(*_events("final"))])
        wires.append(wire)
        return httpx2.Response(
            200, headers={"content-type": "text/event-stream"}, stream=wire
        )

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as client:
        events = parse_sse_text(
            "".join(
                [
                    event
                    async for event in _stream(
                        _transport(client), Endpoint(), responses=True
                    )
                ]
            )
        )
    assert len(wires) == 2 and all(wire.closed for wire in wires)
    assert sum(event.event == "response.created" for event in events) == 1
    assert events[-1].event == "response.completed"
    assert events[-1].data["response"]["output"][0]["content"][0]["text"] == "final"


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", [False, True])
async def test_committed_malformed_text_preserves_output_without_retry(
    missing: bool,
) -> None:
    delta: JsonObject = {"type": "text_delta"}
    if not missing:
        delta["text"] = None
    wire = Wire(
        [
            _sse(*_events("x" * 70_000)[:3]),
            _sse({"type": "content_block_delta", "index": 0, "delta": delta}),
        ]
    )
    calls = 0

    def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal calls
        calls += 1
        return httpx2.Response(
            200, headers={"content-type": "text/event-stream"}, stream=wire
        )

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as client:
        events = parse_sse_text(
            "".join(
                [
                    event
                    async for event in _stream(
                        _transport(client), Endpoint(), responses=True
                    )
                ]
            )
        )
    assert calls == 1 and wire.closed
    assert sum(event.event == "response.failed" for event in events) == 1
    assert not any(event.event == "response.completed" for event in events)
    response = events[-1].data["response"]
    assert response["status"] == "failed"
    assert response["output"][0]["status"] == "incomplete"
    assert response["output"][0]["content"][0]["text"] == "x" * 70_000


@pytest.mark.asyncio
@pytest.mark.parametrize("responses", [False, True])
@pytest.mark.parametrize(
    "failure,item_status",
    [
        pytest.param(httpx2.ReadTimeout("stalled"), "incomplete", id="timeout"),
        pytest.param(
            _sse({"type": "error", "error": {"type": "authentication_error"}}),
            "incomplete",
            id="authentication",
        ),
        pytest.param(_sse(*_events()[4:]), "incomplete", id="unclosed-block"),
        pytest.param(
            _sse(_events()[3], _events()[-1]), "completed", id="missing-stop-reason"
        ),
    ],
)
async def test_committed_failure_never_retries_or_emits_success(
    responses: bool,
    failure: bytes | Exception,
    item_status: str,
) -> None:
    calls = 0
    wire = Wire(
        [
            _sse(*_events("x" * 70_000)[:3]),
            failure,
        ]
    )

    def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal calls
        calls += 1
        return httpx2.Response(
            200, headers={"content-type": "text/event-stream"}, stream=wire
        )

    chunks: list[str] = []
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as client:
        stream = _stream(_transport(client), Endpoint(), responses=responses)
        if responses:
            chunks = [chunk async for chunk in stream]
        else:
            with pytest.raises(ExecutionFailure):
                async for chunk in stream:
                    chunks.append(chunk)
    events = parse_sse_text("".join(chunks))
    assert calls == 1 and wire.closed
    assert not any(
        event.event in {"message_stop", "response.completed"} for event in events
    )
    if responses:
        assert sum(event.event == "response.failed" for event in events) == 1
        assert events[-1].data["response"]["status"] == "failed"
        assert events[-1].data["response"]["output"][0]["status"] == item_status


@pytest.mark.asyncio
@pytest.mark.parametrize("error_code", [False, True])
async def test_context_window_stop_is_nonretryable_canonical_failure(
    error_code: bool,
) -> None:
    calls = 0

    def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal calls
        calls += 1
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=Wire(
                [
                    _sse(
                        {
                            "type": "error",
                            "error": {
                                "type": "invalid_request_error",
                                "code": "context_length_exceeded",
                            },
                        }
                    )
                    if error_code
                    else _sse(*_events(stop="model_context_window_exceeded"))
                ]
            ),
        )

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as client:
        with pytest.raises(ExecutionFailure) as caught:
            _ = [
                event
                async for event in _stream(
                    _transport(client), Endpoint(), responses=True
                )
            ]
    assert caught.value.kind is FailureKind.CONTEXT_WINDOW_EXCEEDED and calls == 1


@pytest.mark.asyncio
async def test_cancelled_consumer_closes_response_before_releasing_admission() -> None:
    gate = asyncio.Event()
    first = Wire([_sse(*_events("x" * 70_000))], close_gate=gate)
    calls = 0

    def handler(request: httpx2.Request) -> httpx2.Response:
        nonlocal calls
        calls += 1
        if calls == 2:
            assert request.headers["Authorization"] == "Bearer new"
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=first if calls == 1 else Wire([_sse(*_events())]),
        )

    admission = ProviderAdmissionController(
        provider_name="TEST",
        rate_limit=1_000_000,
        rate_window=1,
        max_concurrency=1,
        max_attempts=2,
        base_delay=0,
        max_delay=0,
        jitter=0,
    )
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as client:
        transport = _transport(client, admission)
        stream = _stream(transport, Endpoint(), responses=True)
        assert await anext(stream)
        close = asyncio.create_task(maybe_await_aclose(stream))
        next_endpoint = Endpoint()
        next_stream = _stream(transport, next_endpoint, responses=True)
        next_request = asyncio.ensure_future(anext(next_stream))
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert calls == 1 and not close.done() and not next_request.done()
        assert next_endpoint.refreshes == []
        next_endpoint.token = "new"
        gate.set()
        await close
        assert await next_request
        await maybe_await_aclose(next_stream)
    assert first.closed and calls == 2


@pytest.mark.asyncio
async def test_task_cancellation_closes_live_http_response() -> None:
    entered = asyncio.Event()

    class BlockingWire(Wire):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield _sse(*_events("x" * 70_000)[:3])
            entered.set()
            await asyncio.Event().wait()

    wire = BlockingWire([])
    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(
            lambda _: httpx2.Response(
                200, headers={"content-type": "text/event-stream"}, stream=wire
            )
        )
    ) as client:
        stream = _stream(_transport(client), Endpoint(), responses=True)

        async def consume() -> None:
            async for _ in stream:
                pass

        task = asyncio.create_task(consume())
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert wire.closed


@pytest.mark.asyncio
async def test_auth_refresh_cannot_exceed_single_attempt_budget() -> None:
    endpoint = Endpoint()
    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(
            lambda _: httpx2.Response(401, json={"error": "expired"})
        )
    ) as client:
        with pytest.raises(ExecutionFailure):
            _ = [
                event
                async for event in _stream(
                    _transport(client, immediate_admission(max_attempts=1)),
                    endpoint,
                    responses=True,
                )
            ]
    assert endpoint.refreshes == [False]


@pytest.mark.asyncio
async def test_preflight_rejects_invalid_conversion_without_request_io() -> None:
    async with httpx2.AsyncClient() as client:
        transport = _transport(client)
        with pytest.raises(InvalidRequestError):
            transport.stream_responses(
                OpenAIResponsesRequest(
                    model="native", input=[{"type": "input_file", "file_id": "remote"}]
                ),
                endpoint_context=Endpoint(),
            )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("headers", "key", "auth", "api_key"),
    [
        ({"Authorization": "Bearer selected"}, "unused", "Bearer selected", None),
        ({"x-API-KEY": "header-key"}, "unused", None, "header-key"),
        ({}, "fallback-key", None, "fallback-key"),
        ({}, None, None, None),
    ],
)
async def test_sdk_endpoint_headers_override_ambient_credentials_and_cookies(
    monkeypatch, headers, key, auth, api_key
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ambient-key")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "ambient-token")
    monkeypatch.setenv(
        "ANTHROPIC_CUSTOM_HEADERS",
        "Authorization: Bearer ambient\nX-API-KEY: ambient\nCookie: ambient-cookie",
    )
    requests = []

    def respond(request):
        requests.append(request)
        return httpx2.Response(
            200,
            headers={
                "content-type": "text/event-stream",
                "set-cookie": "old=account; Path=/",
            },
            stream=Wire([_sse(*_events())]),
        )

    class Account:
        async def endpoint(self, *, force_refresh=False):
            return HttpEndpoint(
                "https://native.invalid/v1/",
                {
                    **headers,
                    "ANTHROPIC-BETA": "endpoint-beta,test-beta",
                    "Anthropic-Version": "custom-version",
                },
                key,
            )

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as http:
        provider = _transport(http)
        endpoint = RequestEndpoint(Account(), http._transport)
        try:
            for _ in range(2):
                client = await endpoint.anthropic_client(provider._client)
                stream = await client.messages.create(
                    model="generic",
                    messages=[{"role": "user", "content": "hi"}],
                    max_tokens=32,
                    stream=True,
                    extra_headers=endpoint.anthropic_headers(
                        ("test-beta", "request-beta")
                    ),
                )
                async with stream:
                    _ = [event async for event in stream]
        finally:
            await endpoint.aclose()
    assert len(requests) == 2
    for request in requests:
        assert request.headers.get("authorization") == auth
        assert request.headers.get("x-api-key") == api_key
        assert "cookie" not in request.headers
        assert (
            request.headers["anthropic-beta"] == "endpoint-beta,test-beta,request-beta"
        )
        assert request.headers.get_list("anthropic-version") == ["custom-version"]
        assert request.url.path == "/v1/messages"
    assert provider._client.api_key == "" and provider._client.auth_token is None


@pytest.mark.asyncio
@pytest.mark.parametrize("streamed", [False, True])
@pytest.mark.parametrize(
    ("status", "kind", "failure_kind", "expected_status"),
    [
        (400, "invalid_request_error", FailureKind.INVALID_REQUEST, 400),
        (401, "authentication_error", FailureKind.AUTHENTICATION, 401),
        (402, "billing_error", FailureKind.PERMISSION, 402),
        (403, "permission_error", FailureKind.PERMISSION, 403),
        (404, "not_found_error", FailureKind.INVALID_REQUEST, 404),
        (413, "request_too_large", FailureKind.INVALID_REQUEST, 413),
        (429, "rate_limit_error", FailureKind.RATE_LIMIT, 429),
        (500, "api_error", FailureKind.UPSTREAM, 500),
        (529, "overloaded_error", FailureKind.OVERLOADED, 529),
    ],
)
async def test_sdk_failure_status_details_and_single_attempt_budget(
    streamed, status, kind, failure_kind, expected_status
):
    requests = []
    error = {
        "type": kind,
        "message": "synthetic provider detail",
        "param": "test_parameter",
    }

    def respond(request):
        requests.append(request)
        if streamed:
            return httpx2.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=Wire([_sse({"type": "error", "error": error})]),
            )
        return httpx2.Response(status, json={"error": error})

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
        with pytest.raises(ExecutionFailure) as caught:
            _ = [
                event
                async for event in _transport(
                    client, immediate_admission(max_attempts=1)
                ).stream_messages(
                    MessagesRequest(
                        model="generic", messages=[{"role": "user", "content": "hi"}]
                    ),
                    endpoint_context=Endpoint(),
                    request_id="sdk-failure-test",
                )
            ]
    assert len(requests) == 1
    assert (
        caught.value.kind is failure_kind
        and caught.value.status_code == expected_status
    )
    assert "synthetic provider detail" in caught.value.message
    assert "test_parameter" in caught.value.message
    assert "Request ID: sdk-failure-test" in caught.value.message
    assert "HTTP 200" not in caught.value.message


@pytest.mark.asyncio
@pytest.mark.parametrize("responses", [False, True])
@pytest.mark.parametrize("error_type", [httpx2.ConnectError, httpx2.ConnectTimeout])
async def test_sdk_connection_failures_use_fcc_retry_budget(responses, error_type):
    requests = []

    def respond(request):
        requests.append(request)
        if len(requests) == 1:
            raise error_type("synthetic connection failure", request=request)
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=Wire([_sse(*_events())]),
        )

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
        _ = [
            event
            async for event in _stream(
                _transport(client), Endpoint(), responses=responses
            )
        ]
    assert len(requests) == 2


@pytest.mark.asyncio
async def test_sdk_native_server_tools_and_extra_fields_reach_the_client():
    events: list[JsonObject] = [
        _events()[0],
        {"type": "provider_progress", "ignored": True},
        {
            "type": "content_block_start",
            "index": 4,
            "content_block": {
                "type": "server_tool_use",
                "id": "srvtoolu_1",
                "name": "web_search",
                "input": {},
                "extension": 17,
            },
        },
        {
            "type": "content_block_delta",
            "index": 4,
            "delta": {
                "type": "input_json_delta",
                "partial_json": '{"query":"weather"}',
            },
        },
        {"type": "content_block_stop", "index": 4},
        {
            "type": "content_block_start",
            "index": 6,
            "content_block": {
                "type": "web_search_tool_result",
                "tool_use_id": "srvtoolu_1",
                "content": [],
                "extension": "kept",
            },
        },
        {"type": "content_block_stop", "index": 6},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "future_reason"},
            "usage": {"output_tokens": 7, "vendor_usage": 13},
        },
        {"type": "message_stop"},
    ]
    requests = []

    def respond(request):
        requests.append(request)
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=Wire([_sse(*events)]),
        )

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
        output = parse_sse_text(
            "".join(
                [
                    event
                    async for event in _stream(
                        _transport(client), Endpoint(), responses=False
                    )
                ]
            )
        )
    assert len(requests) == 1
    assert [event.data for event in output][1:] == events[2:]
