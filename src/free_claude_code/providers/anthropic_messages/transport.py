"""Native Messages HTTP execution with one admitted recovery budget."""

import asyncio
import json
import sys
from collections.abc import AsyncIterator, Callable, Mapping
from typing import cast

import httpx2
from anthropic import APIStatusError, AsyncAnthropic, AsyncStream
from anthropic.types import MessageParam

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.anthropic.native import (
    NativeMessagesError,
    PreparedMessagesRequest,
    build_native_messages_request,
)
from free_claude_code.core.anthropic.native_stream import NativeMessagesRelay
from free_claude_code.core.history_replay import ReplayOrigin, prepare_history
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.openai_responses import (
    AnthropicToResponsesStream,
    OpenAIResponsesRequest,
    ResponsesConversionError,
    ResponsesMessagesRequest,
    build_responses_messages_request,
)
from free_claude_code.core.reasoning import (
    DEFAULT_REASONING_POLICY,
    ReasoningControl,
    ReasoningPolicy,
)
from free_claude_code.core.trace import trace_event
from free_claude_code.providers.admission import (
    ProviderAdmissionController,
    ProviderCorrectionAction,
    ProviderExecution,
    ProviderOperationKind,
)
from free_claude_code.providers.endpoint import EndpointContext, RequestEndpoint
from free_claude_code.providers.failure_policy import (
    RetryableProviderProtocolError,
    classify_provider_failure,
    context_window_exceeded_provider_failure,
    is_context_window_finish_reason,
    is_retryable_stream_error,
    normalize_anthropic_stream_error,
)
from free_claude_code.providers.history_replay import (
    history_retry_body,
    replay_origin,
    validate_history,
)
from free_claude_code.providers.http import ProviderAttemptScope, maybe_await_aclose
from free_claude_code.providers.reasoning_compatibility import (
    ReasoningCorrection,
    prepare_messages_reasoning,
)
from free_claude_code.providers.stream_recovery import (
    RecoveryController,
    RecoveryFailureAction,
)

from .request_policy import (
    DEFAULT_MESSAGES_OUTPUT_TOKENS,
    MessagesModelCapabilities,
    resolve_messages_options,
)

type _Presenter = NativeMessagesRelay | AnthropicToResponsesStream


class AnthropicMessagesTransport:
    """Borrow HTTP, endpoint and admission owners; retain each response until closed."""

    def __init__(
        self,
        *,
        client: AsyncAnthropic,
        admission: ProviderAdmissionController,
        provider_name: str,
        replay_scope: str,
        read_timeout_s: float,
        endpoint_transport: httpx2.AsyncBaseTransport | None = None,
        capabilities: MessagesModelCapabilities = MessagesModelCapabilities(),
    ) -> None:
        self._client = client
        self._endpoint_transport = endpoint_transport
        self._admission = admission
        self._provider_name = provider_name
        self._replay_scope = replay_scope
        self._read_timeout_s = read_timeout_s
        self._capabilities = capabilities

    def _messages_body(
        self,
        request: MessagesRequest,
        reasoning: ReasoningPolicy,
        model_info: ProviderModelInfo | None = None,
    ) -> PreparedMessagesRequest:
        validate_history(request.model_dump(mode="json"))
        request, reasoning = prepare_messages_reasoning(
            request,
            reasoning,
            model_info=model_info,
            can_disable=True,
            normal_max_tokens=DEFAULT_MESSAGES_OUTPUT_TOKENS,
        )
        try:
            options = resolve_messages_options(
                model=request.model,
                max_tokens=request.max_tokens,
                reasoning=reasoning,
                capabilities=self._capabilities,
                thinking=request.thinking,
                output_effort=request.output_config.get("effort")
                if request.output_config
                else None,
            )
            return build_native_messages_request(request, options=options)
        except (NativeMessagesError, ValueError) as error:
            raise InvalidRequestError(str(error)) from error

    def _responses_body(
        self, request: OpenAIResponsesRequest, reasoning: ReasoningPolicy
    ) -> ResponsesMessagesRequest:
        validate_history(request.model_dump(mode="json"))
        try:
            options = resolve_messages_options(
                model=request.model,
                max_tokens=request.max_output_tokens,
                reasoning=reasoning,
                capabilities=self._capabilities,
                output_effort=request.reasoning.get("effort")
                if request.reasoning
                else None,
            )
            return build_responses_messages_request(request, options=options)
        except (NativeMessagesError, ResponsesConversionError, ValueError) as error:
            raise InvalidRequestError(str(error)) from error

    def stream_messages(
        self,
        request: MessagesRequest,
        *,
        endpoint_context: EndpointContext,
        request_id: str | None = None,
        response_model: str | None = None,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
        model_info: ProviderModelInfo | None = None,
    ) -> AsyncIterator[str]:
        prepared_request, wire_reasoning = prepare_messages_reasoning(
            request,
            reasoning,
            model_info=model_info,
            can_disable=True,
            normal_max_tokens=DEFAULT_MESSAGES_OUTPUT_TOKENS,
        )
        prepared = self._messages_body(prepared_request, wire_reasoning)
        correction = (
            ReasoningCorrection(
                (("thinking",),),
                "max_tokens",
                DEFAULT_MESSAGES_OUTPUT_TOKENS,
                self._capabilities.max_output_tokens,
            )
            if reasoning.control is ReasoningControl.PREFER_OFF
            and wire_reasoning.control is ReasoningControl.OFF
            else None
        )
        return self._stream(
            prepared.body,
            reasoning_correction=correction,
            betas=prepared.betas,
            endpoint_context=endpoint_context,
            request_id=request_id,
            presenter_factory=lambda origin: NativeMessagesRelay(
                public_model=response_model or request.model, replay_origin=origin
            ),
        )

    def stream_responses(
        self,
        request: OpenAIResponsesRequest,
        *,
        endpoint_context: EndpointContext,
        request_id: str | None = None,
        response_model: str | None = None,
        reasoning: ReasoningPolicy = DEFAULT_REASONING_POLICY,
    ) -> AsyncIterator[str]:
        prepared = self._responses_body(request, reasoning)
        return self._stream(
            prepared.body,
            betas=(),
            endpoint_context=endpoint_context,
            request_id=request_id,
            presenter_factory=lambda origin: AnthropicToResponsesStream(
                request,
                public_model=response_model or request.model,
                tool_identities=prepared.tool_identities,
                replay_origin=origin,
            ),
        )

    async def _stream(
        self,
        body: JsonObject,
        *,
        betas: tuple[str, ...],
        endpoint_context: EndpointContext,
        request_id: str | None,
        presenter_factory: Callable[[ReplayOrigin], _Presenter],
        reasoning_correction: ReasoningCorrection | None = None,
    ) -> AsyncIterator[str]:
        execution = self._admission.start_execution(request_id=request_id)
        endpoint = RequestEndpoint(endpoint_context, self._endpoint_transport)
        run = self._run(
            body,
            reasoning_correction=reasoning_correction,
            betas=betas,
            execution=execution,
            endpoint=endpoint,
            presenter_factory=presenter_factory,
        )
        try:
            async for event in run:
                endpoint.commit()
                yield event
        except asyncio.CancelledError, GeneratorExit:
            raise
        except Exception as error:
            execution.fail(error)
            raise
        else:
            execution.succeed()
        finally:
            try:
                await maybe_await_aclose(run)
            finally:
                try:
                    await endpoint.aclose()
                finally:
                    execution.abandon()

    async def _run(
        self,
        body: JsonObject,
        *,
        betas: tuple[str, ...],
        endpoint: RequestEndpoint,
        execution: ProviderExecution,
        presenter_factory: Callable[[ReplayOrigin], _Presenter],
        reasoning_correction: ReasoningCorrection | None = None,
    ) -> AsyncIterator[str]:
        recovery = RecoveryController()
        while execution.can_attempt:
            scope: ProviderAttemptScope | None = None
            stream_opened = False
            sent_body = body
            try:
                attempt = await execution.open_attempt(ProviderOperationKind.GENERATION)
                scope = ProviderAttemptScope(
                    attempt,
                    provider_name=self._provider_name,
                    request_id=execution.request_id,
                )
                client = await endpoint.anthropic_client(self._client)
                snapshot = endpoint.snapshot
                if snapshot is None:
                    raise RuntimeError(
                        "Messages request requires an endpoint snapshot."
                    )
                origin = replay_origin(
                    self._replay_scope,
                    "messages",
                    str(body["model"]),
                    endpoint=snapshot,
                )
                sent_body = prepare_history(body, origin)
                presenter = presenter_factory(origin)
                sdk_stream = await client.messages.create(
                    model=cast(str, sent_body["model"]),
                    messages=cast(list[MessageParam], sent_body["messages"]),
                    max_tokens=cast(int, sent_body["max_tokens"]),
                    stream=True,
                    extra_body={
                        key: value
                        for key, value in sent_body.items()
                        if key not in {"model", "messages", "max_tokens", "stream"}
                    },
                    extra_headers=endpoint.anthropic_headers(betas),
                )
                scope.retain(sdk_stream.response)
                stream_opened = True
                async for event in AsyncStream.raw_events(sdk_stream.response):
                    if event.event == "ping":
                        if isinstance(presenter, NativeMessagesRelay):
                            for held in recovery.push("\n".join(event.raw) + "\n\n"):
                                yield held
                        continue
                    if event.event == "error":
                        try:
                            error_body = event.json()
                        except json.JSONDecodeError:
                            error_body = event.data
                        raise APIStatusError(
                            str(error_body)
                            or f"Error code: {sdk_stream.response.status_code}",
                            response=sdk_stream.response,
                            body=error_body,
                        )
                    if event.event not in {
                        "message_start",
                        "message_delta",
                        "message_stop",
                        "content_block_start",
                        "content_block_delta",
                        "content_block_stop",
                    }:
                        continue
                    payload = event.json()
                    if not isinstance(payload, dict):
                        raise NativeMessagesError(
                            "Messages event must contain an object."
                        )
                    payload.setdefault("type", event.event)
                    event_type = payload["type"]
                    delta = payload.get("delta")
                    if (
                        event_type == "message_delta"
                        and isinstance(delta, Mapping)
                        and is_context_window_finish_reason(delta.get("stop_reason"))
                    ):
                        raise context_window_exceeded_provider_failure()
                    output = presenter.feed(event_type, payload)
                    if event_type != "ping" and not attempt.accepted:
                        await attempt.accept()
                    for event in (output,) if isinstance(output, str) else output:
                        for held in recovery.push(event):
                            yield held
                    if presenter.completed:
                        break
                if not presenter.completed:
                    raise RetryableProviderProtocolError(
                        "Messages stream ended without message_stop."
                    )
                for event in recovery.flush():
                    yield event
                return
            except asyncio.CancelledError, GeneratorExit:
                raise
            except Exception as raw_error:
                error = (
                    RetryableProviderProtocolError(str(raw_error))
                    if isinstance(
                        raw_error,
                        NativeMessagesError | json.JSONDecodeError | UnicodeDecodeError,
                    )
                    else normalize_anthropic_stream_error(
                        raw_error,
                        provider_name=self._provider_name,
                        request_id=execution.request_id,
                    )
                )
                if scope is not None and await endpoint.retry_authentication(
                    error, scope.attempt, execution
                ):
                    recovery.discard()
                    continue
                attempt_failure = None
                if scope is not None and not recovery.committed:
                    corrected_history = history_retry_body(error, sent_body, "messages")
                    if corrected_history is not None:
                        retry = (
                            execution.can_attempt
                            if scope.attempt.accepted
                            else await scope.attempt.correct(error)
                            is ProviderCorrectionAction.RETRY
                        )
                        if retry:
                            body = corrected_history
                            recovery.discard()
                            continue
                if (
                    scope is not None
                    and reasoning_correction is not None
                    and not recovery.committed
                ):
                    corrected_body = reasoning_correction.retry_body(error, body)
                    if corrected_body is not None:
                        retry = (
                            execution.can_attempt
                            if scope.attempt.accepted
                            else await scope.attempt.correct(error)
                            is ProviderCorrectionAction.RETRY
                        )
                        reasoning_correction = None
                        if retry:
                            body = corrected_body
                            recovery.discard()
                            continue
                if scope is not None and not scope.attempt.accepted:
                    attempt_failure = await scope.attempt.fail(error)
                if attempt_failure is not None and attempt_failure.retry_allowed:
                    recovery.discard()
                    continue
                decision = recovery.advance_failure(
                    retryable=attempt_failure.retryable
                    if attempt_failure is not None
                    else is_retryable_stream_error(error),
                    stream_opened=stream_opened,
                    generated_output=recovery.committed,
                    complete_tool_salvageable=False,
                    attempts_remaining=execution.attempts_remaining,
                )
                if decision.action is RecoveryFailureAction.EARLY_RETRY:
                    recovery.discard()
                    continue
                failure = classify_provider_failure(
                    error,
                    provider_name=self._provider_name,
                    read_timeout_s=self._read_timeout_s,
                    request_id=execution.request_id,
                )
                trace_event(
                    stage="provider",
                    event="provider.response.error",
                    source="provider",
                    provider=self._provider_name,
                    request_id=execution.request_id,
                    transport="messages",
                    failure_kind=failure.kind.value,
                )
                if decision.committed and isinstance(
                    presenter, AnthropicToResponsesStream
                ):
                    execution.fail(failure)
                    for event in presenter.terminal_failure(failure):
                        yield event
                    return
                recovery.discard()
                raise failure from raw_error
            finally:
                if scope is not None:
                    await scope.aclose(active_error=sys.exception())
        if execution.last_failure is not None:
            raise execution.last_failure
        raise RuntimeError("Messages execution ended without a terminal result.")
