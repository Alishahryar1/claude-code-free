from __future__ import annotations

import httpx
import pytest

from config.provider_catalog import PROVIDER_CATALOG
from core.anthropic.stream_contracts import (
    assert_anthropic_stream_contract,
    parse_sse_lines,
    text_content,
)
from smoke.lib.config import ProviderModel, SmokeConfig, auth_headers
from smoke.lib.e2e import (
    ConversationDriver,
    ProviderMatrixDriver,
    SmokeServerDriver,
    assert_product_stream,
    echo_tool_schema,
    tool_use_blocks,
)
from smoke.lib.skips import (
    skip_if_upstream_unavailable_events,
    skip_if_upstream_unavailable_exception,
)

pytestmark = [pytest.mark.live, pytest.mark.smoke_target("providers")]


def test_provider_matrix_presence_e2e(smoke_config: SmokeConfig) -> None:
    models = ProviderMatrixDriver(smoke_config).provider_smoke_models()
    assert models or smoke_config.provider_matrix == frozenset()


def test_model_mapping_matrix_e2e(smoke_config: SmokeConfig) -> None:
    models = ProviderMatrixDriver(smoke_config).configured_models()
    sources = {model.source for model in models}
    assert sources <= {"MODEL", "MODEL_OPUS", "MODEL_SONNET", "MODEL_HAIKU"}
    for model in models:
        assert model.provider
        assert model.model_name


def test_provider_text_multiturn_e2e(smoke_config: SmokeConfig) -> None:
    _run_for_each_provider(smoke_config, _scenario_text_multiturn)


def test_provider_adaptive_thinking_history_e2e(smoke_config: SmokeConfig) -> None:
    _run_for_each_provider(smoke_config, _scenario_adaptive_thinking_history)


def test_provider_interleaved_thinking_tool_e2e(smoke_config: SmokeConfig) -> None:
    _run_for_each_provider(smoke_config, _scenario_interleaved_history)


@pytest.mark.smoke_target("tools")
def test_provider_tool_use_then_text_history_e2e(smoke_config: SmokeConfig) -> None:
    """OpenAI-compatible path: history with tool_use + assistant text after tool (issue #206)."""
    _run_for_each_provider(smoke_config, _scenario_tool_use_then_text_in_history)


@pytest.mark.smoke_target("tools")
def test_provider_tool_result_continuation_e2e(smoke_config: SmokeConfig) -> None:
    _run_for_each_provider(smoke_config, _scenario_tool_result_continuation)


@pytest.mark.smoke_target("tools")
def test_provider_reasoning_tool_continuation_e2e(smoke_config: SmokeConfig) -> None:
    _run_for_each_thinking_provider(smoke_config, _scenario_reasoning_tool_continuation)


@pytest.mark.smoke_target("rate_limit")
def test_provider_disconnect_e2e(smoke_config: SmokeConfig) -> None:
    _run_for_each_provider(smoke_config, _scenario_disconnect)


def test_provider_error_e2e(smoke_config: SmokeConfig) -> None:
    provider_model = ProviderMatrixDriver(smoke_config).first_model()
    broken_model = f"{provider_model.provider}/fcc-smoke-missing-model"
    with (
        SmokeServerDriver(
            smoke_config,
            name=f"product-provider-error-{provider_model.provider}",
            env_overrides={"MODEL": broken_model, "MESSAGING_PLATFORM": "none"},
        ).run() as server,
        httpx.stream(
            "POST",
            f"{server.base_url}/v1/messages",
            headers=auth_headers(),
            json={
                "model": "fcc-smoke-default",
                "max_tokens": 32,
                "messages": [{"role": "user", "content": "hello"}],
            },
            timeout=smoke_config.timeout_s,
        ) as response,
    ):
        assert response.status_code == 200, response.read()
        events = parse_sse_lines(response.iter_lines())
    assert_anthropic_stream_contract(events, allow_error=True)
    assert any(event.event == "error" for event in events) or text_content(events)


def test_openrouter_native_e2e(smoke_config: SmokeConfig) -> None:
    models = [
        model
        for model in ProviderMatrixDriver(smoke_config).provider_smoke_models()
        if model.provider == "open_router"
    ]
    if not models:
        pytest.skip("missing_env: open_router is not configured")

    provider_model = models[0]
    with SmokeServerDriver(
        smoke_config,
        name="product-openrouter-native",
        env_overrides={
            "MODEL": provider_model.full_model,
            "MESSAGING_PLATFORM": "none",
        },
    ).run() as server:
        turn = ConversationDriver(server, smoke_config).stream(
            {
                "model": "claude-opus-4-7",
                "max_tokens": 256,
                "messages": [
                    {
                        "role": "user",
                        "content": "Reply with one short sentence.",
                    }
                ],
                "thinking": {"type": "adaptive", "budget_tokens": 1024},
            }
        )
    assert_product_stream(turn.events)


def _run_for_each_provider(smoke_config: SmokeConfig, scenario) -> None:
    failures: list[str] = []
    for provider_model in ProviderMatrixDriver(smoke_config).provider_smoke_models():
        try:
            scenario(smoke_config, provider_model)
        except Exception as exc:
            skip_if_upstream_unavailable_exception(exc)
            failures.append(
                f"{provider_model.source}={provider_model.full_model}: "
                f"{type(exc).__name__}: {exc}"
            )
    assert not failures, "\n".join(failures)


def _run_for_each_thinking_provider(smoke_config: SmokeConfig, scenario) -> None:
    failures: list[str] = []
    models = [
        provider_model
        for provider_model in ProviderMatrixDriver(smoke_config).provider_smoke_models()
        if _provider_smoke_thinking_enabled(smoke_config, provider_model)
    ]
    if not models:
        pytest.skip("missing_env: no thinking-capable provider smoke model configured")
    for provider_model in models:
        try:
            scenario(smoke_config, provider_model)
        except Exception as exc:
            skip_if_upstream_unavailable_exception(exc)
            failures.append(
                f"{provider_model.source}={provider_model.full_model}: "
                f"{type(exc).__name__}: {exc}"
            )
    assert not failures, "\n".join(failures)


def _provider_smoke_thinking_enabled(
    smoke_config: SmokeConfig, provider_model: ProviderModel
) -> bool:
    descriptor = PROVIDER_CATALOG[provider_model.provider]
    return (
        "thinking" in descriptor.capabilities
        and smoke_config.settings.resolve_thinking("claude-sonnet-4-5-20250929")
    )


def _scenario_text_multiturn(
    smoke_config: SmokeConfig, provider_model: ProviderModel
) -> None:
    with _server_for_provider(smoke_config, provider_model, "text") as server:
        driver = ConversationDriver(server, smoke_config)
        first = driver.ask("Reply with one short sentence.")
        second = driver.ask("Reply with a different short sentence.")
    assert_product_stream(first.events)
    assert_product_stream(second.events)


def _scenario_adaptive_thinking_history(
    smoke_config: SmokeConfig, provider_model: ProviderModel
) -> None:
    payload = {
        "model": "claude-opus-4-7",
        "max_tokens": 256,
        "messages": [
            {"role": "user", "content": "hello"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "unsigned hidden thought"},
                    {"type": "redacted_thinking", "data": "opaque"},
                    {"type": "text", "text": "Hello."},
                ],
            },
            {"role": "user", "content": "Reply with one short sentence."},
        ],
        "thinking": {"type": "adaptive", "budget_tokens": 1024},
    }
    with _server_for_provider(smoke_config, provider_model, "adaptive") as server:
        turn = ConversationDriver(server, smoke_config).stream(payload)
    assert_product_stream(turn.events)


def _scenario_interleaved_history(
    smoke_config: SmokeConfig, provider_model: ProviderModel
) -> None:
    payload = {
        "model": "claude-sonnet-4-5-20250929",
        "max_tokens": 256,
        "messages": [
            {"role": "user", "content": "Use the tool."},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Need to inspect first."},
                    {"type": "text", "text": "I will call the tool."},
                    {
                        "type": "tool_use",
                        "id": "toolu_interleaved",
                        "name": "echo_smoke",
                        "input": {"value": "FCC_INTERLEAVED"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_interleaved",
                        "content": "FCC_INTERLEAVED",
                    }
                ],
            },
        ],
        "tools": [echo_tool_schema()],
        "thinking": {"type": "adaptive"},
    }
    with _server_for_provider(smoke_config, provider_model, "interleaved") as server:
        turn = ConversationDriver(server, smoke_config).stream(payload)
    assert_product_stream(turn.events)


def _scenario_tool_use_then_text_in_history(
    smoke_config: SmokeConfig, provider_model: ProviderModel
) -> None:
    tool_id = "toolu_206_smoke"
    payload = {
        "model": "claude-sonnet-4-5-20250929",
        "max_tokens": 256,
        "messages": [
            {"role": "user", "content": "We will use echo_smoke once in this session."},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": tool_id,
                        "name": "echo_smoke",
                        "input": {"value": "FCC_206_SMOKE"},
                    },
                    {
                        "type": "text",
                        "text": "Narration after the tool call (issue #206 shape).",
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": "FCC_206_SMOKE",
                    },
                ],
            },
            {
                "role": "user",
                "content": "Reply in one short sentence: did you see the echo value?",
            },
        ],
        "tools": [echo_tool_schema()],
    }
    with _server_for_provider(smoke_config, provider_model, "tool-206") as server:
        turn = ConversationDriver(server, smoke_config).stream(payload)
    assert_product_stream(turn.events)


def _scenario_tool_result_continuation(
    smoke_config: SmokeConfig, provider_model: ProviderModel
) -> None:
    first_payload = {
        "model": "claude-sonnet-4-5-20250929",
        "max_tokens": 256,
        "messages": [
            {"role": "user", "content": "Use echo_smoke once with value FCC_TOOL."}
        ],
        "tools": [echo_tool_schema()],
        "tool_choice": {"type": "tool", "name": "echo_smoke"},
        "thinking": {"type": "adaptive"},
    }
    with _server_for_provider(smoke_config, provider_model, "tool") as server:
        driver = ConversationDriver(server, smoke_config)
        first = driver.stream(first_payload)
        tool_uses = tool_use_blocks(first.events)
        assert tool_uses, "provider did not emit a tool_use block"
        tool_use = tool_uses[0]
        second_payload = {
            "model": "claude-sonnet-4-5-20250929",
            "max_tokens": 256,
            "messages": [
                first_payload["messages"][0],
                {"role": "assistant", "content": first.assistant_content},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use["id"],
                            "content": "FCC_TOOL",
                        }
                    ],
                },
            ],
            "tools": [echo_tool_schema()],
        }
        second = driver.stream(second_payload)
    assert_product_stream(first.events)
    assert_product_stream(second.events)


def _scenario_reasoning_tool_continuation(
    smoke_config: SmokeConfig, provider_model: ProviderModel
) -> None:
    payload = {
        "model": "claude-sonnet-4-5-20250929",
        "max_tokens": 256,
        "messages": [
            {"role": "user", "content": "Use echo_smoke once with value FCC_TOOL."},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Need to return the echo result."},
                    {
                        "type": "tool_use",
                        "id": "toolu_reasoning_smoke",
                        "name": "echo_smoke",
                        "input": {"value": "FCC_TOOL"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_reasoning_smoke",
                        "content": "FCC_TOOL",
                    }
                ],
            },
        ],
        "tools": [echo_tool_schema()],
        "thinking": {"type": "adaptive"},
    }
    with _server_for_provider(smoke_config, provider_model, "reasoning-tool") as server:
        turn = ConversationDriver(server, smoke_config).stream(payload)
    assert_product_stream(turn.events)


def _scenario_disconnect(
    smoke_config: SmokeConfig, provider_model: ProviderModel
) -> None:
    with _server_for_provider(smoke_config, provider_model, "disconnect") as server:
        with httpx.stream(
            "POST",
            f"{server.base_url}/v1/messages",
            headers=auth_headers(),
            json={
                "model": "fcc-smoke-default",
                "max_tokens": 512,
                "messages": [{"role": "user", "content": smoke_config.prompt}],
            },
            timeout=smoke_config.timeout_s,
        ) as response:
            assert response.status_code == 200, response.read()
            for _line in response.iter_lines():
                break
        health = httpx.get(f"{server.base_url}/health", timeout=5)
        assert health.status_code == 200
        followup = ConversationDriver(server, smoke_config).ask(
            "Reply with one short sentence."
        )
    skip_if_upstream_unavailable_events(followup.events)
    assert_product_stream(followup.events)


def _server_for_provider(
    smoke_config: SmokeConfig, provider_model: ProviderModel, name: str
):
    return SmokeServerDriver(
        smoke_config,
        name=f"product-provider-{provider_model.provider}-{name}",
        env_overrides={
            "MODEL": provider_model.full_model,
            "MESSAGING_PLATFORM": "none",
        },
    ).run()
