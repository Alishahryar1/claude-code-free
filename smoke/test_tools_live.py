from __future__ import annotations

import pytest

from smoke.lib.config import SmokeConfig
from smoke.lib.http import collect_message_stream, message_payload
from smoke.lib.server import start_server
from smoke.lib.sse import assert_anthropic_stream_contract, has_tool_use

pytestmark = [pytest.mark.live, pytest.mark.smoke_target("tools")]


def test_live_tool_use_when_configured_model_supports_tools(
    smoke_config: SmokeConfig,
) -> None:
    models = smoke_config.provider_models()
    if not models:
        pytest.skip("no configured provider model available for tool-use smoke")
    provider_model = models[0]

    payload = message_payload(
        "Use the echo_smoke tool once with value FCC_SMOKE_TOOL.",
        model="fcc-smoke-default",
        max_tokens=256,
        extra={
            "tools": [
                {
                    "name": "echo_smoke",
                    "description": "Echo a test value.",
                    "input_schema": {
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                        "required": ["value"],
                    },
                }
            ],
            "tool_choice": {"type": "tool", "name": "echo_smoke"},
        },
    )

    with start_server(
        smoke_config,
        env_overrides={
            "MODEL": provider_model.full_model,
            "MESSAGING_PLATFORM": "none",
        },
        name="tools",
    ) as server:
        events = collect_message_stream(server, payload, smoke_config)
    assert_anthropic_stream_contract(events)
    assert has_tool_use(events), "model did not emit a tool_use block"
