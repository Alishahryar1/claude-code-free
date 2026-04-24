from __future__ import annotations

import time

import httpx
import pytest

from smoke.lib.config import SmokeConfig, auth_headers
from smoke.lib.http import collect_message_stream, message_payload
from smoke.lib.server import start_server
from smoke.lib.sse import assert_anthropic_stream_contract, text_content

pytestmark = [pytest.mark.live, pytest.mark.smoke_target("providers")]


def test_model_mapping_configuration_is_consistent(smoke_config: SmokeConfig) -> None:
    models = smoke_config.provider_models()
    if not models:
        pytest.skip("no configured provider models with usable credentials/base URLs")
    for provider_model in models:
        assert "/" in provider_model.full_model
        assert provider_model.model_name


def test_configured_provider_models_stream_successfully(
    smoke_config: SmokeConfig,
) -> None:
    models = smoke_config.provider_models()
    if not models:
        pytest.skip("no configured provider models with usable credentials/base URLs")

    failures: list[str] = []
    for provider_model in models:
        try:
            with start_server(
                smoke_config,
                env_overrides={
                    "MODEL": provider_model.full_model,
                    "MESSAGING_PLATFORM": "none",
                },
                name=f"provider-{provider_model.provider}",
            ) as server:
                events = collect_message_stream(
                    server,
                    message_payload(smoke_config.prompt, model="fcc-smoke-default"),
                    smoke_config,
                )
                assert_anthropic_stream_contract(events)
                assert text_content(events).strip(), "provider returned no text"
        except Exception as exc:
            failures.append(
                f"{provider_model.source}={provider_model.full_model}: "
                f"{type(exc).__name__}: {exc}"
            )

    assert not failures, "\n".join(failures)


def test_client_disconnect_mid_stream_does_not_crash_server(
    smoke_config: SmokeConfig,
) -> None:
    models = smoke_config.provider_models()
    if not models:
        pytest.skip("no configured provider model available for disconnect smoke")
    provider_model = models[0]

    with start_server(
        smoke_config,
        env_overrides={
            "MODEL": provider_model.full_model,
            "MESSAGING_PLATFORM": "none",
        },
        name="disconnect",
    ) as server:
        with httpx.stream(
            "POST",
            f"{server.base_url}/v1/messages",
            headers=auth_headers(),
            json=message_payload(smoke_config.prompt, model="fcc-smoke-default"),
            timeout=smoke_config.timeout_s,
        ) as response:
            assert response.status_code == 200, response.read()
            for _line in response.iter_lines():
                break

        time.sleep(0.5)
        health = httpx.get(f"{server.base_url}/health", timeout=5)
        assert health.status_code == 200
