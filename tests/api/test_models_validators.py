from api.models.anthropic import Message, MessagesRequest, TokenCountRequest


def test_messages_request_parses_without_model_mapping_side_effects():
    request = MessagesRequest(
        model="claude-3-opus",
        max_tokens=100,
        messages=[Message(role="user", content="hello")],
    )

    assert request.model == "claude-3-opus"


def test_messages_request_ignores_internal_routing_fields_when_supplied():
    request = MessagesRequest.model_validate(
        {
            "model": "target-model",
            "original_model": "claude-3-opus",
            "resolved_provider_model": "nvidia_nim/target-model",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "hello"}],
        }
    )

    assert request.model == "target-model"
    assert "original_model" not in request.model_dump()
    assert "resolved_provider_model" not in request.model_dump()


def test_token_count_request_parses_without_model_mapping_side_effects():
    request = TokenCountRequest(
        model="claude-3-sonnet", messages=[Message(role="user", content="hello")]
    )

    assert request.model == "claude-3-sonnet"


def test_messages_request_preserves_thinking_signature():
    request = MessagesRequest.model_validate(
        {
            "model": "claude-3-opus",
            "max_tokens": 100,
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": "signed thought",
                            "signature": "sig_123",
                        }
                    ],
                }
            ],
        }
    )

    dumped = request.model_dump(exclude_none=True)

    assert dumped["messages"][0]["content"][0]["signature"] == "sig_123"


def test_messages_request_preserves_native_thinking_budget():
    request = MessagesRequest.model_validate(
        {
            "model": "claude-3-opus",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "think hard"}],
            "thinking": {"type": "enabled", "budget_tokens": 4096},
        }
    )

    dumped = request.model_dump(exclude_none=True)

    assert dumped["thinking"]["type"] == "enabled"
    assert dumped["thinking"]["budget_tokens"] == 4096


def test_messages_request_accepts_adaptive_thinking_type():
    request = MessagesRequest.model_validate(
        {
            "model": "claude-3-opus",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "hello"}],
            "thinking": {"type": "adaptive"},
        }
    )

    dumped = request.model_dump(exclude_none=True)

    assert dumped["thinking"]["type"] == "adaptive"


def test_messages_request_accepts_anthropic_server_tool_without_input_schema():
    request = MessagesRequest.model_validate(
        {
            "model": "claude-opus-4-7",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "search"}],
            "tools": [{"type": "web_search_20250305", "name": "web_search"}],
        }
    )

    dumped = request.model_dump(exclude_none=True)

    assert dumped["tools"] == [{"name": "web_search", "type": "web_search_20250305"}]


def test_messages_request_accepts_redacted_thinking_blocks():
    request = MessagesRequest.model_validate(
        {
            "model": "claude-3-opus",
            "max_tokens": 100,
            "messages": [
                {
                    "role": "assistant",
                    "content": [{"type": "redacted_thinking", "data": "opaque"}],
                }
            ],
        }
    )

    dumped = request.model_dump(exclude_none=True)

    assert dumped["messages"][0]["content"][0] == {
        "type": "redacted_thinking",
        "data": "opaque",
    }
