"""Feature inventory used by the local smoke suite.

This file is intentionally explicit: advertised features should not exist only
in README prose without at least one smoke check or a documented manual gap.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FeatureSmoke:
    feature_id: str
    title: str
    mode: str
    checks: tuple[str, ...]


README_FEATURES: tuple[str, ...] = (
    "zero_cost_provider_access",
    "drop_in_claude_code_replacement",
    "provider_matrix",
    "per_model_mapping",
    "thinking_token_support",
    "heuristic_tool_parser",
    "request_optimization",
    "smart_rate_limiting",
    "discord_telegram_bot",
    "subagent_control",
    "extensible_provider_platform_abcs",
    "optional_authentication",
    "vscode_extension",
    "intellij_extension",
    "voice_notes",
)


FEATURE_SMOKES: tuple[FeatureSmoke, ...] = (
    FeatureSmoke(
        "zero_cost_provider_access",
        "Configured provider accepts a real prompt",
        "live",
        ("test_configured_provider_models_stream_successfully",),
    ),
    FeatureSmoke(
        "drop_in_claude_code_replacement",
        "Claude-compatible routes and CLI environment work",
        "live",
        (
            "test_probe_and_models_routes",
            "test_claude_cli_prompt_when_available",
            "test_vscode_and_jetbrains_shaped_requests",
        ),
    ),
    FeatureSmoke(
        "provider_matrix",
        "All configured provider prefixes can be exercised",
        "live",
        ("test_configured_provider_models_stream_successfully",),
    ),
    FeatureSmoke(
        "per_model_mapping",
        "Opus, Sonnet, Haiku, and fallback mappings are visible",
        "live",
        ("test_model_mapping_configuration_is_consistent",),
    ),
    FeatureSmoke(
        "thinking_token_support",
        "Thinking streams and suppression are contract-tested",
        "contract",
        (
            "test_interleaved_thinking_text_blocks_are_valid",
            "test_split_think_tags_preserve_text_and_thinking",
            "test_enable_thinking_false_suppresses_reasoning_only",
        ),
    ),
    FeatureSmoke(
        "heuristic_tool_parser",
        "Text tool calls become structured tool_use blocks",
        "contract",
        ("test_thinking_tool_text_and_transcript_order_contract",),
    ),
    FeatureSmoke(
        "request_optimization",
        "Fast-path local optimizations respond without providers",
        "live",
        ("test_optimization_fast_paths_do_not_need_provider",),
    ),
    FeatureSmoke(
        "smart_rate_limiting",
        "Concurrency/disconnect and retry-sensitive paths are checked",
        "live",
        ("test_client_disconnect_mid_stream_does_not_crash_server",),
    ),
    FeatureSmoke(
        "discord_telegram_bot",
        "Messaging credentials, send/edit/delete, and transcript behavior",
        "live_or_interactive",
        (
            "test_telegram_bot_api_permissions",
            "test_discord_bot_api_permissions",
            "test_thinking_tool_text_and_transcript_order_contract",
        ),
    ),
    FeatureSmoke(
        "subagent_control",
        "Task tool calls do not run in the background",
        "contract",
        ("test_task_tool_arguments_force_foreground_execution",),
    ),
    FeatureSmoke(
        "extensible_provider_platform_abcs",
        "Provider/platform registries expose expected built-ins",
        "contract",
        ("test_provider_and_platform_registries_include_advertised_builtins",),
    ),
    FeatureSmoke(
        "optional_authentication",
        "Anthropic-style auth headers are accepted and enforced",
        "live",
        ("test_auth_token_is_enforced_for_all_supported_header_shapes",),
    ),
    FeatureSmoke(
        "vscode_extension",
        "VS Code-shaped beta requests work against the proxy",
        "live",
        ("test_vscode_and_jetbrains_shaped_requests",),
    ),
    FeatureSmoke(
        "intellij_extension",
        "JetBrains/ACP-shaped environment requests work against the proxy",
        "live",
        ("test_vscode_and_jetbrains_shaped_requests",),
    ),
    FeatureSmoke(
        "voice_notes",
        "Configured transcription backend can process an audio fixture",
        "live_or_skip",
        ("test_voice_transcription_backend_when_explicitly_enabled",),
    ),
)


def smoke_ids() -> set[str]:
    """Return feature IDs covered by the smoke manifest."""
    return {feature.feature_id for feature in FEATURE_SMOKES}
