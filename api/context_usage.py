"""Session-level context usage tracking from Anthropic-style usage fields."""

from __future__ import annotations

import json
from dataclasses import dataclass
from threading import Lock

from loguru import logger


@dataclass(frozen=True, slots=True)
class ContextUsageSnapshot:
    """Computed context usage values for one model after a request."""

    label: str
    limit_tokens: int
    cumulative_input_tokens: int
    percent_used: int
    ratio_used: float
    bar: str


_KNOWN_MODEL_LIMITS: tuple[tuple[tuple[str, ...], str, int], ...] = (
    (("glm-5.1", "glm5.1"), "GLM-5.1", 200_000),
    (("kimi-k2.5", "k2.5"), "K2.5", 256_000),
    (("glm4.7", "glm-4.7"), "GLM-4.7", 128_000),
    (("glm5", "glm-5"), "GLM-5", 128_000),
)


def _find_model_limit(model_name: str) -> tuple[str, int] | None:
    lowered = model_name.strip().lower()
    for aliases, display_label, limit_tokens in _KNOWN_MODEL_LIMITS:
        if any(alias in lowered for alias in aliases):
            return display_label, limit_tokens
    return None


def _format_tokens_k(tokens: int) -> str:
    if tokens < 1_000:
        return str(tokens)
    return f"{round(tokens / 1_000):.0f}K"


def _build_context_bar(ratio_used: float, width: int = 10) -> str:
    clamped = max(0.0, min(ratio_used, 1.0))
    filled = max(0, min(width, round(clamped * width)))
    return "█" * filled + "░" * (width - filled)


def extract_usage_input_tokens_from_sse_event(event: str) -> int | None:
    """Extract usage.input_tokens from one SSE event string when present."""
    for raw_line in event.splitlines():
        line = raw_line.strip()
        if not line.startswith("data: "):
            continue
        payload = line.removeprefix("data: ").strip()
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            continue

        direct_usage = parsed.get("usage")
        if isinstance(direct_usage, dict):
            value = direct_usage.get("input_tokens")
            if isinstance(value, int):
                return value

        message = parsed.get("message")
        if isinstance(message, dict):
            nested_usage = message.get("usage")
            if isinstance(nested_usage, dict):
                value = nested_usage.get("input_tokens")
                if isinstance(value, int):
                    return value
    return None


class SessionContextUsageTracker:
    """Track cumulative per-model input token usage during proxy process lifetime."""

    def __init__(self, warn_threshold: float):
        self._warn_threshold = warn_threshold
        self._cumulative_input_by_model: dict[str, int] = {}
        self._lock = Lock()

    def record_usage(
        self, *, model_name: str, input_tokens: int
    ) -> ContextUsageSnapshot | None:
        """Record one request's input tokens and emit context logs when model limit is known."""
        if input_tokens < 0:
            return None

        model_limit = _find_model_limit(model_name)
        if model_limit is None:
            return None
        label, limit_tokens = model_limit

        with self._lock:
            prior = self._cumulative_input_by_model.get(label, 0)
            updated = prior + input_tokens
            self._cumulative_input_by_model[label] = updated

        ratio_used = updated / limit_tokens if limit_tokens > 0 else 0.0
        percent_used = round(ratio_used * 100)
        bar = _build_context_bar(ratio_used)

        logger.info(
            "[Context: {} {}% | {}/{} tokens | {}]",
            bar,
            percent_used,
            _format_tokens_k(updated),
            _format_tokens_k(limit_tokens),
            label,
        )

        prior_ratio = prior / limit_tokens if limit_tokens > 0 else 0.0
        crossed_threshold = prior_ratio < self._warn_threshold <= ratio_used
        if crossed_threshold:
            logger.warning(
                "Context usage at {:.1f}% for {} ({} / {} tokens). Run /compact soon.",
                ratio_used * 100.0,
                label,
                updated,
                limit_tokens,
            )

        return ContextUsageSnapshot(
            label=label,
            limit_tokens=limit_tokens,
            cumulative_input_tokens=updated,
            percent_used=percent_used,
            ratio_used=ratio_used,
            bar=bar,
        )
