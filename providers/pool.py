"""Intelligent API key pooling and rotation."""

import random
import time
from dataclasses import dataclass, field

from loguru import logger

from providers.rate_limit import GlobalRateLimiter


@dataclass
class KeyStatus:
    """Status and rate limiter for a single API key."""

    key: str
    limiter: GlobalRateLimiter
    last_used: float = 0
    failure_count: int = 0


class KeyPool:
    """Manages a pool of API keys with intelligent rotation and rate-limit awareness."""

    def __init__(
        self,
        keys: list[str],
        *,
        scope: str,
        rate_limit: int | None = None,
        rate_window: float | None = None,
        max_concurrency: int = 5,
    ):
        if not keys:
            raise ValueError("KeyPool requires at least one key.")

        self._keys = [
            KeyStatus(
                key=key,
                limiter=GlobalRateLimiter(
                    rate_limit=rate_limit or 40,
                    rate_window=rate_window or 60.0,
                    max_concurrency=max_concurrency,
                ),
            )
            for key in keys
        ]
        self._scope = scope
        logger.info(
            "KeyPool initialized for scope '{}' with {} keys", scope, len(self._keys)
        )

    def get_best_key_status(self) -> KeyStatus:
        """Select the best available key from the pool.

        Prefers keys that are not reactively blocked. Among available keys,
        picks the one used least recently. If all are blocked, picks the one
        that will be unblocked soonest.
        """
        now = time.monotonic()
        available = [k for k in self._keys if not k.limiter.is_blocked()]

        if available:
            # Round-robin-ish: pick the one least recently used
            selected = min(available, key=lambda k: k.last_used)
            selected.last_used = now
            return selected

        # All keys are blocked, pick the one with the shortest remaining wait
        selected = min(self._keys, key=lambda k: k.limiter.remaining_wait())
        selected.last_used = now
        return selected

    @property
    def keys(self) -> list[str]:
        """Return all keys in the pool."""
        return [k.key for k in self._keys]
