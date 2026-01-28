"""Reusable async rate limiter with sliding window."""

import asyncio
import time
import threading
from collections import deque
from typing import Deque


class SlidingWindowRateLimiter:
    """
    Async rate limiter using sliding window algorithm.

    Thread-safe for use across multiple async contexts.
    """

    def __init__(
        self,
        rate_limit: int = 40,
        window_seconds: int = 60,
        max_retries: int = 120,
    ):
        self.rate_limit = rate_limit
        self.window_seconds = window_seconds
        self.max_retries = max_retries
        self._timestamps: Deque[float] = deque()
        self._lock = threading.Condition()

    async def acquire(self) -> None:
        """
        Acquire a rate limit slot, waiting if necessary.

        Uses exponential backoff up to max_retries attempts.
        """
        for _ in range(self.max_retries):
            now = time.time()

            with self._lock:
                # Remove expired timestamps
                while self._timestamps and now - self._timestamps[0] > self.window_seconds:
                    self._timestamps.popleft()

                # Check if we can proceed
                if len(self._timestamps) < self.rate_limit:
                    self._timestamps.append(now)
                    return

                # Calculate wait time
                wait_time = self._timestamps[0] + self.window_seconds - now

            if wait_time <= 0:
                continue

            await asyncio.sleep(min(wait_time, 1.0))

        # Fallback: allow request after max retries
        with self._lock:
            self._timestamps.append(time.time())

    def reset(self) -> None:
        """Clear all timestamps."""
        with self._lock:
            self._timestamps.clear()

    @property
    def current_count(self) -> int:
        """Current number of requests in the window."""
        now = time.time()
        with self._lock:
            while self._timestamps and now - self._timestamps[0] > self.window_seconds:
                self._timestamps.popleft()
            return len(self._timestamps)
