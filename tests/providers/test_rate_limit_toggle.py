import asyncio
import time
from unittest.mock import AsyncMock, patch

import openai
import pytest

from providers.rate_limit import GlobalRateLimiter


@pytest.mark.asyncio
async def test_rate_limiter_disabled_bypasses_wait():
    """Verify that when disabled, wait_if_blocked returns immediately even if 'blocked'."""
    limiter = GlobalRateLimiter(rate_limit=1, rate_window=60, enabled=False)
    limiter.set_blocked(30)  # Block for 30s
    
    start = time.monotonic()
    waited = await limiter.wait_if_blocked()
    end = time.monotonic()
    
    assert not waited
    assert end - start < 0.1  # Should be near-instant


@pytest.mark.asyncio
async def test_rate_limiter_disabled_bypasses_concurrency():
    """Verify that when disabled, concurrency_slot does not block."""
    # Limiter with max_concurrency=1
    limiter = GlobalRateLimiter(max_concurrency=1, enabled=False)
    
    # Acquire two slots simultaneously
    async with limiter.concurrency_slot():
        async with limiter.concurrency_slot():
            # Should reach here without blocking
            pass


@pytest.mark.asyncio
async def test_rate_limiter_disabled_bypasses_retry():
    """Verify that when disabled, execute_with_retry does not retry on 429."""
    limiter = GlobalRateLimiter(enabled=False)
    
    mock_fn = AsyncMock(side_effect=openai.RateLimitError("Rate limit exceeded", response=AsyncMock(), body={}))
    
    with pytest.raises(openai.RateLimitError):
        await limiter.execute_with_retry(mock_fn)
    
    # Should only be called once
    assert mock_fn.call_count == 1


@pytest.mark.asyncio
async def test_rate_limiter_enabled_blocks():
    """Verify that when enabled (default), it still blocks."""
    limiter = GlobalRateLimiter(rate_limit=1, rate_window=0.1, enabled=True)
    
    # Use first slot
    await limiter.wait_if_blocked()
    
    start = time.monotonic()
    # This should wait for the window to reset
    await limiter.wait_if_blocked()
    end = time.monotonic()
    
    assert end - start >= 0.1
