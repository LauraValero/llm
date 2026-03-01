"""Per-user token-bucket rate limiter."""

from __future__ import annotations

import asyncio
import time
from typing import Optional


class RateLimitExceeded(Exception):
    def __init__(self, retry_after: float) -> None:
        self.retry_after = retry_after
        super().__init__(f"Rate limit exceeded. Retry after {retry_after:.1f}s")


class TokenBucket:
    """Simple token-bucket rate limiter for a single user."""

    def __init__(self, rate: float, burst: int) -> None:
        self._rate = rate            # tokens per second
        self._burst = burst
        self._tokens = float(burst)
        self._last_refill = time.monotonic()

    def try_consume(self) -> Optional[float]:
        """
        Try to consume one token.
        Returns None on success, or seconds until a token is available.
        """
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._burst, self._tokens + elapsed * self._rate)
        self._last_refill = now

        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return None
        return (1.0 - self._tokens) / self._rate


class RateLimiter:
    """Manages per-user rate limiters."""

    def __init__(self, requests_per_minute: int, burst: int) -> None:
        self._rate = requests_per_minute / 60.0
        self._burst = burst
        self._buckets: dict[str, TokenBucket] = {}
        self._lock = asyncio.Lock()

    async def check(self, user_id: str) -> None:
        async with self._lock:
            bucket = self._buckets.get(user_id)
            if bucket is None:
                bucket = TokenBucket(self._rate, self._burst)
                self._buckets[user_id] = bucket

        retry_after = bucket.try_consume()
        if retry_after is not None:
            raise RateLimitExceeded(retry_after)

    async def cleanup_inactive(self, max_age: float = 300.0) -> None:
        """Remove buckets that haven't been used in max_age seconds."""
        now = time.monotonic()
        async with self._lock:
            expired = [
                uid
                for uid, b in self._buckets.items()
                if (now - b._last_refill) > max_age
            ]
            for uid in expired:
                del self._buckets[uid]
