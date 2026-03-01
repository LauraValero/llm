"""Configurable async retry with exponential back-off."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

DEFAULT_BASE_DELAY = 1.0
DEFAULT_MAX_DELAY = 16.0
DEFAULT_BACKOFF_FACTOR = 2.0


async def retry_async(
    fn: Callable[..., Awaitable[T]],
    *args: Any,
    max_retries: int = 3,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
    operation_name: str = "operation",
    **kwargs: Any,
) -> T:
    """
    Retry an async callable with exponential back-off.

    Returns the result on success or raises the last exception after exhausting
    all retries.
    """
    last_exc: Exception | None = None
    delay = base_delay

    for attempt in range(1, max_retries + 2):  # +1 for the initial attempt
        try:
            return await fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if attempt > max_retries:
                logger.error(
                    "%s failed after %d attempts: %s",
                    operation_name,
                    attempt,
                    exc,
                )
                raise
            logger.warning(
                "%s attempt %d/%d failed: %s — retrying in %.1fs",
                operation_name,
                attempt,
                max_retries + 1,
                exc,
                delay,
            )
            await asyncio.sleep(delay)
            delay = min(delay * backoff_factor, max_delay)

    raise last_exc  # type: ignore[misc]
