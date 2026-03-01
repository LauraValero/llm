"""Async circuit breaker to avoid cascading failures."""

from __future__ import annotations

import asyncio
import logging
import time
from enum import Enum
from typing import Any, Awaitable, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(Exception):
    """Raised when a call is attempted against an open circuit."""


class CircuitBreaker:
    """
    Per-service circuit breaker.

    - CLOSED: requests flow normally; consecutive failures are counted.
    - OPEN: all requests are immediately rejected for `recovery_timeout` seconds.
    - HALF_OPEN: one probe request is allowed; success resets, failure re-opens.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
    ) -> None:
        self.name = name
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: float = 0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.OPEN:
            if time.monotonic() - self._last_failure_time >= self._recovery_timeout:
                return CircuitState.HALF_OPEN
        return self._state

    async def call(
        self,
        fn: Callable[..., Awaitable[T]],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        current = self.state

        if current == CircuitState.OPEN:
            logger.warning("Circuit %s is OPEN – rejecting call", self.name)
            raise CircuitOpenError(f"Circuit '{self.name}' is open")

        async with self._lock:
            try:
                result = await fn(*args, **kwargs)
                self._on_success()
                return result
            except Exception as exc:
                self._on_failure()
                raise exc

    def _on_success(self) -> None:
        self._failure_count = 0
        if self._state != CircuitState.CLOSED:
            logger.info("Circuit %s recovered – now CLOSED", self.name)
        self._state = CircuitState.CLOSED

    def _on_failure(self) -> None:
        self._failure_count += 1
        self._last_failure_time = time.monotonic()
        if self._failure_count >= self._failure_threshold:
            self._state = CircuitState.OPEN
            logger.error(
                "Circuit %s tripped OPEN after %d failures",
                self.name,
                self._failure_count,
            )

    def reset(self) -> None:
        self._state = CircuitState.CLOSED
        self._failure_count = 0
