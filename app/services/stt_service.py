"""STT service — wraps provider with retry + circuit breaker."""

from __future__ import annotations

import logging
from typing import Optional

from app.config import Settings
from app.providers.base import BaseSTTProvider
from app.resilience.circuit_breaker import CircuitBreaker
from app.resilience.retry_manager import retry_async

logger = logging.getLogger(__name__)


class STTService:
    def __init__(
        self,
        settings: Settings,
        provider: BaseSTTProvider,
    ) -> None:
        self._settings = settings
        self._provider = provider
        self._cb = CircuitBreaker(
            name="stt",
            failure_threshold=settings.CB_FAILURE_THRESHOLD,
            recovery_timeout=settings.CB_RECOVERY_TIMEOUT,
        )

    async def transcribe(
        self, audio: bytes, mime_type: str = "audio/webm"
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Returns (transcript, error).  Exactly one will be non-None.
        """
        try:
            result = await retry_async(
                self._cb.call,
                self._provider.transcribe,
                audio,
                mime_type,
                max_retries=self._settings.MAX_RETRIES_STT,
                operation_name="STT",
            )
            if result.error:
                return None, result.error
            return result.text, None
        except Exception as exc:
            logger.error("STT service failure: %s", exc)
            return None, str(exc)
