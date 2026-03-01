"""TTS service — wraps provider with retry + circuit breaker."""

from __future__ import annotations

import logging
from typing import Optional

from app.config import Settings
from app.providers.base import BaseTTSProvider
from app.resilience.circuit_breaker import CircuitBreaker
from app.resilience.retry_manager import retry_async

logger = logging.getLogger(__name__)


class TTSService:
    def __init__(
        self,
        settings: Settings,
        provider: BaseTTSProvider,
    ) -> None:
        self._settings = settings
        self._provider = provider
        self._cb = CircuitBreaker(
            name="tts",
            failure_threshold=settings.CB_FAILURE_THRESHOLD,
            recovery_timeout=settings.CB_RECOVERY_TIMEOUT,
        )

    async def synthesize(
        self, text: str
    ) -> tuple[Optional[bytes], Optional[str]]:
        """
        Returns (audio_bytes, error).  Exactly one will be non-None.
        TTS failure is NEVER fatal — the caller falls back to text_only.
        """
        try:
            result = await retry_async(
                self._cb.call,
                self._provider.synthesize,
                text,
                max_retries=self._settings.MAX_RETRIES_TTS,
                operation_name="TTS",
            )
            if result.error:
                return None, result.error
            return result.audio, None
        except Exception as exc:
            logger.error("TTS service failure: %s", exc)
            return None, str(exc)
