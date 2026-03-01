"""Audio ingestion: validation and decoding of inbound audio payloads."""

from __future__ import annotations

import base64
import logging

from app.config import Settings

logger = logging.getLogger(__name__)


class AudioIngestionError(Exception):
    pass


class AudioIngestionService:
    def __init__(self, settings: Settings) -> None:
        self._max_size_bytes = int(settings.MAX_AUDIO_SIZE_MB * 1024 * 1024)
        self._max_duration = settings.MAX_AUDIO_DURATION_SECONDS

    def validate_and_decode(self, audio_base64: str, mime_type: str) -> bytes:
        """
        Decode base64 audio and enforce size limits.

        Duration checks require codec inspection and are deferred to the
        STT provider's pre-processing; here we enforce the raw byte budget.
        """
        try:
            audio_bytes = base64.b64decode(audio_base64, validate=True)
        except Exception as exc:
            raise AudioIngestionError(f"Invalid base64 audio data: {exc}") from exc

        if len(audio_bytes) > self._max_size_bytes:
            size_mb = len(audio_bytes) / (1024 * 1024)
            raise AudioIngestionError(
                f"Audio too large: {size_mb:.1f} MB exceeds limit of "
                f"{self._max_size_bytes / (1024 * 1024):.1f} MB"
            )

        if not mime_type.startswith("audio/"):
            raise AudioIngestionError(f"Invalid MIME type: {mime_type}")

        logger.debug(
            "Audio ingested: %d bytes, mime=%s",
            len(audio_bytes),
            mime_type,
        )
        return audio_bytes
