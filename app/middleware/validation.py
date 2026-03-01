"""Payload validation middleware logic."""

from __future__ import annotations

from app.config import Settings


class PayloadValidationError(Exception):
    pass


class PayloadValidator:
    def __init__(self, settings: Settings) -> None:
        self._max_text = settings.MAX_TEXT_LENGTH
        self._max_audio_mb = settings.MAX_AUDIO_SIZE_MB

    def validate_text(self, text: str) -> None:
        if len(text) > self._max_text:
            raise PayloadValidationError(
                f"Text length {len(text)} exceeds maximum {self._max_text}"
            )
        if not text.strip():
            raise PayloadValidationError("Text input cannot be empty or whitespace")

    def validate_audio_base64_size(self, audio_b64: str) -> None:
        estimated_bytes = len(audio_b64) * 3 / 4
        max_bytes = self._max_audio_mb * 1024 * 1024
        if estimated_bytes > max_bytes:
            raise PayloadValidationError(
                f"Audio payload too large (~{estimated_bytes / (1024*1024):.1f} MB), "
                f"max {self._max_audio_mb:.1f} MB"
            )
