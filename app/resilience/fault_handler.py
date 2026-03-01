"""Centralized fault-handling policies driven by .env configuration."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from app.config import (
    LlmFaultAction,
    Settings,
    SttFaultAction,
    ToolFaultAction,
    TtsFaultAction,
)

logger = logging.getLogger(__name__)


@dataclass
class FaultResult:
    should_continue: bool
    fallback_text: Optional[str] = None
    request_retry: bool = False


class FaultHandler:
    def __init__(self, settings: Settings) -> None:
        self._s = settings

    def handle_stt_fault(self, error: Exception) -> FaultResult:
        logger.error("STT fault: %s", error)
        action = self._s.STT_FAULT_ACTION
        if action == SttFaultAction.REQUEST_AUDIO_RETRY:
            return FaultResult(should_continue=False, request_retry=True)
        return FaultResult(should_continue=False)

    def handle_tool_fault(self, tool_name: str, error: Exception) -> FaultResult:
        logger.error("Tool '%s' fault: %s", tool_name, error)
        action = self._s.TOOL_FAULT_ACTION
        if action == ToolFaultAction.FALLBACK_RESPONSE:
            return FaultResult(
                should_continue=True,
                fallback_text=(
                    f"No pude ejecutar la herramienta '{tool_name}', "
                    "pero continuaré con la información disponible."
                ),
            )
        return FaultResult(should_continue=True)

    def handle_llm_fault(self, error: Exception) -> FaultResult:
        logger.error("LLM fault: %s", error)
        return FaultResult(
            should_continue=True,
            fallback_text=self._s.SAFE_RESPONSE_MESSAGE,
        )

    def handle_tts_fault(self, error: Exception) -> FaultResult:
        logger.error("TTS fault: %s", error)
        return FaultResult(should_continue=True)
