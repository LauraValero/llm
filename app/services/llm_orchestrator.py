"""LLM Orchestrator — routes requests through the configured provider."""

from __future__ import annotations

import logging
from typing import Any, Optional

from app.config import Settings
from app.providers.base import BaseLLMProvider, LLMResponse
from app.resilience.circuit_breaker import CircuitBreaker
from app.resilience.retry_manager import retry_async

logger = logging.getLogger(__name__)


class LLMOrchestrator:
    def __init__(
        self,
        settings: Settings,
        provider: BaseLLMProvider,
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        self._settings = settings
        self._provider = provider
        self._tools = tools
        self._cb = CircuitBreaker(
            name="llm",
            failure_threshold=settings.CB_FAILURE_THRESHOLD,
            recovery_timeout=settings.CB_RECOVERY_TIMEOUT,
        )

    async def generate(
        self,
        messages: list[dict[str, Any]],
        text: Optional[str] = None,
        audio: Optional[bytes] = None,
    ) -> LLMResponse:
        try:
            response: LLMResponse = await retry_async(
                self._cb.call,
                self._provider.chat,
                messages,
                text,
                audio,
                self._tools if self._settings.LLM_SUPPORTS_TOOLS else None,
                max_retries=self._settings.MAX_RETRIES_LLM,
                operation_name="LLM",
            )
            if response.error:
                logger.error("LLM returned error: %s", response.error)
            return response
        except Exception as exc:
            logger.error("LLM orchestrator failure: %s", exc)
            return LLMResponse(error=str(exc))

    async def generate_with_tool_results(
        self,
        messages: list[dict[str, Any]],
        tool_results: list[dict[str, Any]],
        original_response: LLMResponse,
    ) -> LLMResponse:
        try:
            response: LLMResponse = await retry_async(
                self._cb.call,
                self._provider.chat_with_tool_results,
                messages,
                tool_results,
                original_response,
                max_retries=self._settings.MAX_RETRIES_LLM,
                operation_name="LLM_tool_followup",
            )
            if response.error:
                logger.error("LLM tool follow-up error: %s", response.error)
            return response
        except Exception as exc:
            logger.error("LLM tool follow-up failure: %s", exc)
            return LLMResponse(error=str(exc))
