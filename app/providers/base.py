"""Abstract interfaces for LLM / STT / TTS providers.

Business logic MUST NOT depend on concrete providers — only on these ABCs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


# ------------------------------------------------------------------
# LLM
# ------------------------------------------------------------------

@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    raw: Any = None
    error: Optional[str] = None


class BaseLLMProvider(ABC):
    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, Any]],
        text: Optional[str] = None,
        audio: Optional[bytes] = None,
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> LLMResponse: ...

    @abstractmethod
    async def chat_with_tool_results(
        self,
        messages: list[dict[str, Any]],
        tool_results: list[dict[str, Any]],
        original_response: LLMResponse,
    ) -> LLMResponse: ...


# ------------------------------------------------------------------
# STT
# ------------------------------------------------------------------

@dataclass
class STTResponse:
    text: str = ""
    error: Optional[str] = None


class BaseSTTProvider(ABC):
    @abstractmethod
    async def transcribe(self, audio: bytes, mime_type: str = "audio/webm") -> STTResponse: ...


# ------------------------------------------------------------------
# TTS
# ------------------------------------------------------------------

@dataclass
class TTSResponse:
    audio: bytes = b""
    error: Optional[str] = None


class BaseTTSProvider(ABC):
    @abstractmethod
    async def synthesize(self, text: str) -> TTSResponse: ...
