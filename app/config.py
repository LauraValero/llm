from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class SttFaultAction(str, Enum):
    REQUEST_AUDIO_RETRY = "request_audio_retry"
    END_SESSION = "end_session"


class ToolFaultAction(str, Enum):
    CONTINUE_WITHOUT_TOOL = "continue_without_tool"
    FALLBACK_RESPONSE = "fallback_response"


class LlmFaultAction(str, Enum):
    SAFE_RESPONSE = "safe_response"


class TtsFaultAction(str, Enum):
    TEXT_ONLY = "text_only"


class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Server ---
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 4
    DEBUG: bool = False

    # --- Provider selection ---
    LLM_PROVIDER: LLMProvider = LLMProvider.OPENAI
    LLM_MODEL: str = "gpt-4o"
    STT_PROVIDER: LLMProvider = LLMProvider.OPENAI
    STT_MODEL: str = "whisper-1"
    TTS_PROVIDER: LLMProvider = LLMProvider.OPENAI
    TTS_MODEL: str = "tts-1"
    TTS_VOICE: str = "alloy"

    # --- Provider API keys ---
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None

    # --- Model capabilities ---
    LLM_SUPPORTS_AUDIO: bool = False
    LLM_SUPPORTS_TOOLS: bool = True

    # --- UX ---
    SHOW_USER_TRANSCRIPT: bool = True

    # --- Input limits (security) ---
    MAX_AUDIO_DURATION_SECONDS: int = Field(default=120, ge=1)
    MAX_AUDIO_SIZE_MB: float = Field(default=25.0, gt=0)
    MAX_TEXT_LENGTH: int = Field(default=4096, ge=1)

    # --- Retry limits ---
    MAX_RETRIES_STT: int = Field(default=3, ge=0)
    MAX_RETRIES_TOOL: int = Field(default=2, ge=0)
    MAX_RETRIES_LLM: int = Field(default=3, ge=0)
    MAX_RETRIES_TTS: int = Field(default=2, ge=0)

    # --- Fault actions ---
    STT_FAULT_ACTION: SttFaultAction = SttFaultAction.REQUEST_AUDIO_RETRY
    TOOL_FAULT_ACTION: ToolFaultAction = ToolFaultAction.CONTINUE_WITHOUT_TOOL
    LLM_FAULT_ACTION: LlmFaultAction = LlmFaultAction.SAFE_RESPONSE
    TTS_FAULT_ACTION: TtsFaultAction = TtsFaultAction.TEXT_ONLY

    # --- Rate limiting ---
    RATE_LIMIT_REQUESTS_PER_MINUTE: int = Field(default=30, ge=1)
    RATE_LIMIT_BURST: int = Field(default=5, ge=1)

    # --- Timeouts (seconds) ---
    TIMEOUT_STT: float = Field(default=30.0, gt=0)
    TIMEOUT_LLM: float = Field(default=60.0, gt=0)
    TIMEOUT_TOOL: float = Field(default=30.0, gt=0)
    TIMEOUT_TTS: float = Field(default=30.0, gt=0)

    # --- Circuit breaker ---
    CB_FAILURE_THRESHOLD: int = Field(default=5, ge=1)
    CB_RECOVERY_TIMEOUT: float = Field(default=30.0, gt=0)

    # --- Session ---
    SESSION_TTL_SECONDS: int = Field(default=1800, ge=60)
    MAX_CONCURRENT_SESSIONS: int = Field(default=50000, ge=1)

    # --- Context compression (0 = disabled) ---
    MAX_CONTEXT_MESSAGES: int = Field(default=0, ge=0)
    CONTEXT_KEEP_RECENT: int = Field(default=5, ge=1)
    CONTEXT_SUMMARY_MODEL: str = "gpt-4o-mini"

    # --- Safe response message ---
    SAFE_RESPONSE_MESSAGE: str = (
        "No pude procesar tu solicitud en este momento. "
        "Por favor, intenta de nuevo más tarde."
    )


def get_settings() -> Settings:
    return Settings()
