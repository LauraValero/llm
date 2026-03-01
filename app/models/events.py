"""Event contracts sent to the client via WebSocket."""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel


class EventType(str, Enum):
    # State transitions
    STATE_CHANGED = "state_changed"

    # Transcription (only when SHOW_USER_TRANSCRIPT=true)
    TRANSCRIPTION_PARTIAL = "transcription_partial"
    TRANSCRIPTION_FINAL = "transcription_final"

    # Failures
    STT_FAILED = "stt_failed"
    TOOL_FAILED = "tool_failed"
    LLM_FAILED = "llm_failed"
    TTS_FAILED = "tts_failed"

    # Responses
    AGENT_RESPONSE = "agent_response"
    SAFE_RESPONSE = "safe_response"

    # Session
    SESSION_CREATED = "session_created"
    SESSION_ENDED = "session_ended"

    # Audio retry
    REQUEST_AUDIO_RETRY = "request_audio_retry"

    # Input validation
    INPUT_REJECTED = "input_rejected"

    # Errors
    ERROR = "error"


class SttImpact(str, Enum):
    UX_ONLY = "ux_only"
    BLOCKING = "blocking"


class AgentEvent(BaseModel):
    """Base event dispatched to client."""

    event: EventType
    session_id: str
    data: dict[str, Any] = {}


class TranscriptionEvent(BaseModel):
    event: EventType
    text: str


class SttFailedEvent(BaseModel):
    event: EventType = EventType.STT_FAILED
    impact: SttImpact
    recoverable: bool = True


class ToolFailedEvent(BaseModel):
    event: EventType = EventType.TOOL_FAILED
    tool_name: str = ""
    recoverable: bool = True


class AgentResponseEvent(BaseModel):
    event: EventType = EventType.AGENT_RESPONSE
    text: str
    audio: Optional[str] = None  # base64-encoded audio when available


class SafeResponseEvent(BaseModel):
    event: EventType = EventType.SAFE_RESPONSE
    message: str


class InputRejectedEvent(BaseModel):
    event: EventType = EventType.INPUT_REJECTED
    reason: str


def make_event(event_type: EventType, session_id: str, **data: Any) -> AgentEvent:
    return AgentEvent(event=event_type, session_id=session_id, data=data)
