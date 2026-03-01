"""Session state management models.

The session is lightweight and transient. The client owns the conversation
history (messages[]) and sends it on every request — the backend is stateless
with respect to context. The session only tracks per-turn processing state.
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class AgentState(str, Enum):
    IDLE = "idle"
    RECEIVING_INPUT = "receiving_input"
    TRANSCRIBING = "transcribing"
    THINKING = "thinking"
    CALLING_TOOL = "calling_tool"
    RESPONDING_TEXT = "responding_text"
    GENERATING_AUDIO = "generating_audio"
    DONE = "done"
    ERROR = "error"


VALID_TRANSITIONS: dict[AgentState, list[AgentState]] = {
    AgentState.IDLE: [AgentState.RECEIVING_INPUT],
    AgentState.RECEIVING_INPUT: [
        AgentState.TRANSCRIBING,
        AgentState.THINKING,
        AgentState.ERROR,
    ],
    AgentState.TRANSCRIBING: [
        AgentState.THINKING,
        AgentState.ERROR,
    ],
    AgentState.THINKING: [
        AgentState.CALLING_TOOL,
        AgentState.RESPONDING_TEXT,
        AgentState.ERROR,
    ],
    AgentState.CALLING_TOOL: [
        AgentState.THINKING,
        AgentState.RESPONDING_TEXT,
        AgentState.ERROR,
    ],
    AgentState.RESPONDING_TEXT: [
        AgentState.GENERATING_AUDIO,
        AgentState.DONE,
        AgentState.ERROR,
    ],
    AgentState.GENERATING_AUDIO: [
        AgentState.DONE,
        AgentState.ERROR,
    ],
    AgentState.DONE: [AgentState.IDLE],
    AgentState.ERROR: [AgentState.IDLE],
}


class Session(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    state: AgentState = AgentState.IDLE
    created_at: float = Field(default_factory=time.time)
    last_activity: float = Field(default_factory=time.time)
    metadata: dict[str, Any] = {}

    # Per-turn data — populated by the API layer, consumed by the pipeline
    messages: list[dict[str, Any]] = []
    current_input_text: Optional[str] = None
    current_input_audio: Optional[bytes] = None
    current_transcript: Optional[str] = None
    pending_tool_calls: list[dict[str, Any]] = []
    current_response_text: Optional[str] = None
    current_response_audio: Optional[bytes] = None

    def touch(self) -> None:
        self.last_activity = time.time()

    def is_expired(self, ttl: int) -> bool:
        return (time.time() - self.last_activity) > ttl

    def transition_to(self, new_state: AgentState) -> None:
        allowed = VALID_TRANSITIONS.get(self.state, [])
        if new_state not in allowed:
            raise InvalidStateTransition(
                f"Cannot transition from {self.state} to {new_state}. "
                f"Allowed: {allowed}"
            )
        self.state = new_state
        self.touch()

    def reset_turn(self) -> None:
        self.messages = []
        self.current_input_text = None
        self.current_input_audio = None
        self.current_transcript = None
        self.pending_tool_calls = []
        self.current_response_text = None
        self.current_response_audio = None
        self.state = AgentState.IDLE


class InvalidStateTransition(Exception):
    pass
