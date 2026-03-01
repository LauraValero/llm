"""Inbound request models with strict validation.

The primary contract is `messages: list[ChatMessage]` where:
  - messages[0].role == "system"  → agent identity / instructions
  - messages[-1].role == "user"   → current turn input (for text)
  - everything in between         → conversation context

For audio input, the last user turn is the audio payload itself.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, model_validator


class ChatMessage(BaseModel):
    role: str = Field(..., pattern=r"^(user|assistant|system)$")
    content: str


class ChatTextRequest(BaseModel):
    """Text-based chat: messages must end with a user message."""
    messages: list[ChatMessage] = Field(..., min_length=1)
    session_id: Optional[str] = None

    @model_validator(mode="after")
    def _validate_structure(self):
        if not self.messages:
            raise ValueError("messages cannot be empty")
        if self.messages[0].role != "system":
            raise ValueError("First message must have role 'system' (system prompt)")
        if self.messages[-1].role != "user":
            raise ValueError("Last message must have role 'user' (current turn)")
        return self


class ChatAudioRequest(BaseModel):
    """Audio-based chat: messages provide context, audio is the current turn."""
    messages: list[ChatMessage] = Field(..., min_length=1)
    audio_base64: str = Field(..., min_length=1)
    mime_type: str = Field(default="audio/webm", pattern=r"^audio/")
    session_id: Optional[str] = None

    @model_validator(mode="after")
    def _validate_structure(self):
        if not self.messages:
            raise ValueError("messages cannot be empty")
        if self.messages[0].role != "system":
            raise ValueError("First message must have role 'system' (system prompt)")
        return self


class ToolDefinition(BaseModel):
    name: str
    description: str
    parameters: dict = {}


class ToolResult(BaseModel):
    tool_call_id: str
    name: str
    result: str
    error: Optional[str] = None
