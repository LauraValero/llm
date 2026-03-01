"""OpenAI provider implementation (GPT-4o, Whisper, TTS)."""

from __future__ import annotations

import base64
import logging
from typing import Any, Optional

from openai import AsyncOpenAI

from app.providers.base import (
    BaseLLMProvider,
    BaseSTTProvider,
    BaseTTSProvider,
    LLMResponse,
    STTResponse,
    TTSResponse,
)

logger = logging.getLogger(__name__)


class OpenAILLM(BaseLLMProvider):
    def __init__(self, api_key: str, model: str = "gpt-4o") -> None:
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model

    async def chat(
        self,
        messages: list[dict[str, Any]],
        text: Optional[str] = None,
        audio: Optional[bytes] = None,
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> LLMResponse:
        try:
            chat_messages = list(messages)

            if audio is not None:
                audio_b64 = base64.b64encode(audio).decode()
                user_content: list[dict[str, Any]] = [
                    {
                        "type": "input_audio",
                        "input_audio": {"data": audio_b64, "format": "wav"},
                    }
                ]
                if text:
                    user_content.insert(0, {"type": "text", "text": text})
                chat_messages.append({"role": "user", "content": user_content})
            elif text:
                chat_messages.append({"role": "user", "content": text})

            kwargs: dict[str, Any] = {
                "model": self._model,
                "messages": chat_messages,
            }
            if audio is not None:
                kwargs["modalities"] = ["text"]
            if tools:
                kwargs["tools"] = tools

            response = await self._client.chat.completions.create(**kwargs)
            choice = response.choices[0]
            message = choice.message

            tool_calls_raw = []
            if message.tool_calls:
                for tc in message.tool_calls:
                    tool_calls_raw.append({
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    })

            return LLMResponse(
                text=message.content or "",
                tool_calls=tool_calls_raw,
                raw=response,
            )
        except Exception as exc:
            logger.exception("OpenAI LLM error")
            return LLMResponse(error=str(exc))

    async def chat_with_tool_results(
        self,
        messages: list[dict[str, Any]],
        tool_results: list[dict[str, Any]],
        original_response: LLMResponse,
    ) -> LLMResponse:
        try:
            chat_messages = list(messages)

            if original_response.raw:
                assistant_msg = original_response.raw.choices[0].message
                chat_messages.append(assistant_msg.model_dump())

            for tr in tool_results:
                chat_messages.append({
                    "role": "tool",
                    "tool_call_id": tr["tool_call_id"],
                    "content": tr["result"] if not tr.get("error") else f"Error: {tr['error']}",
                })

            response = await self._client.chat.completions.create(
                model=self._model,
                messages=chat_messages,
            )
            return LLMResponse(
                text=response.choices[0].message.content or "",
                raw=response,
            )
        except Exception as exc:
            logger.exception("OpenAI tool follow-up error")
            return LLMResponse(error=str(exc))


class OpenAISTT(BaseSTTProvider):
    def __init__(self, api_key: str, model: str = "whisper-1") -> None:
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model

    async def transcribe(self, audio: bytes, mime_type: str = "audio/webm") -> STTResponse:
        try:
            ext = mime_type.split("/")[-1].split(";")[0]
            response = await self._client.audio.transcriptions.create(
                model=self._model,
                file=(f"audio.{ext}", audio, mime_type),
            )
            return STTResponse(text=response.text)
        except Exception as exc:
            logger.exception("OpenAI STT error")
            return STTResponse(error=str(exc))


class OpenAITTS(BaseTTSProvider):
    def __init__(self, api_key: str, model: str = "tts-1", voice: str = "alloy") -> None:
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model
        self._voice = voice

    async def synthesize(self, text: str) -> TTSResponse:
        try:
            response = await self._client.audio.speech.create(
                model=self._model,
                voice=self._voice,
                input=text,
            )
            audio_bytes = response.read()
            return TTSResponse(audio=audio_bytes)
        except Exception as exc:
            logger.exception("OpenAI TTS error")
            return TTSResponse(error=str(exc))
