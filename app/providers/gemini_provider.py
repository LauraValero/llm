"""Google Gemini provider implementation."""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from google import genai
from google.genai import types as genai_types

from app.providers.base import (
    BaseLLMProvider,
    BaseSTTProvider,
    BaseTTSProvider,
    LLMResponse,
    STTResponse,
    TTSResponse,
)

logger = logging.getLogger(__name__)


class GeminiLLM(BaseLLMProvider):
    """Google Gemini – supports text, audio input, and tool use."""

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash") -> None:
        self._client = genai.Client(api_key=api_key)
        self._model = model

    async def chat(
        self,
        messages: list[dict[str, Any]],
        text: Optional[str] = None,
        audio: Optional[bytes] = None,
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> LLMResponse:
        try:
            contents = _build_contents(messages, text, audio)

            config: dict[str, Any] = {}
            if tools:
                config["tools"] = _convert_tools(tools)

            response = await self._client.aio.models.generate_content(
                model=self._model,
                contents=contents,
                config=genai_types.GenerateContentConfig(**config) if config else None,
            )

            response_text = ""
            tool_calls: list[dict[str, Any]] = []

            if response.candidates:
                for part in response.candidates[0].content.parts:
                    if part.text:
                        response_text += part.text
                    elif part.function_call:
                        fc = part.function_call
                        tool_calls.append({
                            "id": fc.name,
                            "name": fc.name,
                            "arguments": json.dumps(dict(fc.args)) if fc.args else "{}",
                        })

            return LLMResponse(text=response_text, tool_calls=tool_calls, raw=response)

        except Exception as exc:
            logger.exception("Gemini LLM error")
            return LLMResponse(error=str(exc))

    async def chat_with_tool_results(
        self,
        messages: list[dict[str, Any]],
        tool_results: list[dict[str, Any]],
        original_response: LLMResponse,
    ) -> LLMResponse:
        try:
            contents = _build_contents(messages)

            if original_response.raw and original_response.raw.candidates:
                contents.append(original_response.raw.candidates[0].content)

            function_responses = []
            for tr in tool_results:
                function_responses.append(
                    genai_types.Part.from_function_response(
                        name=tr["name"],
                        response={"result": tr["result"], "error": tr.get("error")},
                    )
                )
            contents.append(genai_types.Content(role="user", parts=function_responses))

            response = await self._client.aio.models.generate_content(
                model=self._model,
                contents=contents,
            )

            response_text = ""
            if response.candidates:
                for part in response.candidates[0].content.parts:
                    if part.text:
                        response_text += part.text

            return LLMResponse(text=response_text, raw=response)

        except Exception as exc:
            logger.exception("Gemini tool follow-up error")
            return LLMResponse(error=str(exc))


class GeminiSTT(BaseSTTProvider):
    """Use Gemini model to transcribe audio (multimodal input)."""

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash") -> None:
        self._client = genai.Client(api_key=api_key)
        self._model = model

    async def transcribe(self, audio: bytes, mime_type: str = "audio/webm") -> STTResponse:
        try:
            contents = [
                genai_types.Content(
                    role="user",
                    parts=[
                        genai_types.Part.from_bytes(data=audio, mime_type=mime_type),
                        genai_types.Part.from_text(
                            "Transcribe the audio exactly. Return only the transcription, nothing else."
                        ),
                    ],
                )
            ]
            response = await self._client.aio.models.generate_content(
                model=self._model,
                contents=contents,
            )
            text = response.text or ""
            return STTResponse(text=text.strip())
        except Exception as exc:
            logger.exception("Gemini STT error")
            return STTResponse(error=str(exc))


class GeminiTTS(BaseTTSProvider):
    """Gemini TTS placeholder — falls back to text-only if not available."""

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash") -> None:
        self._api_key = api_key
        self._model = model

    async def synthesize(self, text: str) -> TTSResponse:
        # Gemini doesn't have a dedicated TTS API yet; return error to trigger text_only fallback
        return TTSResponse(error="Gemini TTS not available; falling back to text-only")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _build_contents(
    messages: list[dict[str, Any]],
    text: Optional[str] = None,
    audio: Optional[bytes] = None,
) -> list[genai_types.Content]:
    contents: list[genai_types.Content] = []

    for m in messages:
        role = "model" if m.get("role") == "assistant" else "user"
        parts = [genai_types.Part.from_text(m.get("content", ""))]
        contents.append(genai_types.Content(role=role, parts=parts))

    if text or audio:
        parts: list[genai_types.Part] = []
        if audio:
            parts.append(genai_types.Part.from_bytes(data=audio, mime_type="audio/webm"))
        if text:
            parts.append(genai_types.Part.from_text(text))
        contents.append(genai_types.Content(role="user", parts=parts))

    return contents


def _convert_tools(tools: list[dict[str, Any]]) -> list[genai_types.Tool]:
    declarations = []
    for t in tools:
        fn = t.get("function", t)
        declarations.append(
            genai_types.FunctionDeclaration(
                name=fn.get("name", ""),
                description=fn.get("description", ""),
                parameters=fn.get("parameters"),
            )
        )
    return [genai_types.Tool(function_declarations=declarations)]
