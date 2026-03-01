"""Anthropic provider implementation (Claude)."""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from anthropic import AsyncAnthropic

from app.providers.base import (
    BaseLLMProvider,
    LLMResponse,
)

logger = logging.getLogger(__name__)


class AnthropicLLM(BaseLLMProvider):
    """Anthropic Claude – text-only LLM with tool use support."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514") -> None:
        self._client = AsyncAnthropic(api_key=api_key)
        self._model = model

    async def chat(
        self,
        messages: list[dict[str, Any]],
        text: Optional[str] = None,
        audio: Optional[bytes] = None,
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> LLMResponse:
        try:
            chat_messages = _convert_messages(messages)

            if text:
                chat_messages.append({"role": "user", "content": text})

            # audio not supported by Anthropic — caller must ensure STT ran first

            kwargs: dict[str, Any] = {
                "model": self._model,
                "max_tokens": 4096,
                "messages": chat_messages,
            }

            if tools:
                kwargs["tools"] = _convert_tools(tools)

            response = await self._client.messages.create(**kwargs)

            response_text = ""
            tool_calls: list[dict[str, Any]] = []

            for block in response.content:
                if block.type == "text":
                    response_text += block.text
                elif block.type == "tool_use":
                    tool_calls.append({
                        "id": block.id,
                        "name": block.name,
                        "arguments": json.dumps(block.input) if isinstance(block.input, dict) else str(block.input),
                    })

            return LLMResponse(text=response_text, tool_calls=tool_calls, raw=response)

        except Exception as exc:
            logger.exception("Anthropic LLM error")
            return LLMResponse(error=str(exc))

    async def chat_with_tool_results(
        self,
        messages: list[dict[str, Any]],
        tool_results: list[dict[str, Any]],
        original_response: LLMResponse,
    ) -> LLMResponse:
        try:
            chat_messages = _convert_messages(messages)

            if original_response.raw:
                chat_messages.append({
                    "role": "assistant",
                    "content": original_response.raw.content,
                })

            tool_result_blocks = []
            for tr in tool_results:
                tool_result_blocks.append({
                    "type": "tool_result",
                    "tool_use_id": tr["tool_call_id"],
                    "content": tr["result"] if not tr.get("error") else f"Error: {tr['error']}",
                })

            chat_messages.append({"role": "user", "content": tool_result_blocks})

            response = await self._client.messages.create(
                model=self._model,
                max_tokens=4096,
                messages=chat_messages,
            )

            response_text = ""
            for block in response.content:
                if block.type == "text":
                    response_text += block.text

            return LLMResponse(text=response_text, raw=response)

        except Exception as exc:
            logger.exception("Anthropic tool follow-up error")
            return LLMResponse(error=str(exc))


def _convert_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert generic messages to Anthropic format (no 'system' role in messages list)."""
    converted = []
    for m in messages:
        role = m.get("role", "user")
        if role == "system":
            continue  # system prompt handled separately if needed
        converted.append({"role": role, "content": m.get("content", "")})
    return converted


def _convert_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI-style tool definitions to Anthropic format."""
    anthropic_tools = []
    for t in tools:
        fn = t.get("function", t)
        anthropic_tools.append({
            "name": fn.get("name", ""),
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
        })
    return anthropic_tools
