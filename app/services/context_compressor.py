"""Context window compression service.

When MAX_CONTEXT_MESSAGES > 0 and the incoming messages[] exceeds that
threshold, the oldest messages (excluding the system prompt and the most
recent CONTEXT_KEEP_RECENT turns) are summarized into a single system-role
message using a lightweight model.

When MAX_CONTEXT_MESSAGES == 0, compression is disabled — messages pass
through untouched.
"""

from __future__ import annotations

import logging
from typing import Any

from app.config import Settings
from app.providers.base import BaseLLMProvider, LLMResponse

logger = logging.getLogger(__name__)

SUMMARY_PROMPT = (
    "Summarize the following conversation concisely. "
    "Preserve all key facts, decisions, and context the assistant needs "
    "to continue the conversation. Reply ONLY with the summary, nothing else."
)


class ContextCompressor:
    def __init__(self, settings: Settings, summary_provider: BaseLLMProvider) -> None:
        self._max_messages = settings.MAX_CONTEXT_MESSAGES
        self._keep_recent = settings.CONTEXT_KEEP_RECENT
        self._provider = summary_provider

    async def maybe_compress(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Returns the (possibly compressed) message list.

        If compression is disabled (MAX_CONTEXT_MESSAGES == 0) or the
        message count is within the limit, returns the original list.
        """
        if self._max_messages == 0:
            return messages

        if len(messages) <= self._max_messages:
            return messages

        system_prompt = messages[0] if messages[0].get("role") == "system" else None
        body = messages[1:] if system_prompt else messages

        # Keep the most recent turns intact (each turn = 1 message)
        keep_count = min(self._keep_recent, len(body))
        to_summarize = body[: len(body) - keep_count]
        to_keep = body[len(body) - keep_count :]

        if not to_summarize:
            return messages

        summary = await self._summarize(to_summarize)

        compressed: list[dict[str, Any]] = []
        if system_prompt:
            compressed.append(system_prompt)

        compressed.append({
            "role": "system",
            "content": f"[Conversation summary]: {summary}",
        })
        compressed.extend(to_keep)

        logger.info(
            "Context compressed: %d messages → %d (summarized %d)",
            len(messages),
            len(compressed),
            len(to_summarize),
        )
        return compressed

    async def _summarize(self, messages: list[dict[str, Any]]) -> str:
        conversation_text = "\n".join(
            f"{m.get('role', 'unknown')}: {m.get('content', '')}"
            for m in messages
        )

        summary_messages: list[dict[str, Any]] = [
            {"role": "system", "content": SUMMARY_PROMPT},
            {"role": "user", "content": conversation_text},
        ]

        try:
            response: LLMResponse = await self._provider.chat(
                messages=summary_messages,
                text=None,
                audio=None,
                tools=None,
            )
            if response.error:
                logger.error("Summary LLM error: %s — skipping compression", response.error)
                return conversation_text

            return response.text.strip()

        except Exception:
            logger.exception("Context compression failed — returning raw text")
            return conversation_text
