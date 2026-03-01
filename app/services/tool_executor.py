"""Tool executor — runs registered tools with retry + fault isolation.

This module provides the execution shell. Actual tool implementations are
registered externally (not part of this system's scope per requirements).
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Awaitable, Callable, Optional

from app.config import Settings
from app.resilience.retry_manager import retry_async

logger = logging.getLogger(__name__)

ToolFunction = Callable[..., Awaitable[str]]


class ToolExecutor:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._registry: dict[str, ToolFunction] = {}

    def register(self, name: str, fn: ToolFunction) -> None:
        self._registry[name] = fn

    async def execute(
        self, tool_call: dict[str, Any]
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Execute a tool call.  Returns (result, error).
        Tool failures never break the pipeline.
        """
        name = tool_call.get("name", "")
        raw_args = tool_call.get("arguments", "{}")

        fn = self._registry.get(name)
        if fn is None:
            err = f"Tool '{name}' not registered"
            logger.warning(err)
            return None, err

        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except json.JSONDecodeError as exc:
            return None, f"Invalid tool arguments: {exc}"

        try:
            result = await retry_async(
                self._run_with_timeout,
                fn,
                args,
                max_retries=self._settings.MAX_RETRIES_TOOL,
                operation_name=f"tool:{name}",
            )
            return result, None
        except Exception as exc:
            logger.error("Tool '%s' failed: %s", name, exc)
            return None, str(exc)

    async def _run_with_timeout(
        self, fn: ToolFunction, args: dict[str, Any]
    ) -> str:
        return await asyncio.wait_for(
            fn(**args),
            timeout=self._settings.TIMEOUT_TOOL,
        )
