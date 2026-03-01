"""Event dispatcher — sends structured events to connected WebSocket clients."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import WebSocket

from app.models.events import AgentEvent

logger = logging.getLogger(__name__)


class EventDispatcher:
    """Manages WebSocket connections and broadcasts typed events."""

    def __init__(self) -> None:
        self._connections: dict[str, WebSocket] = {}
        self._lock = asyncio.Lock()

    async def register(self, session_id: str, ws: WebSocket) -> None:
        async with self._lock:
            self._connections[session_id] = ws
        logger.debug("WS registered for session %s", session_id)

    async def unregister(self, session_id: str) -> None:
        async with self._lock:
            self._connections.pop(session_id, None)
        logger.debug("WS unregistered for session %s", session_id)

    async def send(self, session_id: str, event: AgentEvent) -> None:
        ws = self._connections.get(session_id)
        if ws is None:
            logger.warning("No WS connection for session %s — event dropped", session_id)
            return
        try:
            await ws.send_json(event.model_dump())
        except Exception:
            logger.exception("Failed to send event to session %s", session_id)
            await self.unregister(session_id)

    async def broadcast(self, event: AgentEvent) -> None:
        tasks = [self.send(sid, event) for sid in list(self._connections)]
        await asyncio.gather(*tasks, return_exceptions=True)

    @property
    def connected_count(self) -> int:
        return len(self._connections)
