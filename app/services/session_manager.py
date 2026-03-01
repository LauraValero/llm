"""In-memory session store.

For horizontal scaling, swap this for a Redis-backed store behind the same
interface. The rest of the application is stateless.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

from app.config import Settings
from app.models.session import Session

logger = logging.getLogger(__name__)


class SessionManager:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._sessions: dict[str, Session] = {}
        self._lock = asyncio.Lock()

    async def create(self) -> Session:
        async with self._lock:
            if len(self._sessions) >= self._settings.MAX_CONCURRENT_SESSIONS:
                raise SessionLimitReached(
                    f"Max concurrent sessions ({self._settings.MAX_CONCURRENT_SESSIONS}) reached"
                )
            session = Session()
            self._sessions[session.id] = session
            logger.info("Session created: %s", session.id)
            return session

    async def get(self, session_id: str) -> Optional[Session]:
        session = self._sessions.get(session_id)
        if session and session.is_expired(self._settings.SESSION_TTL_SECONDS):
            await self.delete(session_id)
            return None
        return session

    async def get_or_create(self, session_id: Optional[str]) -> Session:
        if session_id:
            session = await self.get(session_id)
            if session:
                return session
        return await self.create()

    async def delete(self, session_id: str) -> None:
        async with self._lock:
            self._sessions.pop(session_id, None)
            logger.info("Session deleted: %s", session_id)

    async def cleanup_expired(self) -> int:
        """Remove expired sessions. Returns count of removed sessions."""
        now = time.time()
        ttl = self._settings.SESSION_TTL_SECONDS
        expired = [
            sid
            for sid, s in self._sessions.items()
            if (now - s.last_activity) > ttl
        ]
        async with self._lock:
            for sid in expired:
                self._sessions.pop(sid, None)
        if expired:
            logger.info("Cleaned up %d expired sessions", len(expired))
        return len(expired)

    @property
    def active_count(self) -> int:
        return len(self._sessions)


class SessionLimitReached(Exception):
    pass
