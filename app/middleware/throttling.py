"""FastAPI middleware for per-IP throttling and payload-size enforcement."""

from __future__ import annotations

import logging

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from app.middleware.rate_limiter import RateLimiter, RateLimitExceeded

logger = logging.getLogger(__name__)


class ThrottlingMiddleware(BaseHTTPMiddleware):
    """Applies token-bucket rate limiting keyed by client IP."""

    def __init__(self, app, rate_limiter: RateLimiter) -> None:  # noqa: ANN001
        super().__init__(app)
        self._limiter = rate_limiter

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        client_ip = request.client.host if request.client else "unknown"

        try:
            await self._limiter.check(client_ip)
        except RateLimitExceeded as exc:
            return Response(
                content=f'{{"error":"rate_limit_exceeded","retry_after":{exc.retry_after:.1f}}}',
                status_code=429,
                media_type="application/json",
                headers={"Retry-After": str(int(exc.retry_after + 1))},
            )

        return await call_next(request)
