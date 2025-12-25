"""
Rate Limiting Middleware.

Redis-backed rate limiting with per-endpoint limits and trusted proxy support.
Falls back to in-memory rate limiting when Redis is unavailable.

Phase 2 of API Restructure.
"""

import logging
import time
from typing import Any

from config.settings import settings
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

# Try to import Redis rate limiting (optional dependency)
try:
    from shared.security.redis_rate_limit import (
        extract_client_ip,
        get_rate_limit_config,
        get_rate_limiter,
    )

    _REDIS_RATE_LIMIT_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Redis rate limiting not available: {e}")
    _REDIS_RATE_LIMIT_AVAILABLE = False
    extract_client_ip = None
    get_rate_limiter = None


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Redis-backed rate limiting with per-endpoint limits and trusted proxy support."""

    def __init__(self, app):
        super().__init__(app)
        self.default_window = settings.RATE_LIMIT_WINDOW_SEC
        self.default_limit = settings.RATE_LIMIT_DEFAULT
        self.auth_limit = settings.RATE_LIMIT_AUTH
        self.skip_prefixes = settings.rate_limit_skip_prefixes_tuple
        self.skip_paths = settings.rate_limit_skip_paths_set
        self.enabled = settings.RATE_LIMIT_ENABLED

        # Initialize Redis rate limiter if available
        self.rate_limiter = None
        if _REDIS_RATE_LIMIT_AVAILABLE:
            try:
                from config import RedisConfig

                redis_url = RedisConfig.get_redis_url()
                self.rate_limiter = get_rate_limiter(redis_url)
                logger.info("✅ Rate limiting upgraded to Redis backend")
            except Exception as e:
                logger.warning(f"⚠️ Using in-memory fallback rate limiting: {e}")

        # In-memory fallback buckets
        self.buckets: dict[str, dict[str, Any]] = {}

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP, considering trusted proxies if Redis module available."""
        if _REDIS_RATE_LIMIT_AVAILABLE and extract_client_ip:
            return extract_client_ip(request)
        return request.client.host if request.client else "unknown"

    def _key(self, request: Request, scope: str) -> str:
        """Generate rate limit bucket key."""
        return f"{self._get_client_ip(request)}:{scope}"

    async def dispatch(self, request: Request, call_next):
        if not self.enabled:
            return await call_next(request)

        path = request.url.path or "/"

        # Skip static resources and health endpoints
        if path in self.skip_paths or any(path.startswith(prefix) for prefix in self.skip_prefixes):
            return await call_next(request)

        # Use Redis rate limiter if available
        if self.rate_limiter:
            key = self._key(request, "redis")
            allowed, remaining, retry_after = self.rate_limiter.check_rate_limit(key, path)

            if not allowed:
                limit = self.rate_limiter.get_limit_for_path(path)
                logger.warning(f"Rate limit exceeded for {key}: path={path} limit={limit}")
                return JSONResponse(
                    status_code=429,
                    content={
                        "detail": "Too Many Requests",
                        "limit": limit,
                        "window_sec": self.default_window,
                        "retry_after_sec": retry_after,
                    },
                    headers={"Retry-After": str(retry_after)},
                )
            return await call_next(request)

        # Fallback to in-memory rate limiting
        window = self.default_window
        limit = self.default_limit
        scope = "global"
        if path.startswith("/api/auth/login"):
            scope = "auth"
            limit = self.auth_limit

        now = int(time.time())
        key = self._key(request, scope)
        bucket = self.buckets.get(key, {"count": 0, "window_start": now})
        if now - bucket["window_start"] >= window:
            bucket = {"count": 0, "window_start": now}
        bucket["count"] += 1
        self.buckets[key] = bucket
        if bucket["count"] > limit:
            retry_after = max(1, window - (now - bucket["window_start"]))
            logger.warning(f"Rate limit exceeded for {key}: {bucket['count']}/{limit}")
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Too Many Requests",
                    "limit": limit,
                    "window_sec": window,
                    "retry_after_sec": retry_after,
                },
                headers={"Retry-After": str(retry_after)},
            )

        return await call_next(request)
