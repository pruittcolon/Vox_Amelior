"""
Security Headers Middleware.

Attaches security headers (CSP, HSTS, X-Frame-Options, etc.) to every response.

Phase 2 of API Restructure.
"""

import logging

from config.settings import settings
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Attach common security headers to every response."""

    def __init__(self, app, secure_cookies: bool = False):
        super().__init__(app)
        self.include_hsts = secure_cookies or settings.FORCE_HSTS
        # DEV ONLY: Set ALLOW_FRAMING=true to allow iframe embedding
        # WARNING: Do NOT enable in production - disables clickjacking protection
        self.allow_framing = settings.ALLOW_FRAMING

        if self.allow_framing:
            logger.warning("⚠️ X-Frame-Options DISABLED - clickjacking protection off")

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        response.headers.setdefault("X-Content-Type-Options", "nosniff")

        if not self.allow_framing:
            response.headers.setdefault("X-Frame-Options", "DENY")

        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        response.headers.setdefault("Permissions-Policy", "geolocation=(), microphone=(self), camera=()")

        # Content Security Policy
        # Note: 'unsafe-inline' required for inline styles/scripts
        # Consider nonce-based CSP in future
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; "
            "img-src 'self' data: https:; "
            "script-src 'self' 'unsafe-inline' 'wasm-unsafe-eval' https://unpkg.com https://cdn.jsdelivr.net https://cdn.plot.ly; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com; "
            "connect-src 'self' http://localhost:* ws://localhost:* wss://localhost:* "
            "http://127.0.0.1:* ws://127.0.0.1:* wss://127.0.0.1:* "
            "https://unpkg.com https://cdn.jsdelivr.net https://cdn.plot.ly; "
            "frame-ancestors 'none'; "
            "base-uri 'self'; "
            "form-action 'self';",
        )

        if self.include_hsts:
            response.headers.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains; preload")

        return response
