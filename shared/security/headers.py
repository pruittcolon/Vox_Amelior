"""
Security Headers Middleware

Adds security headers to all HTTP responses for defense-in-depth protection.
These headers protect against XSS, clickjacking, MIME sniffing, and other attacks.
"""

import logging

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware that adds security headers to all responses.

    Headers added:
    - X-Content-Type-Options: nosniff (prevent MIME sniffing)
    - X-Frame-Options: DENY (prevent clickjacking)
    - Referrer-Policy: strict-origin-when-cross-origin
    - Permissions-Policy: restrict sensitive features
    - Cache-Control: no-store for API responses
    """

    def __init__(
        self,
        app,
        frame_options: str = "DENY",
        content_type_options: str = "nosniff",
        referrer_policy: str = "strict-origin-when-cross-origin",
        permissions_policy: str = "geolocation=(), microphone=(self), camera=()",
        cache_control: str = "no-store",
        exempt_paths: list = None,
    ):
        """
        Initialize security headers middleware.

        Args:
            app: FastAPI/Starlette app
            frame_options: X-Frame-Options value (DENY, SAMEORIGIN, or ALLOW-FROM uri)
            content_type_options: X-Content-Type-Options value
            referrer_policy: Referrer-Policy value
            permissions_policy: Permissions-Policy value
            cache_control: Cache-Control value for API responses
            exempt_paths: Paths exempt from Cache-Control header
        """
        super().__init__(app)
        self.frame_options = frame_options
        self.content_type_options = content_type_options
        self.referrer_policy = referrer_policy
        self.permissions_policy = permissions_policy
        self.cache_control = cache_control
        self.exempt_paths = exempt_paths or ["/docs", "/openapi.json", "/redoc"]

        logger.info(f"SecurityHeadersMiddleware enabled (X-Frame-Options: {frame_options})")

    async def dispatch(self, request: Request, call_next):
        """Add security headers to response."""
        response: Response = await call_next(request)

        # Core security headers
        response.headers["X-Content-Type-Options"] = self.content_type_options
        response.headers["X-Frame-Options"] = self.frame_options
        response.headers["Referrer-Policy"] = self.referrer_policy
        response.headers["Permissions-Policy"] = self.permissions_policy

        # Cache-Control for API responses (exempting docs)
        path = request.url.path
        if not any(path.startswith(exempt) for exempt in self.exempt_paths):
            response.headers["Cache-Control"] = self.cache_control

        # Remove potentially leaky headers
        response.headers.pop("Server", None)
        response.headers.pop("X-Powered-By", None)

        return response


# Pre-configured header sets for different use cases
STRICT_HEADERS = {
    "frame_options": "DENY",
    "content_type_options": "nosniff",
    "referrer_policy": "no-referrer",
    "permissions_policy": "geolocation=(), microphone=(), camera=(), payment=(), usb=()",
    "cache_control": "no-store, no-cache, must-revalidate",
}

STANDARD_HEADERS = {
    "frame_options": "SAMEORIGIN",
    "content_type_options": "nosniff",
    "referrer_policy": "strict-origin-when-cross-origin",
    "permissions_policy": "geolocation=(), microphone=(self), camera=()",
    "cache_control": "no-store",
}


def get_security_headers_middleware(app, strict: bool = False, **kwargs):
    """
    Factory function to create SecurityHeadersMiddleware with presets.

    Args:
        app: FastAPI/Starlette app
        strict: Use strict security headers (recommended for APIs)
        **kwargs: Override specific headers

    Returns:
        Configured SecurityHeadersMiddleware
    """
    preset = STRICT_HEADERS if strict else STANDARD_HEADERS
    config = {**preset, **kwargs}
    return SecurityHeadersMiddleware(app, **config)
