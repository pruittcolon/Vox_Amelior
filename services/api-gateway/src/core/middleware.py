"""
Core Middleware Module - Security and Performance Middleware.

This module contains enterprise-grade middleware classes for:
- Rate limiting (Redis-backed with in-memory fallback)
- Security headers (CSP, HSTS, X-Frame-Options, etc.)
- CSRF protection (double-submit cookie pattern)
- Canonical host redirection

All middleware follows ISO 27002 and OWASP API Security guidelines.
"""

import logging
import os
import time
import uuid

from fastapi import Request
from fastapi.responses import JSONResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import RedirectResponse

# Import SecurityConfig with fallback for test environment
try:
    from src.config import SecurityConfig as SecConf
except ImportError:
    try:
        from config.legacy_config import SecurityConfig as SecConf
    except ImportError:
        # Minimal defaults if config not available
        class SecConf:
            SESSION_COOKIE_NAME = "ws_session"
            CSRF_COOKIE_NAME = "ws_csrf"
            CSRF_HEADER_NAME = "X-CSRF-Token"
            SESSION_COOKIE_SECURE = True
            ENABLE_CSRF = True

logger = logging.getLogger(__name__)

# Centralized cookie and CSRF names from security config
SESSION_COOKIE_NAME = SecConf.SESSION_COOKIE_NAME
CSRF_COOKIE_NAME = SecConf.CSRF_COOKIE_NAME
CSRF_HEADER_NAME = SecConf.CSRF_HEADER_NAME
SESSION_COOKIE_SECURE = SecConf.SESSION_COOKIE_SECURE

# Rate limiting configuration
RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "true").lower() in {"1", "true", "yes"}
RATE_LIMIT_SKIP_PREFIXES = tuple(
    filter(None, os.getenv("RATE_LIMIT_SKIP_PREFIXES", "/ui/,/ui,/assets/,/static/,/docs/").split(","))
)
RATE_LIMIT_SKIP_PATHS = set(
    filter(
        None,
        os.getenv("RATE_LIMIT_SKIP_PATHS", "/health,/,/api/gemma/warmup,/api/gemma/stats,/upload,/api/upload").split(
            ","
        ),
    )
)

# Redis rate limiting (optional)
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


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Redis-backed rate limiting with per-endpoint limits and trusted proxy support.

    Implements sliding window rate limiting per client IP. Falls back to
    in-memory buckets if Redis is unavailable.

    Attributes:
        default_window: Time window in seconds for rate limit calculation.
        default_limit: Maximum requests per window for general endpoints.
        auth_limit: Maximum requests per window for authentication endpoints.
        rate_limiter: Redis rate limiter instance (or None for fallback).
        buckets: In-memory fallback rate limit buckets.

    Example:
        >>> app.add_middleware(RateLimitMiddleware)
    """

    def __init__(self, app):
        """Initialize rate limiting middleware.

        Args:
            app: FastAPI/Starlette application instance.
        """
        super().__init__(app)
        self.default_window = int(os.getenv("RATE_LIMIT_WINDOW_SEC", "60"))
        self.default_limit = int(os.getenv("RATE_LIMIT_DEFAULT", "120"))
        self.auth_limit = int(os.getenv("RATE_LIMIT_AUTH", "20"))

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
        self.buckets: dict[str, dict[str, int]] = {}

    def _key(self, request: Request, scope: str) -> str:
        """Generate rate limit key from client IP and scope.

        Args:
            request: FastAPI request object.
            scope: Rate limit scope (e.g., 'global', 'auth').

        Returns:
            Rate limit key string.
        """
        if _REDIS_RATE_LIMIT_AVAILABLE:
            client_ip = extract_client_ip(request)
        else:
            client_ip = request.client.host if request.client else "unknown"
        return f"{client_ip}:{scope}"

    async def dispatch(self, request: Request, call_next):
        """Process request through rate limiting.

        Args:
            request: Incoming HTTP request.
            call_next: Next middleware/handler in chain.

        Returns:
            HTTP response (429 if rate limited, otherwise from handler).
        """
        if not RATE_LIMIT_ENABLED:
            return await call_next(request)

        path = request.url.path or "/"

        # Skip static resources and health endpoints
        if path in RATE_LIMIT_SKIP_PATHS or any(path.startswith(prefix) for prefix in RATE_LIMIT_SKIP_PREFIXES):
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
            retry_after = window - (now - bucket["window_start"]) or 1
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


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Attach common security headers to every response.

    Implements OWASP secure headers guidelines including:
    - X-Content-Type-Options: nosniff
    - X-Frame-Options: DENY (unless ALLOW_FRAMING=true)
    - Referrer-Policy: strict-origin-when-cross-origin
    - Content-Security-Policy with strict defaults
    - Strict-Transport-Security (if HTTPS enabled)

    Attributes:
        include_hsts: Whether to include HSTS header.
        allow_framing: Whether to allow iframe embedding (dev only).
    """

    def __init__(self, app):
        """Initialize security headers middleware.

        Args:
            app: FastAPI/Starlette application instance.
        """
        super().__init__(app)
        force_hsts = os.getenv("FORCE_HSTS", "false").lower() in {"1", "true", "yes"}
        self.include_hsts = SESSION_COOKIE_SECURE or force_hsts
        # DEV ONLY: Set ALLOW_FRAMING=true to allow iframe embedding
        # WARNING: Do NOT enable in production - this disables clickjacking protection
        self.allow_framing = os.getenv("ALLOW_FRAMING", "false").lower() in {"1", "true", "yes"}

    async def dispatch(self, request: Request, call_next):
        """Add security headers to response.

        Args:
            request: Incoming HTTP request.
            call_next: Next middleware/handler in chain.

        Returns:
            HTTP response with security headers added.
        """
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        if not self.allow_framing:
            response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        response.headers.setdefault("Permissions-Policy", "geolocation=(), microphone=(self), camera=()")
        # Content Security Policy
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; "
            "img-src 'self' data: https:; "
            "script-src 'self' 'unsafe-inline' 'wasm-unsafe-eval' https://unpkg.com https://cdn.jsdelivr.net https://cdn.plot.ly; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com; "
            "connect-src 'self' http://localhost:* ws://localhost:* wss://localhost:* http://127.0.0.1:* ws://127.0.0.1:* wss://127.0.0.1:* https://unpkg.com https://cdn.jsdelivr.net https://cdn.plot.ly; "
            "frame-ancestors 'none'; "
            "base-uri 'self'; "
            "form-action 'self';",
        )
        if self.include_hsts:
            response.headers.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains; preload")
        return response


class CSRFMiddleware(BaseHTTPMiddleware):
    """CSRF enforcement middleware using double-submit cookie pattern.

    Validates that mutating requests (POST, PUT, DELETE) include a valid
    CSRF token that matches both the cookie and session. Bearer token
    authentication (for mobile clients) bypasses CSRF checks.

    Attributes:
        exempt_paths: Paths that don't require CSRF validation.
        exempt_prefixes: Path prefixes that don't require CSRF validation.
        bearer_auth_paths: Paths that accept Bearer token without CSRF.
    """

    def __init__(self, app):
        """Initialize CSRF middleware.

        Args:
            app: FastAPI/Starlette application instance.
        """
        super().__init__(app)
        self.exempt_paths: set[str] = {
            "/health",
            "/api/auth/login",
            "/api/auth/register",
            "/api/auth/logout",
            "/docs",
            "/openapi.json",
            "/upload",
            "/api/upload",
            "/api/public/chat",
            "/vectorize/database",
            "/vectorize/status",
            # RAG chatbot endpoints
            "/embed",
            "/ask",
            "/databases",
            "/api/databases",
            # Service-to-service endpoints
            "/api/gemma/generate",
            "/api/gemma/warmup",
            "/api/gemma/chat",
            "/api/gemma/stats",
            # GPU release - validates CSRF from body internally (for sendBeacon)
            "/api/gemma/release-session",
            # Transcript endpoints - internal API calls
            "/api/transcripts/count",
            "/api/transcripts/query",
            "/api/transcripts/recent",
            "/api/transcripts/speakers",
        }
        # Paths that start with these prefixes are exempt
        self.exempt_prefixes: set[str] = {
            "/analytics/",
            "/api/analytics/",
            "/vectorize/",
            "/analyze-full/",
        }
        # Paths that allow Bearer token auth without CSRF
        self.bearer_auth_paths: set[str] = {
            "/api/analyze/stream",
            "/docs",
            "/openapi.json",
            "/redoc",
            "/health",
            "/vectorize/database",
            "/vectorize/status",
            "/upload",
        }
        self.logger = logging.getLogger(__name__)
        self.logger.info("CSRFMiddleware initialized")

    @staticmethod
    def _request_id(request: Request) -> str:
        """Extract or generate request ID for logging.

        Args:
            request: FastAPI request object.

        Returns:
            Request ID string.
        """
        return request.headers.get("X-Request-Id") or getattr(request.state, "request_id", None) or "-"

    async def dispatch(self, request: Request, call_next):
        """Process request through CSRF validation.

        Args:
            request: Incoming HTTP request.
            call_next: Next middleware/handler in chain.

        Returns:
            HTTP response (401/403 if CSRF fails, otherwise from handler).
        """
        # Respect global CSRF toggle
        try:
            if not SecConf.ENABLE_CSRF:
                return await call_next(request)
        except Exception:
            pass

        try:
            # Skip exempt paths
            if request.url.path in self.exempt_paths:
                return await call_next(request)
            if any(request.url.path.startswith(prefix) for prefix in self.exempt_prefixes):
                return await call_next(request)

            if request.method in {"POST", "PUT", "DELETE"}:
                req_id = self._request_id(request)

                # Get session from cookie or Bearer token
                ws_session = request.cookies.get(SESSION_COOKIE_NAME)

                if not ws_session:
                    auth_header = request.headers.get("Authorization", "")
                    if auth_header.startswith("Bearer "):
                        ws_session = auth_header[7:]

                # Import auth_manager lazily to avoid circular imports
                from src.core.dependencies import get_auth_manager

                try:
                    auth_manager = get_auth_manager()
                except Exception:
                    auth_manager = None

                if not auth_manager or not ws_session:
                    self.logger.warning(
                        "[CSRF] not authenticated path=%s rid=%s",
                        request.url.path,
                        req_id,
                    )
                    return Response(
                        content='{"detail":"Not authenticated"}',
                        media_type="application/json",
                        status_code=401,
                    )

                session = auth_manager.validate_session(ws_session)
                if not session:
                    self.logger.warning(
                        "[CSRF] session validation failed path=%s rid=%s",
                        request.url.path,
                        req_id,
                    )
                    return Response(
                        content='{"detail":"Invalid session"}',
                        media_type="application/json",
                        status_code=401,
                    )

                # Store session in request.state for later use
                request.state.session = session

                # Skip CSRF for Bearer auth paths
                if request.url.path in self.bearer_auth_paths:
                    return await call_next(request)

                # Double-submit CSRF check for web clients
                header_token = request.headers.get(CSRF_HEADER_NAME)
                cookie_token = request.cookies.get(CSRF_COOKIE_NAME)
                if (
                    not header_token
                    or not cookie_token
                    or header_token != cookie_token
                    or header_token != session.csrf_token
                ):
                    self.logger.warning(
                        "[CSRF] invalid token path=%s rid=%s",
                        request.url.path,
                        req_id,
                    )
                    return Response(
                        content='{"detail":"CSRF token invalid"}',
                        media_type="application/json",
                        status_code=403,
                    )
        except Exception:
            req_id = self._request_id(request)
            self.logger.exception("[CSRF] unexpected error path=%s rid=%s", request.url.path, req_id)

        return await call_next(request)


class CanonicalHostMiddleware(BaseHTTPMiddleware):
    """Redirect frontend traffic to a single canonical host.

    Ensures all frontend traffic uses a consistent host/port for
    cookies and security headers to work correctly.

    Attributes:
        canonical_host: Target hostname for redirects.
        canonical_port: Target port for redirects (empty for default).
        enabled: Whether canonical host redirection is active.
    """

    def __init__(self, app):
        """Initialize canonical host middleware.

        Args:
            app: FastAPI/Starlette application instance.
        """
        super().__init__(app)
        self.canonical_host = os.getenv("CANONICAL_HOST", "localhost").strip()
        self.canonical_port = os.getenv("CANONICAL_PORT", "").strip()
        self.enabled = bool(self.canonical_host)

    async def dispatch(self, request: Request, call_next):
        """Redirect to canonical host if needed.

        Args:
            request: Incoming HTTP request.
            call_next: Next middleware/handler in chain.

        Returns:
            Redirect response or response from handler.
        """
        if not self.enabled:
            return await call_next(request)

        try:
            if request.method not in {"GET", "HEAD"}:
                return await call_next(request)

            path = request.url.path
            if not (path == "/" or path.startswith("/ui")):
                return await call_next(request)

            host_header = request.headers.get("host")
            if not host_header:
                return await call_next(request)

            hostname, _, port = host_header.partition(":")
            request_port = port or (str(request.url.port) if request.url.port else "")
            request_port = request_port if request_port not in {"80", "443"} else ""

            target_port = self.canonical_port or request_port
            target_port = target_port if target_port and target_port not in {"80", "443"} else ""

            if hostname == self.canonical_host and (not target_port or target_port == request_port):
                return await call_next(request)

            netloc = self.canonical_host
            if target_port:
                netloc = f"{self.canonical_host}:{target_port}"

            redirect_url = request.url.replace(netloc=netloc)
            return RedirectResponse(str(redirect_url), status_code=307)
        except Exception as exc:
            logger.warning(f"[CANONICAL] Redirect handling failed: {exc}")
            return await call_next(request)


def create_audit_middleware(app, audit_logger_instance):
    """Create audit request middleware with the provided logger.

    Factory function to create HTTP middleware that logs all requests
    with timing and anomaly detection.

    Args:
        app: FastAPI application instance.
        audit_logger_instance: Configured audit logger.

    Returns:
        Configured middleware function.
    """

    @app.middleware("http")
    async def audit_request_middleware(request: Request, call_next):
        """Log all requests with timing and anomaly detection.

        Args:
            request: Incoming HTTP request.
            call_next: Next middleware/handler in chain.

        Returns:
            HTTP response from handler.
        """
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        start_time = time.time()
        error_msg = None
        status_code = 500

        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        except Exception as e:
            error_msg = str(e)
            raise
        finally:
            latency_ms = (time.time() - start_time) * 1000
            await audit_logger_instance.log_request(
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                status_code=status_code,
                latency_ms=latency_ms,
                user_id=getattr(request.state, "user_id", None),
                ip_address=request.client.host if request.client else None,
                user_agent=request.headers.get("User-Agent"),
                error=error_msg,
            )

    logger.info("✅ Enterprise audit middleware enabled with anomaly detection")
    return audit_request_middleware
