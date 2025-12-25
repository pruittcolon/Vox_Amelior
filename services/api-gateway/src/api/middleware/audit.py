"""
Audit Logging Middleware.

Logs all HTTP requests with timing, status codes, and anomaly detection.
Integrates with shared.audit for enterprise-grade logging.

Phase 2 of API Restructure.
"""

import logging
import time
import uuid

from config.settings import settings
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

# Try to import audit logger (optional if not needed in tests)
audit_logger_instance = None
if settings.AUDIT_ENABLED:
    try:
        from shared.audit import get_audit_logger

        audit_logger_instance = get_audit_logger()
        logger.info("✅ Enterprise audit logger initialized")
    except ImportError as e:
        logger.warning(f"Audit logger not available: {e}")


class AuditMiddleware(BaseHTTPMiddleware):
    """Logs all HTTP requests with timing and anomaly detection."""

    def __init__(self, app):
        super().__init__(app)
        self.enabled = settings.AUDIT_ENABLED
        self.audit_logger = audit_logger_instance

    async def dispatch(self, request: Request, call_next):
        if not self.enabled or not self.audit_logger:
            return await call_next(request)

        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        start_time = time.time()
        error_msg: str | None = None
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
            await self.audit_logger.log_request(
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


def get_audit_logger_instance():
    """Get the audit logger instance for use in routes."""
    return audit_logger_instance
