"""
Prometheus metrics for Nemo services.

Phase 3: Observability - HTTP request metrics exposed at /metrics endpoint.

Usage:
    from shared.telemetry.metrics import (
        metrics_router,
        REQUEST_COUNT, REQUEST_LATENCY,
        record_request
    )

    app.include_router(metrics_router)
"""

import logging
import time
from collections.abc import Callable

from fastapi import APIRouter, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

# Try to import prometheus_client, fall back to no-op if not available
try:
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.info("prometheus_client not installed - metrics disabled")


# ============================================================================
# Prometheus Metrics
# ============================================================================

if PROMETHEUS_AVAILABLE:
    REQUEST_COUNT = Counter("http_requests_total", "Total HTTP requests", ["method", "endpoint", "status"])

    REQUEST_LATENCY = Histogram(
        "http_request_duration_seconds",
        "HTTP request latency",
        ["method", "endpoint"],
        buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    )

    ACTIVE_REQUESTS = Gauge("http_active_requests", "Currently active HTTP requests", ["method"])

    GPU_LOCK_WAITING = Gauge("gpu_lock_waiting_services", "Number of services waiting for GPU lock")

    MODEL_LOADED = Gauge(
        "model_loaded", "Whether the model is loaded (1=loaded, 0=not loaded)", ["service", "model_name"]
    )
else:
    # No-op metrics
    class NoOpMetric:
        def labels(self, *args, **kwargs):
            return self

        def inc(self, *args, **kwargs):
            pass

        def dec(self, *args, **kwargs):
            pass

        def observe(self, *args, **kwargs):
            pass

        def set(self, *args, **kwargs):
            pass

    REQUEST_COUNT = NoOpMetric()
    REQUEST_LATENCY = NoOpMetric()
    ACTIVE_REQUESTS = NoOpMetric()
    GPU_LOCK_WAITING = NoOpMetric()
    MODEL_LOADED = NoOpMetric()


def record_request(method: str, endpoint: str, status: int, duration: float) -> None:
    """Record metrics for an HTTP request."""
    if PROMETHEUS_AVAILABLE:
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=str(status)).inc()
        REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(duration)


# ============================================================================
# Metrics Router
# ============================================================================

router = APIRouter(tags=["Metrics"])


@router.get("/metrics")
async def metrics_endpoint() -> Response:
    """Prometheus metrics endpoint."""
    if not PROMETHEUS_AVAILABLE:
        return Response(content="# prometheus_client not installed\n", media_type="text/plain")

    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ============================================================================
# Metrics Middleware
# ============================================================================


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware to automatically record HTTP request metrics."""

    SKIP_PATHS = {"/health", "/ready", "/metrics", "/favicon.ico"}

    async def dispatch(self, request: Request, call_next: Callable):
        if request.url.path in self.SKIP_PATHS:
            return await call_next(request)

        method = request.method

        # Track active requests
        if PROMETHEUS_AVAILABLE:
            ACTIVE_REQUESTS.labels(method=method).inc()

        start_time = time.perf_counter()

        try:
            response = await call_next(request)
            status = response.status_code
        except Exception:
            status = 500
            raise
        finally:
            duration = time.perf_counter() - start_time

            # Normalize endpoint (remove IDs, query params)
            endpoint = self._normalize_endpoint(request.url.path)

            record_request(method, endpoint, status, duration)

            if PROMETHEUS_AVAILABLE:
                ACTIVE_REQUESTS.labels(method=method).dec()

        return response

    @staticmethod
    def _normalize_endpoint(path: str) -> str:
        """Normalize endpoint by replacing IDs with placeholders."""
        import re

        # Replace UUIDs
        path = re.sub(r"/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", "/{id}", path)
        # Replace numeric IDs
        path = re.sub(r"/\d+", "/{id}", path)
        return path
