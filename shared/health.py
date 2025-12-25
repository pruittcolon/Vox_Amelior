"""
Health Check Utilities - Production Readiness.

This module provides standardized health check responses for all services,
following Kubernetes readiness/liveness probe patterns.

Phase 24 of ultimateseniordevplan.md.
"""

import asyncio
from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, Field


class HealthStatus(str, Enum):
    """Health check status values."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ComponentHealth(BaseModel):
    """Health status for an individual component.

    Attributes:
        name: Component identifier.
        status: Current health status.
        message: Optional status message.
        latency_ms: Response latency in milliseconds.
    """

    name: str
    status: HealthStatus
    message: str | None = None
    latency_ms: float | None = None


class HealthResponse(BaseModel):
    """Standardized health check response.

    Compatible with Kubernetes liveness and readiness probes.

    Attributes:
        status: Overall service health status.
        service: Name of the service.
        version: Service version.
        timestamp: Time of health check.
        components: Health of individual dependencies.
        uptime_seconds: Service uptime in seconds.
    """

    status: HealthStatus
    service: str
    version: str = "1.0.0"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    components: list[ComponentHealth] = Field(default_factory=list)
    uptime_seconds: float = 0.0

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "healthy",
                "service": "api-gateway",
                "version": "1.0.0",
                "timestamp": "2024-12-20T12:00:00Z",
                "components": [
                    {"name": "database", "status": "healthy", "latency_ms": 5.2},
                    {"name": "redis", "status": "healthy", "latency_ms": 1.1},
                ],
                "uptime_seconds": 3600.0,
            }
        }
    }


class HealthChecker:
    """Service health checker with component checks.

    Performs async health checks against service dependencies
    and aggregates results into a single health response.

    Attributes:
        service_name: Name of the service.
        version: Service version string.
        start_time: Service start time for uptime calculation.
    """

    def __init__(self, service_name: str, version: str = "1.0.0"):
        """Initialize health checker.

        Args:
            service_name: Name of the service.
            version: Version string.
        """
        self.service_name = service_name
        self.version = version
        self.start_time = datetime.now(UTC)
        self._checks: dict[str, callable] = {}

    def register_check(self, name: str, check_fn: callable) -> None:
        """Register a component health check.

        Args:
            name: Component name.
            check_fn: Async function returning (bool, Optional[str]).
        """
        self._checks[name] = check_fn

    async def _run_check(self, name: str, check_fn: callable) -> ComponentHealth:
        """Run a single health check.

        Args:
            name: Component name.
            check_fn: Check function.

        Returns:
            ComponentHealth: Check result.
        """
        start = asyncio.get_event_loop().time()
        try:
            result = await check_fn()
            if isinstance(result, tuple):
                is_healthy, message = result
            else:
                is_healthy, message = result, None

            latency_ms = (asyncio.get_event_loop().time() - start) * 1000

            return ComponentHealth(
                name=name,
                status=HealthStatus.HEALTHY if is_healthy else HealthStatus.UNHEALTHY,
                message=message,
                latency_ms=round(latency_ms, 2),
            )
        except Exception as e:
            latency_ms = (asyncio.get_event_loop().time() - start) * 1000
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=str(e),
                latency_ms=round(latency_ms, 2),
            )

    async def check_health(self) -> HealthResponse:
        """Perform all health checks.

        Returns:
            HealthResponse: Aggregated health status.
        """
        # Run all checks concurrently
        check_tasks = [self._run_check(name, fn) for name, fn in self._checks.items()]
        components = await asyncio.gather(*check_tasks)

        # Determine overall status
        if all(c.status == HealthStatus.HEALTHY for c in components):
            overall_status = HealthStatus.HEALTHY
        elif any(c.status == HealthStatus.UNHEALTHY for c in components):
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY

        uptime = (datetime.now(UTC) - self.start_time).total_seconds()

        return HealthResponse(
            status=overall_status,
            service=self.service_name,
            version=self.version,
            components=list(components),
            uptime_seconds=round(uptime, 2),
        )


# Example check functions
async def check_database_connection() -> tuple:
    """Check database connectivity.

    Returns:
        tuple: (is_healthy, message)
    """
    # Implementation would check actual DB connection
    return True, "Connected"


async def check_redis_connection() -> tuple:
    """Check Redis connectivity.

    Returns:
        tuple: (is_healthy, message)
    """
    # Implementation would check actual Redis connection
    return True, "Connected"


async def check_gpu_available() -> tuple:
    """Check GPU availability.

    Returns:
        tuple: (is_healthy, message)
    """
    # Implementation would check CUDA/GPU status
    return True, "GPU available"
