"""
Health check router.

Provides /health, /ready, and /api/audit/status endpoints for container orchestration.
"""

import logging
import os
from typing import Any

from fastapi import APIRouter

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Health"])

# Audit configuration
AUDIT_ENABLED = os.getenv("AUDIT_ENABLED", "true").lower() in {"1", "true", "yes"}
audit_logger_instance = None


# Lazy load audit logger
def _get_audit_logger():
    global audit_logger_instance
    if audit_logger_instance is None and AUDIT_ENABLED:
        try:
            from shared.audit import get_audit_logger

            audit_logger_instance = get_audit_logger()
        except ImportError:
            logger.warning("Audit logger not available")
    return audit_logger_instance


@router.get("/health")
def health() -> dict[str, Any]:
    """Basic health check for container orchestration."""
    return {"status": "healthy", "service": "api-gateway"}


@router.get("/api/health")
def api_health() -> dict[str, Any]:
    """Alias for legacy frontend that prefixes requests with /api."""
    return health()


@router.get("/ready")
def ready() -> dict[str, Any]:
    """Readiness check - indicates service is ready to accept traffic."""
    return {"status": "ready", "service": "api-gateway"}


@router.get("/api/audit/status")
async def audit_status() -> dict[str, Any]:
    """Get audit logging status and statistics - for CLI testing."""
    audit = _get_audit_logger()
    if not AUDIT_ENABLED or audit is None:
        return {"enabled": False, "message": "Audit logging disabled"}

    stats = audit.get_stats()
    anomalies = audit.get_anomalies()

    return {
        "enabled": True,
        "stats": stats,
        "anomalies": [
            {
                "type": a.anomaly_type,
                "path": a.path,
                "description": a.description,
                "severity": a.severity,
                "timestamp": a.timestamp,
            }
            for a in anomalies[-10:]  # Last 10 anomalies
        ],
        "thresholds": audit.thresholds,
    }


# =============================================================================
# GPU Status Proxy Endpoints (for frontend monitoring)
# =============================================================================

ML_SERVICE_URL = os.getenv("ML_SERVICE_URL", "http://ml-service:8006")
GPU_COORDINATOR_URL = os.getenv("GPU_COORDINATOR_URL", "http://gpu-coordinator:8002")


@router.get("/api/ml/gpu-status")
async def ml_gpu_status() -> dict[str, Any]:
    """Proxy GPU status from ML service."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{ML_SERVICE_URL}/gpu-status")
            if response.status_code == 200:
                return response.json()
            return {"cuda_available": False, "error": "ML service unavailable"}
    except Exception as e:
        logger.warning(f"GPU status check failed: {e}")
        return {"cuda_available": False, "error": str(e)}


@router.get("/api/gpu-coordinator/gpu/state")
async def gpu_coordinator_state() -> dict[str, Any]:
    """Proxy GPU state from coordinator."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{GPU_COORDINATOR_URL}/gpu/state")
            if response.status_code == 200:
                return response.json()
            return {"owner": "unknown", "available": False}
    except Exception as e:
        logger.warning(f"GPU coordinator state check failed: {e}")
        return {"owner": "unknown", "available": False, "error": str(e)}


@router.get("/api/gpu-coordinator/status")
async def gpu_coordinator_status() -> dict[str, Any]:
    """Proxy status from GPU coordinator health endpoint."""
    import httpx
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{GPU_COORDINATOR_URL}/health")
            if response.status_code == 200:
                return response.json()
            return {"status": "unavailable"}
    except Exception as e:
        logger.warning(f"GPU coordinator status check failed: {e}")
        return {"status": "unavailable", "error": str(e)}
