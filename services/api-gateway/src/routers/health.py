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
