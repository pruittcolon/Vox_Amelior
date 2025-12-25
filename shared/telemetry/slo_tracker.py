"""
SLO Tracker - Service Level Objective Monitoring.

Tracks SLIs and calculates error budgets for SLO compliance.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional


logger = logging.getLogger(__name__)


class SLOType(str, Enum):
    """Types of SLOs."""

    AVAILABILITY = "availability"
    LATENCY = "latency"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"


@dataclass
class SLODefinition:
    """Definition of an SLO."""

    name: str
    slo_type: SLOType
    target: float  # e.g., 0.999 for 99.9%
    window_days: int = 30
    description: str = ""

    @property
    def error_budget_pct(self) -> float:
        """Error budget as percentage."""
        return (1.0 - self.target) * 100


@dataclass
class SLIRecord:
    """A single SLI measurement."""

    endpoint: str
    latency_ms: float
    success: bool
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict = field(default_factory=dict)


# Default SLO definitions
DEFAULT_SLOS: list[SLODefinition] = [
    SLODefinition(
        name="api_availability",
        slo_type=SLOType.AVAILABILITY,
        target=0.999,  # 99.9%
        description="API endpoints should be available 99.9% of the time",
    ),
    SLODefinition(
        name="api_latency_p95",
        slo_type=SLOType.LATENCY,
        target=500.0,  # 500ms
        description="95th percentile API latency should be under 500ms",
    ),
    SLODefinition(
        name="ai_latency_p95",
        slo_type=SLOType.LATENCY,
        target=2000.0,  # 2s
        description="95th percentile AI endpoint latency should be under 2s",
    ),
    SLODefinition(
        name="error_rate",
        slo_type=SLOType.ERROR_RATE,
        target=0.01,  # 1%
        description="Error rate should be under 1%",
    ),
]


class SLOTracker:
    """
    Track SLIs and monitor SLO compliance.

    Usage:
        tracker = SLOTracker()
        tracker.record_request("/api/v1/health", latency_ms=50, success=True)
        status = tracker.get_slo_status("api_availability")
    """

    def __init__(self, slos: Optional[list[SLODefinition]] = None):
        """
        Initialize SLO tracker.

        Args:
            slos: Optional list of SLO definitions
        """
        self.slos = {s.name: s for s in (slos or DEFAULT_SLOS)}
        self._records: list[SLIRecord] = []
        self._endpoint_stats: dict[str, dict] = defaultdict(
            lambda: {"total": 0, "success": 0, "latencies": []}
        )

    def record_request(
        self,
        endpoint: str,
        latency_ms: float,
        success: bool,
        **metadata,
    ) -> SLIRecord:
        """
        Record a request for SLI measurement.

        Args:
            endpoint: API endpoint
            latency_ms: Request latency
            success: Whether request was successful

        Returns:
            SLIRecord
        """
        record = SLIRecord(
            endpoint=endpoint,
            latency_ms=latency_ms,
            success=success,
            metadata=metadata,
        )

        self._records.append(record)

        # Update aggregates
        stats = self._endpoint_stats[endpoint]
        stats["total"] += 1
        if success:
            stats["success"] += 1
        stats["latencies"].append(latency_ms)

        # Keep only last 10000 latencies per endpoint
        if len(stats["latencies"]) > 10000:
            stats["latencies"] = stats["latencies"][-10000:]

        return record

    def get_slo_status(self, slo_name: str) -> dict:
        """
        Get current status for an SLO.

        Args:
            slo_name: Name of the SLO

        Returns:
            SLO status with compliance info
        """
        slo = self.slos.get(slo_name)
        if not slo:
            return {"error": f"SLO not found: {slo_name}"}

        cutoff = datetime.utcnow() - timedelta(days=slo.window_days)
        records = [r for r in self._records if r.timestamp >= cutoff]

        if not records:
            return {
                "slo_name": slo_name,
                "status": "no_data",
                "message": "No data in window",
            }

        if slo.slo_type == SLOType.AVAILABILITY:
            return self._calculate_availability_slo(slo, records)
        elif slo.slo_type == SLOType.LATENCY:
            return self._calculate_latency_slo(slo, records)
        elif slo.slo_type == SLOType.ERROR_RATE:
            return self._calculate_error_rate_slo(slo, records)
        else:
            return {"error": f"Unknown SLO type: {slo.slo_type}"}

    def _calculate_availability_slo(
        self, slo: SLODefinition, records: list[SLIRecord]
    ) -> dict:
        """Calculate availability SLO status."""
        total = len(records)
        successful = sum(1 for r in records if r.success)
        availability = successful / total if total > 0 else 0.0

        meeting_slo = availability >= slo.target
        error_budget_remaining = (availability - slo.target) / (1 - slo.target)
        error_budget_remaining = max(0, min(1, error_budget_remaining))

        return {
            "slo_name": slo.name,
            "slo_type": slo.slo_type.value,
            "target": slo.target,
            "current": round(availability, 4),
            "meeting_slo": meeting_slo,
            "error_budget_remaining_pct": round(error_budget_remaining * 100, 2),
            "window_days": slo.window_days,
            "total_requests": total,
            "successful_requests": successful,
        }

    def _calculate_latency_slo(
        self, slo: SLODefinition, records: list[SLIRecord]
    ) -> dict:
        """Calculate latency SLO status (P95)."""
        latencies = sorted([r.latency_ms for r in records])

        if not latencies:
            return {"slo_name": slo.name, "status": "no_data"}

        # Calculate P95
        p95_index = int(len(latencies) * 0.95)
        p95 = latencies[p95_index] if p95_index < len(latencies) else latencies[-1]

        meeting_slo = p95 <= slo.target

        return {
            "slo_name": slo.name,
            "slo_type": slo.slo_type.value,
            "target_ms": slo.target,
            "current_p95_ms": round(p95, 2),
            "meeting_slo": meeting_slo,
            "window_days": slo.window_days,
            "total_requests": len(records),
            "p50_ms": round(latencies[len(latencies) // 2], 2),
            "p99_ms": round(latencies[int(len(latencies) * 0.99)], 2),
        }

    def _calculate_error_rate_slo(
        self, slo: SLODefinition, records: list[SLIRecord]
    ) -> dict:
        """Calculate error rate SLO status."""
        total = len(records)
        errors = sum(1 for r in records if not r.success)
        error_rate = errors / total if total > 0 else 0.0

        meeting_slo = error_rate <= slo.target

        return {
            "slo_name": slo.name,
            "slo_type": slo.slo_type.value,
            "target": slo.target,
            "current": round(error_rate, 4),
            "meeting_slo": meeting_slo,
            "window_days": slo.window_days,
            "total_requests": total,
            "error_count": errors,
        }

    def get_all_slo_status(self) -> list[dict]:
        """Get status for all defined SLOs."""
        return [self.get_slo_status(name) for name in self.slos.keys()]

    def check_slo_breach(self) -> list[dict]:
        """Check for any SLO breaches."""
        breaches = []
        for status in self.get_all_slo_status():
            if status.get("meeting_slo") is False:
                breaches.append(status)
        return breaches

    def get_error_budget(self, slo_name: str) -> dict:
        """Get error budget status for an SLO."""
        status = self.get_slo_status(slo_name)

        if "error_budget_remaining_pct" in status:
            return {
                "slo_name": slo_name,
                "error_budget_remaining_pct": status["error_budget_remaining_pct"],
                "is_critical": status["error_budget_remaining_pct"] < 20,
                "is_warning": status["error_budget_remaining_pct"] < 50,
            }

        return {"slo_name": slo_name, "status": "not_applicable"}


# Singleton instance
_tracker: Optional[SLOTracker] = None


def get_slo_tracker() -> SLOTracker:
    """Get or create singleton SLO tracker."""
    global _tracker
    if _tracker is None:
        _tracker = SLOTracker()
    return _tracker
