"""
Cost Tracker - FinOps Usage and Cost Analytics.

Tracks per-tenant, per-service usage for cost attribution.
Integrates with existing OpenTelemetry infrastructure.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional
from uuid import UUID


logger = logging.getLogger(__name__)


class ServiceType(str, Enum):
    """Trackable service types."""

    API = "api"
    AI_CHAT = "ai_chat"
    AI_EMBEDDING = "ai_embedding"
    TRANSCRIPTION = "transcription"
    STORAGE = "storage"
    COMPUTE = "compute"


@dataclass
class UsageRecord:
    """A single usage record."""

    tenant_id: Optional[UUID] = None
    service: ServiceType = ServiceType.API
    tokens_in: int = 0
    tokens_out: int = 0
    requests: int = 1
    latency_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        """Total tokens (input + output)."""
        return self.tokens_in + self.tokens_out


@dataclass
class CostConfig:
    """Cost configuration per service."""

    cost_per_1k_tokens_in: float = 0.0
    cost_per_1k_tokens_out: float = 0.0
    cost_per_request: float = 0.0
    cost_per_gb_storage: float = 0.0


# Default cost configs (local models = $0, storage = nominal)
DEFAULT_COSTS: dict[ServiceType, CostConfig] = {
    ServiceType.API: CostConfig(cost_per_request=0.0),
    ServiceType.AI_CHAT: CostConfig(
        cost_per_1k_tokens_in=0.0,  # Local Gemma
        cost_per_1k_tokens_out=0.0,
    ),
    ServiceType.AI_EMBEDDING: CostConfig(cost_per_1k_tokens_in=0.0),
    ServiceType.TRANSCRIPTION: CostConfig(cost_per_request=0.0),
    ServiceType.STORAGE: CostConfig(cost_per_gb_storage=0.01),
    ServiceType.COMPUTE: CostConfig(cost_per_request=0.0),
}


class CostTracker:
    """
    Track usage and costs per tenant and service.

    Usage:
        tracker = CostTracker()
        tracker.track_request(tenant_id, ServiceType.AI_CHAT, tokens_in=100, tokens_out=50)
        summary = tracker.get_tenant_usage(tenant_id)
    """

    def __init__(self, costs: Optional[dict[ServiceType, CostConfig]] = None):
        """
        Initialize cost tracker.

        Args:
            costs: Optional cost configuration overrides
        """
        self.costs = costs or DEFAULT_COSTS
        self._records: list[UsageRecord] = []
        self._tenant_usage: dict[UUID, dict[ServiceType, dict]] = defaultdict(
            lambda: defaultdict(lambda: {
                "requests": 0,
                "tokens_in": 0,
                "tokens_out": 0,
                "latency_total_ms": 0.0,
                "cost": 0.0,
            })
        )

    def track_request(
        self,
        tenant_id: Optional[UUID],
        service: ServiceType,
        tokens_in: int = 0,
        tokens_out: int = 0,
        latency_ms: float = 0.0,
        **metadata,
    ) -> UsageRecord:
        """
        Track a single request.

        Args:
            tenant_id: Tenant UUID
            service: Service type
            tokens_in: Input tokens
            tokens_out: Output tokens
            latency_ms: Request latency
            **metadata: Additional metadata

        Returns:
            UsageRecord
        """
        record = UsageRecord(
            tenant_id=tenant_id,
            service=service,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            metadata=metadata,
        )

        self._records.append(record)

        # Update aggregates
        if tenant_id:
            usage = self._tenant_usage[tenant_id][service]
            usage["requests"] += 1
            usage["tokens_in"] += tokens_in
            usage["tokens_out"] += tokens_out
            usage["latency_total_ms"] += latency_ms
            usage["cost"] += self._calculate_cost(record)

        return record

    def _calculate_cost(self, record: UsageRecord) -> float:
        """Calculate cost for a usage record."""
        config = self.costs.get(record.service, CostConfig())

        cost = config.cost_per_request
        cost += (record.tokens_in / 1000) * config.cost_per_1k_tokens_in
        cost += (record.tokens_out / 1000) * config.cost_per_1k_tokens_out

        return cost

    def get_tenant_usage(
        self,
        tenant_id: UUID,
        period_days: int = 30,
    ) -> dict:
        """
        Get usage summary for a tenant.

        Args:
            tenant_id: Tenant UUID
            period_days: Period to summarize

        Returns:
            Usage summary dict
        """
        cutoff = datetime.utcnow() - timedelta(days=period_days)

        # Filter records for tenant and period
        records = [
            r for r in self._records
            if r.tenant_id == tenant_id and r.timestamp >= cutoff
        ]

        by_service = defaultdict(lambda: {
            "requests": 0,
            "tokens_in": 0,
            "tokens_out": 0,
            "avg_latency_ms": 0.0,
            "cost": 0.0,
        })

        for record in records:
            svc = by_service[record.service.value]
            svc["requests"] += 1
            svc["tokens_in"] += record.tokens_in
            svc["tokens_out"] += record.tokens_out
            svc["cost"] += self._calculate_cost(record)

        # Calculate averages
        for svc in by_service.values():
            if svc["requests"] > 0:
                matching = [r for r in records if r.latency_ms > 0]
                if matching:
                    svc["avg_latency_ms"] = sum(r.latency_ms for r in matching) / len(matching)

        total_cost = sum(s["cost"] for s in by_service.values())
        total_requests = sum(s["requests"] for s in by_service.values())

        return {
            "tenant_id": str(tenant_id),
            "period_days": period_days,
            "total_requests": total_requests,
            "total_cost": round(total_cost, 4),
            "by_service": dict(by_service),
        }

    def get_cost_summary(self, period_days: int = 30) -> dict:
        """
        Get overall cost summary across all tenants.

        Args:
            period_days: Period to summarize

        Returns:
            Cost summary dict
        """
        cutoff = datetime.utcnow() - timedelta(days=period_days)

        records = [r for r in self._records if r.timestamp >= cutoff]

        by_service = defaultdict(lambda: {"requests": 0, "cost": 0.0})
        by_tenant = defaultdict(lambda: {"requests": 0, "cost": 0.0})

        for record in records:
            cost = self._calculate_cost(record)

            by_service[record.service.value]["requests"] += 1
            by_service[record.service.value]["cost"] += cost

            if record.tenant_id:
                by_tenant[str(record.tenant_id)]["requests"] += 1
                by_tenant[str(record.tenant_id)]["cost"] += cost

        total_cost = sum(s["cost"] for s in by_service.values())
        total_requests = sum(s["requests"] for s in by_service.values())

        return {
            "period_days": period_days,
            "total_requests": total_requests,
            "total_cost": round(total_cost, 4),
            "by_service": dict(by_service),
            "by_tenant": dict(by_tenant),
            "top_tenants": sorted(
                by_tenant.items(),
                key=lambda x: x[1]["cost"],
                reverse=True,
            )[:10],
        }

    def get_trends(self, days: int = 7) -> list[dict]:
        """Get daily usage trends."""
        today = datetime.utcnow().date()
        trends = []

        for i in range(days):
            day = today - timedelta(days=i)
            day_start = datetime.combine(day, datetime.min.time())
            day_end = datetime.combine(day, datetime.max.time())

            day_records = [
                r for r in self._records
                if day_start <= r.timestamp <= day_end
            ]

            trends.append({
                "date": day.isoformat(),
                "requests": len(day_records),
                "cost": round(sum(self._calculate_cost(r) for r in day_records), 4),
            })

        return list(reversed(trends))


# Singleton instance
_tracker: Optional[CostTracker] = None


def get_cost_tracker() -> CostTracker:
    """Get or create singleton cost tracker."""
    global _tracker
    if _tracker is None:
        _tracker = CostTracker()
    return _tracker
