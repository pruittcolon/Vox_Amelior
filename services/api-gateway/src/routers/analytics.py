"""
Analytics Router - Usage, Cost, and SLO Analytics API.

Provides endpoints for:
- Usage summary
- Cost breakdown
- SLO status
- Trend analysis
"""

import logging
from typing import Optional

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/analytics", tags=["analytics"])


# =============================================================================
# RESPONSE MODELS
# =============================================================================


class UsageSummaryResponse(BaseModel):
    """Usage summary response."""

    total_requests: int
    total_cost: float
    period_days: int
    by_service: dict


class CostBreakdownResponse(BaseModel):
    """Cost breakdown response."""

    period_days: int
    total_cost: float
    by_service: dict
    by_tenant: dict
    top_tenants: list


class SLOStatusResponse(BaseModel):
    """SLO status response."""

    slo_name: str
    slo_type: str
    target: float
    current: float
    meeting_slo: bool
    window_days: int


class TrendResponse(BaseModel):
    """Usage trend response."""

    trends: list[dict]
    period_days: int


class HealthSummaryResponse(BaseModel):
    """Health summary with SLO status."""

    status: str
    slo_compliance: dict
    error_budget_warning: bool


# =============================================================================
# ENDPOINTS
# =============================================================================


@router.get("/summary", response_model=UsageSummaryResponse)
async def get_usage_summary(
    tenant_id: Optional[str] = Query(None),
    period_days: int = Query(30, ge=1, le=365),
):
    """
    Get usage summary.

    Args:
        tenant_id: Optional tenant ID filter
        period_days: Period for summary
    """
    try:
        from shared.telemetry.cost_tracker import get_cost_tracker
        from uuid import UUID

        tracker = get_cost_tracker()

        if tenant_id:
            summary = tracker.get_tenant_usage(UUID(tenant_id), period_days)
        else:
            summary = tracker.get_cost_summary(period_days)

        return UsageSummaryResponse(
            total_requests=summary.get("total_requests", 0),
            total_cost=summary.get("total_cost", 0.0),
            period_days=period_days,
            by_service=summary.get("by_service", {}),
        )
    except Exception as e:
        logger.error(f"Error getting usage summary: {e}")
        return UsageSummaryResponse(
            total_requests=0,
            total_cost=0.0,
            period_days=period_days,
            by_service={},
        )


@router.get("/costs", response_model=CostBreakdownResponse)
async def get_cost_breakdown(
    period_days: int = Query(30, ge=1, le=365),
):
    """
    Get cost breakdown.

    Args:
        period_days: Period for breakdown
    """
    try:
        from shared.telemetry.cost_tracker import get_cost_tracker

        tracker = get_cost_tracker()
        summary = tracker.get_cost_summary(period_days)

        return CostBreakdownResponse(
            period_days=period_days,
            total_cost=summary.get("total_cost", 0.0),
            by_service=summary.get("by_service", {}),
            by_tenant=summary.get("by_tenant", {}),
            top_tenants=summary.get("top_tenants", []),
        )
    except Exception as e:
        logger.error(f"Error getting cost breakdown: {e}")
        return CostBreakdownResponse(
            period_days=period_days,
            total_cost=0.0,
            by_service={},
            by_tenant={},
            top_tenants=[],
        )


@router.get("/trends", response_model=TrendResponse)
async def get_usage_trends(
    days: int = Query(7, ge=1, le=30),
):
    """
    Get usage trends.

    Args:
        days: Number of days for trends
    """
    try:
        from shared.telemetry.cost_tracker import get_cost_tracker

        tracker = get_cost_tracker()
        trends = tracker.get_trends(days)

        return TrendResponse(trends=trends, period_days=days)
    except Exception as e:
        logger.error(f"Error getting trends: {e}")
        return TrendResponse(trends=[], period_days=days)


@router.get("/slos", response_model=list[SLOStatusResponse])
async def get_all_slo_status():
    """Get status of all SLOs."""
    try:
        from shared.telemetry.slo_tracker import get_slo_tracker

        tracker = get_slo_tracker()
        statuses = tracker.get_all_slo_status()

        return [
            SLOStatusResponse(
                slo_name=s.get("slo_name", ""),
                slo_type=s.get("slo_type", ""),
                target=s.get("target", 0.0),
                current=s.get("current", 0.0),
                meeting_slo=s.get("meeting_slo", True),
                window_days=s.get("window_days", 30),
            )
            for s in statuses
            if "slo_name" in s
        ]
    except Exception as e:
        logger.error(f"Error getting SLO status: {e}")
        return []


@router.get("/slos/{slo_name}", response_model=SLOStatusResponse)
async def get_slo_status(slo_name: str):
    """Get status of a specific SLO."""
    try:
        from shared.telemetry.slo_tracker import get_slo_tracker

        tracker = get_slo_tracker()
        status = tracker.get_slo_status(slo_name)

        return SLOStatusResponse(
            slo_name=status.get("slo_name", slo_name),
            slo_type=status.get("slo_type", ""),
            target=status.get("target", 0.0),
            current=status.get("current", 0.0),
            meeting_slo=status.get("meeting_slo", True),
            window_days=status.get("window_days", 30),
        )
    except Exception as e:
        logger.error(f"Error getting SLO {slo_name}: {e}")
        return SLOStatusResponse(
            slo_name=slo_name,
            slo_type="unknown",
            target=0.0,
            current=0.0,
            meeting_slo=True,
            window_days=30,
        )


@router.get("/health-summary", response_model=HealthSummaryResponse)
async def get_health_summary():
    """Get health summary with SLO compliance."""
    try:
        from shared.telemetry.slo_tracker import get_slo_tracker

        tracker = get_slo_tracker()
        breaches = tracker.check_slo_breach()

        slo_compliance = {
            "meeting_all": len(breaches) == 0,
            "breaches": len(breaches),
            "breach_details": [b.get("slo_name") for b in breaches],
        }

        # Check error budgets
        warning = False
        for slo_name in ["api_availability", "error_rate"]:
            budget = tracker.get_error_budget(slo_name)
            if budget.get("is_warning", False):
                warning = True

        status = "healthy" if len(breaches) == 0 else "degraded"

        return HealthSummaryResponse(
            status=status,
            slo_compliance=slo_compliance,
            error_budget_warning=warning,
        )
    except Exception as e:
        logger.error(f"Error getting health summary: {e}")
        return HealthSummaryResponse(
            status="unknown",
            slo_compliance={"meeting_all": True, "breaches": 0},
            error_budget_warning=False,
        )


logger.info("✅ Analytics Router initialized with usage and SLO endpoints")
