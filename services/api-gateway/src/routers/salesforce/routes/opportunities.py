"""
Opportunity Routes

CRUD operations for Salesforce Opportunity objects.
"""

from fastapi import APIRouter, HTTPException, Query

from ..client import get_client
from ..errors import SalesforceError
from ..models import OpportunityCreate, OpportunityUpdate

router = APIRouter(tags=["salesforce-opportunities"])


@router.get("/opportunities")
async def list_opportunities(
    limit: int = Query(50, ge=1, le=200), stage: str | None = Query(None, description="Filter by stage name")
):
    """
    Fetch Salesforce opportunities.

    Args:
        limit: Maximum number of records
        stage: Optional filter by StageName
    """
    try:
        client = await get_client()
        where = f" WHERE StageName = '{stage}'" if stage else ""
        soql = f"""
            SELECT Id, Name, StageName, Amount, CloseDate, Probability, AccountId
            FROM Opportunity{where}
            ORDER BY CloseDate DESC
            LIMIT {limit}
        """
        records = await client.query(soql)
        return {"success": True, "records": records, "total": len(records)}
    except SalesforceError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)


@router.get("/opportunities/{opp_id}")
async def get_opportunity(opp_id: str):
    """Get a single Opportunity by ID."""
    try:
        client = await get_client()
        record = await client.get("Opportunity", opp_id)
        return {"success": True, "record": record}
    except SalesforceError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)


@router.post("/opportunities")
async def create_opportunity(opportunity: OpportunityCreate):
    """Create a new Opportunity."""
    try:
        client = await get_client()
        data = opportunity.model_dump(exclude_none=True)
        result = await client.create("Opportunity", data)
        return {"success": True, "id": result.get("id"), "created": True}
    except SalesforceError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)


@router.patch("/opportunities/{opp_id}")
async def update_opportunity(opp_id: str, opportunity: OpportunityUpdate):
    """Update an existing Opportunity."""
    try:
        client = await get_client()
        data = opportunity.model_dump(exclude_none=True)
        if not data:
            raise HTTPException(status_code=400, detail="No fields to update")
        await client.update("Opportunity", opp_id, data)
        return {"success": True, "id": opp_id, "updated": True}
    except SalesforceError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)


@router.delete("/opportunities/{opp_id}")
async def delete_opportunity(opp_id: str):
    """Delete an Opportunity."""
    try:
        client = await get_client()
        await client.delete("Opportunity", opp_id)
        return {"success": True, "id": opp_id, "deleted": True}
    except SalesforceError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)


# =============================================================================
# Pipeline Analytics
# =============================================================================


@router.get("/pipeline")
async def get_pipeline():
    """Get pipeline summary with stage breakdown."""
    try:
        client = await get_client()
        soql = "SELECT Id, Name, StageName, Amount, CloseDate FROM Opportunity"
        opps = await client.query(soql)

        stage_summary = {}
        total_value = 0

        for opp in opps:
            stage = opp.get("StageName", "Unknown")
            amount = opp.get("Amount") or 0

            if stage not in stage_summary:
                stage_summary[stage] = {"count": 0, "value": 0}

            stage_summary[stage]["count"] += 1
            stage_summary[stage]["value"] += amount
            total_value += amount

        closed_won = [o for o in opps if o.get("StageName") == "Closed Won"]
        closed_lost = [o for o in opps if o.get("StageName") == "Closed Lost"]
        total_closed = len(closed_won) + len(closed_lost)
        win_rate = (len(closed_won) / total_closed * 100) if total_closed > 0 else 0

        return {
            "success": True,
            "total_opportunities": len(opps),
            "total_pipeline_value": total_value,
            "win_rate": round(win_rate, 1),
            "stages": stage_summary,
        }
    except SalesforceError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)


@router.get("/metrics")
async def get_metrics():
    """Get key CRM metrics for dashboard KPIs."""
    try:
        client = await get_client()

        opps = await client.query("SELECT Id, Name, StageName, Amount, CloseDate FROM Opportunity")

        closed_won = [o for o in opps if o.get("StageName") == "Closed Won"]
        closed_lost = [o for o in opps if o.get("StageName") == "Closed Lost"]
        open_opps = [o for o in opps if "Closed" not in (o.get("StageName") or "")]

        total_revenue = sum(o.get("Amount", 0) or 0 for o in closed_won)
        pipeline_value = sum(o.get("Amount", 0) or 0 for o in open_opps)

        total_closed = len(closed_won) + len(closed_lost)
        win_rate = (len(closed_won) / total_closed * 100) if total_closed > 0 else 0

        return {
            "success": True,
            "metrics": {
                "total_revenue": total_revenue,
                "pipeline_value": pipeline_value,
                "win_rate": round(win_rate, 1),
                "total_opportunities": len(opps),
                "closed_won_count": len(closed_won),
                "open_opportunities": len(open_opps),
                "avg_deal_size": round(total_revenue / len(closed_won), 2) if closed_won else 0,
            },
        }
    except SalesforceError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
