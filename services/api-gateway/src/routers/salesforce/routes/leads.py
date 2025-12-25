"""
Lead Routes

CRUD operations for Salesforce Lead objects.
"""

from fastapi import APIRouter, HTTPException, Query

from ..client import get_client
from ..errors import SalesforceError
from ..models import LeadCreate, LeadUpdate

router = APIRouter(tags=["salesforce-leads"])


@router.get("/leads")
async def list_leads(limit: int = Query(50, ge=1, le=200)):
    """Fetch Salesforce leads."""
    try:
        client = await get_client()
        soql = f"""
            SELECT Id, FirstName, LastName, Company, Status, Email, Phone, LeadSource, Industry
            FROM Lead
            ORDER BY CreatedDate DESC
            LIMIT {limit}
        """
        records = await client.query(soql)
        return {"success": True, "records": records, "total": len(records)}
    except SalesforceError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)


@router.get("/leads/{lead_id}")
async def get_lead(lead_id: str):
    """Get a single Lead by ID."""
    try:
        client = await get_client()
        record = await client.get("Lead", lead_id)
        return {"success": True, "record": record}
    except SalesforceError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)


@router.post("/leads")
async def create_lead(lead: LeadCreate):
    """Create a new Lead."""
    try:
        client = await get_client()
        data = lead.model_dump(exclude_none=True)
        result = await client.create("Lead", data)
        return {"success": True, "id": result.get("id"), "created": True}
    except SalesforceError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)


@router.patch("/leads/{lead_id}")
async def update_lead(lead_id: str, lead: LeadUpdate):
    """Update an existing Lead."""
    try:
        client = await get_client()
        data = lead.model_dump(exclude_none=True)
        if not data:
            raise HTTPException(status_code=400, detail="No fields to update")
        await client.update("Lead", lead_id, data)
        return {"success": True, "id": lead_id, "updated": True}
    except SalesforceError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)


@router.delete("/leads/{lead_id}")
async def delete_lead(lead_id: str):
    """Delete a Lead."""
    try:
        client = await get_client()
        await client.delete("Lead", lead_id)
        return {"success": True, "id": lead_id, "deleted": True}
    except SalesforceError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
