"""
Case Routes

CRUD operations for Salesforce Case (support ticket) objects.
"""

from fastapi import APIRouter, HTTPException, Query

from ..client import get_client
from ..errors import SalesforceError
from ..models import CaseCreate, CaseUpdate

router = APIRouter(tags=["salesforce-cases"])


@router.get("/cases")
async def list_cases(limit: int = Query(50, ge=1, le=200)):
    """Fetch Salesforce cases."""
    try:
        client = await get_client()
        soql = f"""
            SELECT Id, CaseNumber, Subject, Status, Priority, AccountId, Origin, Type
            FROM Case
            ORDER BY CreatedDate DESC
            LIMIT {limit}
        """
        records = await client.query(soql)
        return {"success": True, "records": records, "total": len(records)}
    except SalesforceError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)


@router.get("/cases/{case_id}")
async def get_case(case_id: str):
    """Get a single Case by ID."""
    try:
        client = await get_client()
        record = await client.get("Case", case_id)
        return {"success": True, "record": record}
    except SalesforceError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)


@router.post("/cases")
async def create_case(case: CaseCreate):
    """Create a new Case (support ticket)."""
    try:
        client = await get_client()
        data = case.model_dump(exclude_none=True)
        result = await client.create("Case", data)
        return {"success": True, "id": result.get("id"), "created": True}
    except SalesforceError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)


@router.patch("/cases/{case_id}")
async def update_case(case_id: str, case: CaseUpdate):
    """Update an existing Case."""
    try:
        client = await get_client()
        data = case.model_dump(exclude_none=True)
        if not data:
            raise HTTPException(status_code=400, detail="No fields to update")
        await client.update("Case", case_id, data)
        return {"success": True, "id": case_id, "updated": True}
    except SalesforceError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)


@router.delete("/cases/{case_id}")
async def delete_case(case_id: str):
    """Delete a Case."""
    try:
        client = await get_client()
        await client.delete("Case", case_id)
        return {"success": True, "id": case_id, "deleted": True}
    except SalesforceError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
