"""
Contact Routes

CRUD operations for Salesforce Contact objects.
"""

from fastapi import APIRouter, HTTPException, Query

from ..client import get_client
from ..errors import SalesforceError
from ..models import ContactCreate, ContactUpdate

router = APIRouter(tags=["salesforce-contacts"])


@router.get("/contacts")
async def list_contacts(
    limit: int = Query(50, ge=1, le=200), account_id: str | None = Query(None, description="Filter by Account ID")
):
    """
    Fetch Salesforce contacts.

    Args:
        limit: Maximum number of records to return
        account_id: Optional filter by parent Account
    """
    try:
        client = await get_client()
        where = f" WHERE AccountId = '{account_id}'" if account_id else ""
        soql = f"""
            SELECT Id, FirstName, LastName, Email, Title, Phone, AccountId, Department
            FROM Contact{where}
            ORDER BY LastName
            LIMIT {limit}
        """
        records = await client.query(soql)
        return {"success": True, "records": records, "total": len(records)}
    except SalesforceError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)


@router.get("/contacts/{contact_id}")
async def get_contact(contact_id: str):
    """Get a single Contact by ID."""
    try:
        client = await get_client()
        record = await client.get("Contact", contact_id)
        return {"success": True, "record": record}
    except SalesforceError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)


@router.post("/contacts")
async def create_contact(contact: ContactCreate):
    """Create a new Contact."""
    try:
        client = await get_client()
        data = contact.model_dump(exclude_none=True)
        result = await client.create("Contact", data)
        return {"success": True, "id": result.get("id"), "created": True}
    except SalesforceError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)


@router.patch("/contacts/{contact_id}")
async def update_contact(contact_id: str, contact: ContactUpdate):
    """Update an existing Contact."""
    try:
        client = await get_client()
        data = contact.model_dump(exclude_none=True)
        if not data:
            raise HTTPException(status_code=400, detail="No fields to update")
        await client.update("Contact", contact_id, data)
        return {"success": True, "id": contact_id, "updated": True}
    except SalesforceError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)


@router.delete("/contacts/{contact_id}")
async def delete_contact(contact_id: str):
    """Delete a Contact."""
    try:
        client = await get_client()
        await client.delete("Contact", contact_id)
        return {"success": True, "id": contact_id, "deleted": True}
    except SalesforceError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
