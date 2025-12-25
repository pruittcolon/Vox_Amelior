"""
Account Routes

CRUD operations for Salesforce Account objects.
"""

from fastapi import APIRouter, HTTPException, Query

from ..client import get_client
from ..errors import SalesforceError
from ..models import AccountCreate, AccountUpdate

router = APIRouter(tags=["salesforce-accounts"])


@router.get("/accounts")
async def list_accounts(limit: int = Query(50, ge=1, le=200)):
    """
    Fetch Salesforce accounts.

    Args:
        limit: Maximum number of records to return (1-200)
    """
    try:
        client = await get_client()
        soql = f"""
            SELECT Id, Name, Industry, AnnualRevenue, BillingCity, 
                   BillingState, Phone, Website, NumberOfEmployees
            FROM Account 
            ORDER BY Name 
            LIMIT {limit}
        """
        records = await client.query(soql)
        return {"success": True, "records": records, "total": len(records)}
    except SalesforceError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)


@router.get("/accounts/{account_id}")
async def get_account(account_id: str):
    """Get a single Account by ID."""
    try:
        client = await get_client()
        record = await client.get("Account", account_id)
        return {"success": True, "record": record}
    except SalesforceError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)


@router.post("/accounts")
async def create_account(account: AccountCreate):
    """Create a new Account."""
    try:
        client = await get_client()
        data = account.model_dump(exclude_none=True)
        result = await client.create("Account", data)
        return {"success": True, "id": result.get("id"), "created": True}
    except SalesforceError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)


@router.patch("/accounts/{account_id}")
async def update_account(account_id: str, account: AccountUpdate):
    """Update an existing Account."""
    try:
        client = await get_client()
        data = account.model_dump(exclude_none=True)
        if not data:
            raise HTTPException(status_code=400, detail="No fields to update")
        await client.update("Account", account_id, data)
        return {"success": True, "id": account_id, "updated": True}
    except SalesforceError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)


@router.delete("/accounts/{account_id}")
async def delete_account(account_id: str):
    """Delete an Account."""
    try:
        client = await get_client()
        await client.delete("Account", account_id)
        return {"success": True, "id": account_id, "deleted": True}
    except SalesforceError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
