"""
Query & Metadata Routes

SOQL query execution, Composite API, and metadata endpoints.
"""

from fastapi import APIRouter, HTTPException

from ..client import get_client
from ..config import SALESFORCE_ENABLED, get_config
from ..errors import SalesforceError
from ..models import CompositeRequest, SOQLRequest

router = APIRouter(tags=["salesforce-query"])


# =============================================================================
# Status
# =============================================================================


@router.get("/status")
async def get_status():
    """Get Salesforce connection status and configuration."""
    if not SALESFORCE_ENABLED:
        return {"connected": False, "enabled": False, "message": "Salesforce integration is disabled"}

    config = get_config()
    if config is None:
        return {
            "connected": False,
            "enabled": True,
            "configured": False,
            "message": "Set SALESFORCE_CLIENT_ID, SALESFORCE_CLIENT_SECRET, SALESFORCE_DOMAIN",
        }

    try:
        client = await get_client()
        return {
            "connected": True,
            "enabled": True,
            "configured": True,
            "instance_url": client.instance_url,
            "api_version": client.config.api_version,
            "request_count": client._request_count,
        }
    except Exception as e:
        return {"connected": False, "enabled": True, "configured": True, "error": str(e)}


# =============================================================================
# SOQL Query
# =============================================================================


@router.post("/query")
async def execute_query(request: SOQLRequest):
    """
    Execute a custom SOQL query.

    Only SELECT queries are allowed for security.
    For DML operations, use the appropriate CRUD endpoints.
    """
    try:
        query = request.query.strip()
        query_upper = query.upper()

        # Security: Only allow SELECT queries
        if not query_upper.startswith("SELECT"):
            raise HTTPException(status_code=403, detail="Only SELECT queries allowed")

        # Block dangerous keywords
        dangerous = ["DELETE", "UPDATE", "INSERT", "MERGE", "DROP", "TRUNCATE"]
        if any(word in query_upper for word in dangerous):
            raise HTTPException(status_code=403, detail="Dangerous keywords not allowed")

        client = await get_client()
        records = await client.query(query)

        return {"success": True, "records": records, "total": len(records)}

    except SalesforceError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)


# =============================================================================
# Composite API
# =============================================================================


@router.post("/composite")
async def execute_composite(request: CompositeRequest):
    """
    Execute multiple API operations in a single request.

    Supports up to 25 subrequests per call.
    Results from earlier requests can be referenced in later requests.

    Example subrequest:
    {
        "method": "POST",
        "url": "/services/data/v59.0/sobjects/Account",
        "referenceId": "newAccount",
        "body": {"Name": "New Corp"}
    }
    """
    try:
        if len(request.subrequests) > 25:
            raise HTTPException(status_code=400, detail="Maximum 25 subrequests allowed")

        # Convert Pydantic models to dicts
        subrequests = [sr.model_dump(exclude_none=True) for sr in request.subrequests]

        client = await get_client()
        result = await client.composite(subrequests, all_or_none=request.allOrNone)

        return {"success": True, "result": result}

    except SalesforceError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)


# =============================================================================
# Metadata
# =============================================================================


@router.get("/describe/{object_name}")
async def describe_object(object_name: str):
    """
    Get metadata about a Salesforce object.

    Returns field definitions, relationships, and object properties.
    """
    try:
        client = await get_client()
        metadata = await client.describe(object_name)

        return {
            "success": True,
            "name": metadata.get("name"),
            "label": metadata.get("label"),
            "fields": [
                {
                    "name": f["name"],
                    "type": f["type"],
                    "label": f["label"],
                    "required": not f.get("nillable", True) and f.get("createable", False),
                }
                for f in metadata.get("fields", [])[:100]
            ],
            "total_fields": len(metadata.get("fields", [])),
        }
    except SalesforceError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)


@router.get("/objects")
async def list_objects():
    """List all available Salesforce objects."""
    try:
        client = await get_client()
        objects = await client.list_objects()
        return {"success": True, "objects": objects, "total": len(objects)}
    except SalesforceError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
