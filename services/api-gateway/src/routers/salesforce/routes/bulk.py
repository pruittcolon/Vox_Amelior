"""
Bulk API Routes

Bulk API 2.0 endpoints for large data operations.
"""

from fastapi import APIRouter, HTTPException

from ..client import get_client
from ..errors import SalesforceError
from ..models import BulkIngestRequest, BulkQueryRequest

router = APIRouter(tags=["salesforce-bulk"])


@router.post("/bulk/query")
async def create_bulk_query(request: BulkQueryRequest):
    """
    Create a Bulk API 2.0 query job for large data exports.

    Use for queries returning 2000+ records.
    Returns job ID for status polling.
    """
    try:
        query_upper = request.query.upper().strip()
        if not query_upper.startswith("SELECT"):
            raise HTTPException(status_code=400, detail="Only SELECT queries allowed")

        client = await get_client()
        job = await client.create_bulk_query_job(request.query)

        return {
            "success": True,
            "job_id": job.get("id"),
            "state": job.get("state"),
            "message": "Query job created. Poll /bulk/status/{job_id} for results.",
        }
    except SalesforceError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)


@router.post("/bulk/ingest")
async def create_bulk_ingest(request: BulkIngestRequest):
    """
    Create a Bulk API 2.0 ingest job for large data operations.

    Supports: insert, update, upsert, delete
    Use for operations on 2000+ records.
    """
    try:
        valid_ops = ("insert", "update", "upsert", "delete")
        if request.operation not in valid_ops:
            raise HTTPException(status_code=400, detail=f"Invalid operation. Must be one of: {valid_ops}")

        client = await get_client()
        job = await client.create_bulk_ingest_job(
            operation=request.operation, object_name=request.object_name, external_id_field=request.external_id_field
        )

        return {
            "success": True,
            "job_id": job.get("id"),
            "state": job.get("state"),
            "object": job.get("object"),
            "operation": job.get("operation"),
        }
    except SalesforceError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)


@router.get("/bulk/status/{job_id}")
async def get_bulk_job_status(job_id: str, job_type: str = "query"):
    """
    Get the status of a bulk job.

    Args:
        job_id: The bulk job ID
        job_type: Either "query" or "ingest"
    """
    try:
        client = await get_client()
        status = await client.get_bulk_job_status(job_id, job_type)

        return {
            "success": True,
            "job_id": job_id,
            "state": status.get("state"),
            "records_processed": status.get("numberRecordsProcessed", 0),
            "records_failed": status.get("numberRecordsFailed", 0),
        }
    except SalesforceError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
