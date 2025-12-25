"""
Transcripts Router - Transcript query and analytics endpoints.

Provides transcript searching, filtering, pagination, and analytics signals/segments.
"""

import logging
import os
from typing import Any

from fastapi import APIRouter, Depends, Request

try:
    from src.auth.permissions import Session, require_auth
except ImportError:

    def require_auth():
        return None

    Session = None

logger = logging.getLogger(__name__)

router = APIRouter(tags=["transcripts"])

RAG_URL = os.getenv("RAG_URL", "http://rag-service:8001")


def _get_proxy_request():
    """Lazy import to avoid circular dependencies."""
    from src.main import proxy_request

    return proxy_request


# =============================================================================
# Analytics Signals & Segments
# =============================================================================


@router.get("/api/analytics/signals")
async def analytics_signals(
    start_date: str | None = None,
    end_date: str | None = None,
    speakers: str | None = None,
    limit: int = 50,
    session: Session = Depends(require_auth),
):
    """Get analytics signals (emotion trends, patterns, etc.)."""
    proxy_request = _get_proxy_request()

    params: dict[str, Any] = {"limit": limit}
    if start_date:
        params["start_date"] = start_date
    if end_date:
        params["end_date"] = end_date
    if speakers:
        params["speakers"] = speakers

    return await proxy_request(f"{RAG_URL}/analytics/signals", "GET", params=params)


@router.get("/api/analytics/segments")
async def analytics_segments(
    start_date: str | None = None,
    end_date: str | None = None,
    speakers: str | None = None,
    emotions: str | None = None,
    limit: int = 50,
    offset: int = 0,
    order: str = "desc",
    session: Session = Depends(require_auth),
):
    """Proxy segment drill-downs to the insights service."""
    proxy_request = _get_proxy_request()

    params: dict[str, Any] = {
        "limit": limit,
        "offset": offset,
        "order": order,
    }
    if start_date:
        params["start_date"] = start_date
    if end_date:
        params["end_date"] = end_date
    if speakers:
        params["speakers"] = speakers
    if emotions:
        params["emotions"] = emotions

    return await proxy_request(f"{RAG_URL}/analytics/segments", "GET", params=params)


# =============================================================================
# Transcript Query Endpoints
# =============================================================================


@router.get("/api/transcripts/speakers")
async def transcripts_speakers(session: Session = Depends(require_auth)):
    """Get list of all speakers."""
    proxy_request = _get_proxy_request()
    return await proxy_request(f"{RAG_URL}/transcripts/speakers", "GET")


@router.post("/api/transcripts/count")
async def transcripts_count(payload: dict[str, Any], http_request: Request, session: Session = Depends(require_auth)):
    """Count transcripts matching filters."""
    proxy_request = _get_proxy_request()
    return await proxy_request(f"{RAG_URL}/transcripts/count", "POST", json=payload)


@router.post("/api/transcripts/query")
async def transcripts_query(payload: dict[str, Any], http_request: Request, session: Session = Depends(require_auth)):
    """Query transcripts with filters and pagination."""
    proxy_request = _get_proxy_request()
    return await proxy_request(f"{RAG_URL}/transcripts/query", "POST", json=payload)


@router.get("/api/transcripts/recent")
async def transcripts_recent(limit: int = 10, session: Session = Depends(require_auth)):
    """Get recent transcripts."""
    proxy_request = _get_proxy_request()
    return await proxy_request(f"{RAG_URL}/transcripts/recent?limit={limit}", "GET")


@router.get("/api/transcript/{job_id}")
async def transcript_get(job_id: str, session: Session = Depends(require_auth)):
    """Get transcript by job_id."""
    proxy_request = _get_proxy_request()
    return await proxy_request(f"{RAG_URL}/transcripts/{job_id}", "GET")


@router.get("/api/result/{job_id}")
async def api_result_get(job_id: str, session: Session = Depends(require_auth)):
    """Alias for /api/transcript/{job_id} used by older frontend components."""
    return await transcript_get(job_id, session)


@router.get("/api/latest_result")
async def latest_result(session: Session = Depends(require_auth)):
    """Return the most recent transcript entry for quick UI previews."""
    proxy_request = _get_proxy_request()

    recent_resp = await proxy_request(f"{RAG_URL}/transcripts/recent?limit=1", "GET")

    transcripts = []
    if isinstance(recent_resp, dict):
        transcripts = recent_resp.get("transcripts") or recent_resp.get("items") or []
    elif isinstance(recent_resp, list):
        transcripts = recent_resp

    if not transcripts:
        return {"success": False, "message": "No transcripts available"}

    return {"success": True, "transcript": transcripts[0]}


logger.info("✅ Transcripts Router initialized with query endpoints")
