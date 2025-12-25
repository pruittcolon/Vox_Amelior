"""
Analysis Artifacts Router - Analysis archive and retrieval endpoints.

Provides analysis archiving, listing, searching, and meta-analysis via streaming.
"""

import json
import logging
import os
from typing import Any

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import StreamingResponse

try:
    from src.auth.permissions import Session, require_auth
except ImportError:

    def require_auth():
        return None

    Session = None

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/analysis", tags=["analysis"])

RAG_URL = os.getenv("RAG_URL", "http://rag-service:8001")


def _get_proxy_request():
    """Lazy import to avoid circular dependencies."""
    from src.main import proxy_request

    return proxy_request


# =============================================================================
# Analysis Archive Endpoints
# =============================================================================


@router.post("/archive")
async def analysis_archive(payload: dict[str, Any], http_request: Request, session: Session = Depends(require_auth)):
    """Archive an analysis result."""
    proxy_request = _get_proxy_request()

    # Add user context
    payload.setdefault("user_id", getattr(session, "user_id", None))

    return await proxy_request(f"{RAG_URL}/analysis/archive", "POST", json=payload)


@router.get("/list")
async def analysis_list(
    limit: int = 50,
    offset: int = 0,
    scope: str = Query("user", description="user or global"),
    session: Session = Depends(require_auth),
):
    """List analysis artifacts."""
    proxy_request = _get_proxy_request()

    params = {
        "limit": limit,
        "offset": offset,
        "scope": scope,
        "user_id": getattr(session, "user_id", None),
    }

    return await proxy_request(f"{RAG_URL}/analysis/list", "GET", params=params)


@router.get("/{artifact_id}")
async def analysis_get(
    artifact_id: str, scope: str = Query("user", description="user or global"), session: Session = Depends(require_auth)
):
    """Get a specific analysis artifact."""
    proxy_request = _get_proxy_request()

    params = {
        "scope": scope,
        "user_id": getattr(session, "user_id", None),
    }

    return await proxy_request(f"{RAG_URL}/analysis/{artifact_id}", "GET", params=params)


@router.post("/search")
async def analysis_search(payload: dict[str, Any], session: Session = Depends(require_auth)):
    """Search analysis artifacts."""
    proxy_request = _get_proxy_request()

    payload.setdefault("user_id", getattr(session, "user_id", None))

    return await proxy_request(f"{RAG_URL}/analysis/search", "POST", json=payload)


# =============================================================================
# Meta-Analysis (Streaming)
# =============================================================================


@router.post("/meta")
async def analysis_meta(
    payload: dict[str, Any],
    http_request: Request,
    session: Session = Depends(require_auth),
):
    """Run meta-analysis on stored artifacts via streaming."""
    proxy_request = _get_proxy_request()

    analysis_id = http_request.headers.get("X-Analysis-Id")
    headers = {"X-Analysis-Id": analysis_id} if analysis_id else None

    async def streamer():
        try:
            # In production, this would be a streaming call
            result = await proxy_request(
                f"{RAG_URL}/analysis/meta",
                "POST",
                json=payload,
                extra_headers=headers,
            )
            yield f"event: result\ndata: {json.dumps(result)}\n\n"
            yield f"event: done\ndata: {json.dumps({'status': 'complete'})}\n\n"
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'detail': str(e)})}\n\n"

    return StreamingResponse(streamer(), media_type="text/event-stream")


logger.info("✅ Analysis Router initialized with artifact endpoints")
