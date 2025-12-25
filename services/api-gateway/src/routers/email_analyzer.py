"""
Email Analyzer Router - Email analysis and search endpoints.

Provides email querying, statistics, and Gemma-powered analysis with streaming.
"""

import base64
import json
import logging
import os
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

try:
    from src.auth.permissions import Session, require_auth
except ImportError:

    def require_auth():
        return None

    Session = None

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/email", tags=["email"])

GEMMA_URL = os.getenv("GEMMA_URL", "http://gemma-service:8004")
RAG_URL = os.getenv("RAG_URL", "http://rag-service:8001")

# Feature flag
EMAIL_ANALYZER_ENABLED = os.getenv("EMAIL_ANALYZER_ENABLED", "false").lower() in {"1", "true", "yes"}


def _get_proxy_request():
    """Lazy import to avoid circular dependencies."""
    from src.main import proxy_request

    return proxy_request


def _ensure_email_analyzer_enabled():
    """Raise 503 if email analyzer is disabled."""
    if not EMAIL_ANALYZER_ENABLED:
        raise HTTPException(status_code=503, detail="Email analyzer is not enabled")


def _prepare_email_stream_payload(raw: str) -> dict[str, Any]:
    """Decode base64-encoded JSON payload for streaming endpoints."""
    try:
        decoded = base64.b64decode(raw).decode("utf-8")
        return json.loads(decoded)
    except Exception:
        return {}


# =============================================================================
# Email Query Endpoints
# =============================================================================


@router.get("/users")
async def email_users(session: Session = Depends(require_auth)):
    """Get list of email users."""
    _ensure_email_analyzer_enabled()
    proxy_request = _get_proxy_request()
    return await proxy_request(f"{RAG_URL}/email/users", "GET")


@router.get("/labels")
async def email_labels(session: Session = Depends(require_auth)):
    """Get list of email labels/folders."""
    _ensure_email_analyzer_enabled()
    proxy_request = _get_proxy_request()
    return await proxy_request(f"{RAG_URL}/email/labels", "GET")


@router.get("/stats")
async def email_stats(
    start_date: str | None = None,
    end_date: str | None = None,
    user: str | None = None,
    label: str | None = None,
    session: Session = Depends(require_auth),
):
    """Get email statistics with optional filters."""
    _ensure_email_analyzer_enabled()
    proxy_request = _get_proxy_request()

    params: dict[str, Any] = {}
    if start_date:
        params["start_date"] = start_date
    if end_date:
        params["end_date"] = end_date
    if user:
        params["user"] = user
    if label:
        params["label"] = label

    return await proxy_request(f"{RAG_URL}/email/stats", "GET", params=params)


@router.post("/query")
async def email_query(payload: dict[str, Any], session: Session = Depends(require_auth)):
    """Query emails with filters."""
    _ensure_email_analyzer_enabled()
    proxy_request = _get_proxy_request()
    return await proxy_request(f"{RAG_URL}/email/query", "POST", json=payload)


# =============================================================================
# Email Analysis Endpoints
# =============================================================================


@router.post("/analyze/quick")
async def email_analyze_quick(payload: dict[str, Any], session: Session = Depends(require_auth)):
    """Quick email analysis."""
    _ensure_email_analyzer_enabled()
    proxy_request = _get_proxy_request()
    return await proxy_request(f"{RAG_URL}/email/analyze/quick", "POST", json=payload)


@router.post("/analyze/gemma/quick")
async def email_analyze_gemma_quick(payload: dict[str, Any], session: Session = Depends(require_auth)):
    """Quick email analysis backed by Gemma.

    Body: { question: str(>=3), filters: object, max_emails?: int }
    Flow: query RAG for top emails, construct prompt, call Gemma /generate, return summary
    """
    _ensure_email_analyzer_enabled()
    proxy_request = _get_proxy_request()

    question = (payload.get("question") or "").strip()
    if len(question) < 3:
        raise HTTPException(status_code=400, detail="question must be at least 3 characters")

    filters = payload.get("filters") or {}
    max_emails = int(payload.get("max_emails") or 10)
    max_emails = max(1, min(max_emails, 25))

    logger.info(
        "[EMAIL][GEMMA] quick requested by %s qlen=%s max_emails=%s",
        getattr(session, "user_id", None),
        len(question),
        max_emails,
    )

    # Query RAG for matching emails
    rag_query = {
        "query": question,
        "limit": max_emails,
        "filters": filters,
    }

    rag_resp = await proxy_request(f"{RAG_URL}/email/query", "POST", json=rag_query)
    items = rag_resp.get("items") or rag_resp.get("emails") or []

    # Build prompt with clipped email snippets
    snippets = []
    for item in items[:max_emails]:
        subject = item.get("subject") or "No Subject"
        sender = item.get("from") or item.get("sender") or "Unknown"
        body = (item.get("body") or item.get("snippet") or "")[:500]
        snippets.append(f"From: {sender}\nSubject: {subject}\n{body}\n")

    prompt = "\n".join(
        [
            "You are a helpful assistant analyzing email threads.",
            "Provide a concise, actionable answer to the user's question based on the emails.",
            "",
            f"User Question: {question}",
            "\nRelevant emails (most recent first):",
            "\n---\n".join(snippets) if snippets else "(No matching emails found)",
            "\nProvide your analysis:",
        ]
    )

    gemma_resp = await proxy_request(
        f"{GEMMA_URL}/generate",
        "POST",
        json={
            "prompt": prompt,
            "max_tokens": 384,
            "temperature": 0.4,
        },
    )

    return {
        "success": True,
        "answer": gemma_resp.get("text") or gemma_resp.get("response") or "",
        "emails_used": len(items),
        "model": gemma_resp.get("model"),
    }


@router.get("/analyze/stream")
async def email_analyze_stream(
    payload: str = Query(..., description="Base64 encoded JSON payload"), session: Session = Depends(require_auth)
):
    """Stream email analysis via SSE."""
    _ensure_email_analyzer_enabled()
    proxy_request = _get_proxy_request()
    decoded_payload = _prepare_email_stream_payload(payload)

    async def event_generator():
        try:
            async with (
                httpx.AsyncClient(timeout=120.0) as client,
                client.stream("GET", f"{RAG_URL}/email/analyze/stream", params={"payload": payload}) as resp,
            ):
                async for line in resp.aiter_lines():
                    yield line + "\n"
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'detail': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/analyze/gemma/stream")
async def email_analyze_gemma_stream(
    payload: str = Query(..., description="Base64 encoded JSON payload"), session: Session = Depends(require_auth)
):
    """Stream Gemma-powered email analysis via SSE."""
    _ensure_email_analyzer_enabled()
    proxy_request = _get_proxy_request()
    decoded_payload = _prepare_email_stream_payload(payload)

    # This would be more complex in production - simplified for now
    return await email_analyze_stream(payload, session)


@router.post("/analyze/cancel")
async def email_analyze_cancel(payload: dict[str, Any], session: Session = Depends(require_auth)):
    """Cancel an ongoing email analysis."""
    _ensure_email_analyzer_enabled()
    proxy_request = _get_proxy_request()
    return await proxy_request(f"{RAG_URL}/email/analyze/cancel", "POST", json=payload)


logger.info("✅ Email Router initialized with analysis endpoints")
