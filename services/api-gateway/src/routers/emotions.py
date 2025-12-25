"""
Emotions Router - Authenticated proxy to Emotion Service.

Handles emotion analytics, stats, and moment feeds.
Refactored from main.py
"""

import logging
import os
from typing import Any, Optional

from fastapi import APIRouter, Depends, Query, Request
from src.auth.permissions import Session, require_auth

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/emotions", tags=["emotions"])

RAG_URL = os.getenv("RAG_URL", "http://rag-service:8001")


def _get_proxy_request():
    """Lazy import to avoid circular dependencies."""
    from src.main import proxy_request
    return proxy_request


# =============================================================================
# Emotion Endpoints (Data from RAG Service)
# =============================================================================

@router.get("/stats")
async def emotion_stats(
    period: str = "today",
    session: Session = Depends(require_auth)
):
    """Get emotion statistics for a period."""
    proxy_request = _get_proxy_request()
    return await proxy_request(
        f"{RAG_URL}/emotions/stats",
        "GET",
        params={"period": period, "user_id": getattr(session, "user_id", None)}
    )


@router.get("/analytics")
async def emotion_analytics(
    period: str = "today",
    session: Session = Depends(require_auth)
):
    """Get detailed emotion analytics (charts, timeline)."""
    proxy_request = _get_proxy_request()
    return await proxy_request(
        f"{RAG_URL}/emotions/analytics",
        "GET",
        params={"period": period, "user_id": getattr(session, "user_id", None)}
    )


@router.get("/moments")
async def emotion_moments(
    limit: int = 20,
    offset: int = 0,
    emotion: Optional[str] = None,
    speaker: Optional[str] = None,
    session: Session = Depends(require_auth)
):
    """Get feed of emotional moments."""
    proxy_request = _get_proxy_request()
    
    params = {
        "limit": limit,
        "offset": offset,
        "user_id": getattr(session, "user_id", None)
    }
    
    if emotion and emotion != 'all':
        params['emotion'] = emotion
    if speaker:
        params['speaker'] = speaker
        
    return await proxy_request(f"{RAG_URL}/emotions/moments", "GET", params=params)


logger.info("✅ Emotions Router initialized")
