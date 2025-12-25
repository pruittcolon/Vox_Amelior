"""
RAG & Memory Router - Retrieval-Augmented Generation endpoints.

Provides RAG queries, semantic search, memory management, and personalization.
"""

import logging
import os
from typing import Any

from fastapi import APIRouter, Depends

try:
    from src.auth.permissions import Session, require_auth
except ImportError:

    def require_auth():
        return None

    Session = None

logger = logging.getLogger(__name__)

router = APIRouter(tags=["rag", "memory"])

RAG_URL = os.getenv("RAG_URL", "http://rag-service:8001")


def _get_proxy_request():
    """Lazy import to avoid circular dependencies."""
    from src.main import proxy_request

    return proxy_request


# =============================================================================
# RAG Query Endpoints
# =============================================================================


@router.post("/api/rag/query")
async def rag_query(request: dict[str, Any], session: Session = Depends(require_auth)):
    """Execute a RAG query against the knowledge base."""
    proxy_request = _get_proxy_request()
    return await proxy_request(f"{RAG_URL}/query", "POST", json=request)


@router.post("/api/search/semantic")
async def search_semantic(request: dict[str, Any], session: Session = Depends(require_auth)):
    """Proxy unified semantic search to RAG service."""
    proxy_request = _get_proxy_request()
    return await proxy_request(f"{RAG_URL}/search/semantic", "POST", json=request)


@router.post("/api/rag/memory/search")
async def rag_memory_search(request: dict[str, Any], session: Session = Depends(require_auth)):
    """Search through RAG memory."""
    proxy_request = _get_proxy_request()
    return await proxy_request(f"{RAG_URL}/memory/search", "POST", json=request)


# =============================================================================
# Memory Endpoints
# =============================================================================


@router.get("/api/memory/list")
async def memory_list(limit: int = 100, offset: int = 0, session: Session = Depends(require_auth)):
    """List all memories with pagination."""
    proxy_request = _get_proxy_request()
    return await proxy_request(f"{RAG_URL}/memory/list?limit={limit}&offset={offset}", "GET")


@router.post("/api/memory/search")
async def memory_search(request: dict[str, Any], session: Session = Depends(require_auth)):
    """Search memories by query."""
    proxy_request = _get_proxy_request()
    return await proxy_request(f"{RAG_URL}/memory/search", "POST", json=request)


@router.post("/api/memory/add")
async def memory_add(request: dict[str, Any], session: Session = Depends(require_auth)):
    """Add a new memory entry."""
    proxy_request = _get_proxy_request()
    return await proxy_request(f"{RAG_URL}/memory/add", "POST", json=request)


@router.post("/api/memory/create")
async def memory_create(request: dict[str, Any], session: Session = Depends(require_auth)):
    """Create a new memory entry (alias for add)."""
    proxy_request = _get_proxy_request()
    return await proxy_request(f"{RAG_URL}/memory/create", "POST", json=request)


@router.get("/api/memory/stats")
async def memory_stats(session: Session = Depends(require_auth)):
    """Get memory statistics."""
    proxy_request = _get_proxy_request()
    return await proxy_request(f"{RAG_URL}/memory/stats", "GET")


@router.get("/api/memory/emotions/stats")
async def memory_emotions_stats(session: Session = Depends(require_auth)):
    """Get emotion statistics from memory."""
    proxy_request = _get_proxy_request()
    return await proxy_request(f"{RAG_URL}/memory/emotions/stats", "GET")


# =============================================================================
# Personalization & Debug Endpoints
# =============================================================================


@router.post("/api/rag/personalize")
async def rag_personalize(session: Session = Depends(require_auth)):
    """Trigger RAG personalization pipeline."""
    proxy_request = _get_proxy_request()
    return await proxy_request(f"{RAG_URL}/personalize", "POST", json={})


@router.get("/api/debug/transcripts/time-range")
async def debug_transcript_time_range(session: Session = Depends(require_auth)):
    """Expose RAG dataset coverage stats to the UI."""
    proxy_request = _get_proxy_request()
    return await proxy_request(f"{RAG_URL}/debug/transcripts/time-range", "GET")


logger.info("✅ RAG Router initialized with memory and search endpoints")
