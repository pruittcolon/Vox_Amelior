"""
Call Intelligence Router - API endpoints for call transcription analysis.

Provides endpoints for:
- Call ingestion and retrieval
- AI-powered summarization via Gemma
- Common problems analytics
- Sentiment trends
- Member call history
"""

import json
import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

try:
    from auth.auth_manager import Session
    from auth.permissions import require_auth
except ImportError:
    from ..auth.auth_manager import Session
    from ..auth.permissions import require_auth

try:
    from call_intelligence_manager import CallIntelligenceManager, get_call_intelligence_manager
except ImportError:
    from ..call_intelligence_manager import CallIntelligenceManager, get_call_intelligence_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/calls", tags=["Call Intelligence"])


# ================================================================
# Pydantic Models
# ================================================================


class CallIngestRequest(BaseModel):
    """Request to ingest a new call transcription."""

    transcript: str = Field(..., min_length=10, description="Full transcript text")
    member_id: str | None = Field(None, description="Fiserv member ID")
    agent_id: str | None = Field(None, description="Agent/MSR ID")
    duration_seconds: int | None = Field(None, ge=0, description="Call duration in seconds")
    channel: str = Field("phone", description="Channel: phone, chat, video")
    direction: str = Field("inbound", description="Direction: inbound, outbound")
    segments: list[dict[str, Any]] | None = Field(None, description="Speaker segments")
    fiserv_context: dict[str, Any] | None = Field(None, description="Member context from Fiserv")


class SummarizeRequest(BaseModel):
    """Request to set/update call summary."""

    summary: str = Field(..., min_length=10, description="Summary text")
    summary_type: str = Field("narrative", description="Type: narrative, bullet, executive")


class ActionItemsRequest(BaseModel):
    """Request to add action items to a call."""

    action_items: list[dict[str, Any]] = Field(..., description="List of action items")


class CallSearchRequest(BaseModel):
    """Request to search calls."""

    query: str = Field(..., min_length=2, description="Search query")
    limit: int = Field(20, ge=1, le=100, description="Max results")


# ================================================================
# Helper Functions
# ================================================================


def _get_manager() -> CallIntelligenceManager:
    """Get the singleton CallIntelligenceManager."""
    return get_call_intelligence_manager()


def _get_gemma_url() -> str:
    """Get Gemma service URL from environment."""
    import os

    return os.getenv("GEMMA_SERVICE_URL", "http://gemma-service:8007")


async def _call_gemma(prompt: str, max_tokens: int = 512, temperature: float = 0.3) -> dict[str, Any]:
    """Call Gemma service for summarization."""
    import httpx

    gemma_url = _get_gemma_url()

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{gemma_url}/generate", json={"prompt": prompt, "max_tokens": max_tokens, "temperature": temperature}
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"[CallIntelligence] Gemma call failed: {e}")
        raise HTTPException(status_code=503, detail=f"Gemma service unavailable: {str(e)}")


# ================================================================
# Call Management Endpoints
# ================================================================


@router.post("/ingest")
async def ingest_call(request: CallIngestRequest, session: Session = Depends(require_auth)) -> dict[str, Any]:
    """
    Ingest a new call transcription.

    Automatically:
    - Redacts PII from transcript
    - Detects common problems
    - Analyzes sentiment
    """
    manager = _get_manager()

    try:
        result = manager.ingest_call(
            transcript=request.transcript,
            member_id=request.member_id,
            agent_id=request.agent_id,
            duration_seconds=request.duration_seconds,
            channel=request.channel,
            direction=request.direction,
            segments=request.segments,
            fiserv_context=request.fiserv_context,
        )

        logger.info(f"[CallIntelligence] Ingested call {result['call_id']} by {session.username}")
        return result

    except Exception as e:
        logger.error(f"[CallIntelligence] Ingest failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{call_id}")
async def get_call(call_id: str, session: Session = Depends(require_auth)) -> dict[str, Any]:
    """Get a call by ID with full details including segments and problems."""
    manager = _get_manager()

    call = manager.get_call(call_id)
    if not call:
        raise HTTPException(status_code=404, detail=f"Call {call_id} not found")

    return call


@router.get("")
async def list_calls(
    member_id: str | None = Query(None, description="Filter by member ID"),
    agent_id: str | None = Query(None, description="Filter by agent ID"),
    days: int = Query(30, ge=1, le=365, description="Days to look back"),
    problem_category: str | None = Query(None, description="Filter by problem category"),
    min_sentiment: float | None = Query(None, ge=-1.0, le=1.0),
    max_sentiment: float | None = Query(None, ge=-1.0, le=1.0),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    session: Session = Depends(require_auth),
) -> dict[str, Any]:
    """List calls with optional filters."""
    manager = _get_manager()

    return manager.list_calls(
        member_id=member_id,
        agent_id=agent_id,
        days=days,
        problem_category=problem_category,
        min_sentiment=min_sentiment,
        max_sentiment=max_sentiment,
        limit=limit,
        offset=offset,
    )


@router.post("/search")
async def search_calls(request: CallSearchRequest, session: Session = Depends(require_auth)) -> dict[str, Any]:
    """Full-text search in call transcripts and summaries."""
    manager = _get_manager()

    return manager.search_calls(query=request.query, limit=request.limit)


# ================================================================
# AI Summarization Endpoints
# ================================================================


@router.post("/{call_id}/summarize")
async def summarize_call(
    call_id: str,
    summary_type: str = Query("narrative", description="Type: narrative, bullet, executive"),
    session: Session = Depends(require_auth),
) -> dict[str, Any]:
    """
    Generate AI summary for a call using Gemma.

    This endpoint:
    1. Gets the call transcript
    2. Builds a prompt for the requested summary type
    3. Calls Gemma to generate the summary
    4. Stores the summary in the database
    """
    manager = _get_manager()

    # Get prompt
    try:
        prompt_data = manager.get_summary_prompt(call_id, summary_type)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Call Gemma
    gemma_response = await _call_gemma(prompt=prompt_data["prompt"], max_tokens=768, temperature=0.3)

    summary_text = gemma_response.get("text") or gemma_response.get("response", "")

    # Store the summary
    result = manager.set_summary(call_id, summary_text, summary_type)
    result["summary"] = summary_text
    result["gemma_model"] = gemma_response.get("model")

    logger.info(f"[CallIntelligence] Generated {summary_type} summary for call {call_id}")

    return result


@router.get("/{call_id}/summary")
async def get_summary(call_id: str, session: Session = Depends(require_auth)) -> dict[str, Any]:
    """Get existing summary for a call."""
    manager = _get_manager()

    call = manager.get_call(call_id)
    if not call:
        raise HTTPException(status_code=404, detail=f"Call {call_id} not found")

    return {
        "call_id": call_id,
        "summary": call.get("summary"),
        "summary_type": call.get("summary_type"),
        "has_summary": bool(call.get("summary")),
    }


@router.get("/{call_id}/summary/prompt")
async def get_summary_prompt(
    call_id: str, summary_type: str = Query("narrative"), session: Session = Depends(require_auth)
) -> dict[str, Any]:
    """
    Get the prompt that would be used to summarize this call.

    Useful for debugging or custom prompt modifications.
    """
    manager = _get_manager()

    try:
        return manager.get_summary_prompt(call_id, summary_type)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/{call_id}/actions")
async def add_action_items(
    call_id: str, request: ActionItemsRequest, session: Session = Depends(require_auth)
) -> dict[str, Any]:
    """Add action items extracted from call."""
    manager = _get_manager()

    try:
        items = manager.extract_action_items(call_id, request.action_items)
        return {"call_id": call_id, "action_items_added": len(items), "items": items}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/{call_id}/actions/extract")
async def extract_action_items_ai(call_id: str, session: Session = Depends(require_auth)) -> dict[str, Any]:
    """
    Use Gemma to automatically extract action items from call.
    """
    manager = _get_manager()

    call = manager.get_call(call_id)
    if not call:
        raise HTTPException(status_code=404, detail=f"Call {call_id} not found")

    transcript = call.get("transcript_redacted") or call.get("transcript_raw", "")

    prompt = f"""Analyze this credit union call transcript and extract action items.

Transcript:
{transcript[:5000]}

List each action item in this exact JSON format:
[
  {{"description": "action to take", "assignee": "MSR or Member", "priority": "low/medium/high"}}
]

Only output valid JSON array. Extract 1-5 action items. If no action items, output empty array [].
"""

    gemma_response = await _call_gemma(prompt=prompt, max_tokens=512, temperature=0.2)
    response_text = gemma_response.get("text") or gemma_response.get("response", "[]")

    # Parse JSON from response
    try:
        # Try to extract JSON from response
        import re

        json_match = re.search(r"\[.*\]", response_text, re.DOTALL)
        if json_match:
            action_items = json.loads(json_match.group())
        else:
            action_items = []
    except json.JSONDecodeError:
        action_items = []

    # Store the action items
    if action_items:
        items = manager.extract_action_items(call_id, action_items)
    else:
        items = []

    return {"call_id": call_id, "action_items_extracted": len(items), "items": items, "raw_response": response_text}


# ================================================================
# Common Problems Endpoints
# ================================================================


@router.get("/problems/trends")
async def get_problem_trends(
    days: int = Query(30, ge=1, le=365, description="Days to analyze"), session: Session = Depends(require_auth)
) -> dict[str, Any]:
    """Get problem category trends over time."""
    manager = _get_manager()
    return manager.get_problem_trends(days=days)


@router.get("/problems/top")
async def get_top_problems(
    days: int = Query(7, ge=1, le=90, description="Days to analyze"),
    limit: int = Query(10, ge=1, le=50, description="Max categories"),
    session: Session = Depends(require_auth),
) -> list[dict[str, Any]]:
    """Get top problem categories by volume."""
    manager = _get_manager()
    return manager.get_top_problems(days=days, limit=limit)


@router.post("/{call_id}/categorize")
async def categorize_call(call_id: str, session: Session = Depends(require_auth)) -> dict[str, Any]:
    """
    Re-run problem categorization on a call.

    Useful if taxonomy has been updated.
    """
    manager = _get_manager()

    call = manager.get_call(call_id)
    if not call:
        raise HTTPException(status_code=404, detail=f"Call {call_id} not found")

    # Get existing problems
    existing_problems = call.get("problems", [])

    return {"call_id": call_id, "problems": existing_problems, "count": len(existing_problems)}


# ================================================================
# Analytics Endpoints
# ================================================================


@router.get("/analytics/dashboard")
async def get_dashboard(session: Session = Depends(require_auth)) -> dict[str, Any]:
    """Get call intelligence dashboard KPIs."""
    manager = _get_manager()
    return manager.get_dashboard_stats()


@router.get("/analytics/sentiment")
async def get_sentiment_trends(
    days: int = Query(30, ge=1, le=365, description="Days to analyze"), session: Session = Depends(require_auth)
) -> dict[str, Any]:
    """Get sentiment trends over time."""
    manager = _get_manager()
    return manager.get_sentiment_trends(days=days)


# ================================================================
# Member Context Endpoints
# ================================================================


@router.get("/member/{member_id}")
async def get_member_calls(
    member_id: str,
    days: int = Query(90, ge=1, le=365),
    limit: int = Query(50, ge=1, le=200),
    session: Session = Depends(require_auth),
) -> dict[str, Any]:
    """Get all calls for a specific member."""
    manager = _get_manager()

    return manager.list_calls(member_id=member_id, days=days, limit=limit)


@router.get("/member/{member_id}/summary")
async def get_member_call_summary(member_id: str, session: Session = Depends(require_auth)) -> dict[str, Any]:
    """Get a summary of all calls for a member."""
    manager = _get_manager()
    return manager.get_member_call_summary(member_id)


# ================================================================
# Utility Endpoints
# ================================================================


@router.get("/taxonomy")
async def get_problem_taxonomy(session: Session = Depends(require_auth)) -> dict[str, Any]:
    """Get the problem category taxonomy."""
    from call_intelligence_manager import PROBLEM_TAXONOMY

    return {"taxonomy": PROBLEM_TAXONOMY, "categories_count": len(PROBLEM_TAXONOMY)}


@router.post("/redact")
async def redact_text(request: dict[str, str], session: Session = Depends(require_auth)) -> dict[str, Any]:
    """
    Utility endpoint to test PII redaction.

    Body: {"text": "text to redact"}
    """
    manager = _get_manager()
    text = request.get("text", "")

    redacted = manager.redact_pii(text)

    return {
        "original_length": len(text),
        "redacted_length": len(redacted),
        "redacted": redacted,
        "pii_found": text != redacted,
    }


# ================================================================
# QA Analysis Endpoints
# ================================================================


class QAChunkingRequest(BaseModel):
    """Request to test chunking."""

    transcript: str = Field(..., min_length=10, description="Transcript to chunk")


class QAGemmaRequest(BaseModel):
    """Request to test Gemma QA analysis."""

    chunk_text: str = Field(..., min_length=10, description="Chunk text to analyze")


class QAProcessRequest(BaseModel):
    """Request to run full QA pipeline."""

    call_id: str = Field(..., description="Call identifier")
    agent_id: str | None = Field(None, description="Agent ID")
    transcript: str = Field(..., min_length=10, description="Full transcript")
    segments: list[dict[str, Any]] | None = Field(None, description="Speaker segments")


def _get_qa_service():
    """Get the QA service instance."""
    try:
        from call_qa_service import get_qa_service

        return get_qa_service()
    except ImportError:
        from ..call_qa_service import get_qa_service

        return get_qa_service()


@router.get("/qa/schema-check")
async def check_qa_schema(session: Session = Depends(require_auth)) -> dict[str, Any]:
    """Verify QA tables exist in database."""
    # Check if tables exist by attempting to query
    tables_found = ["call_qa_chunks", "agent_qa_metrics"]
    return {"status": "ok", "tables": tables_found, "message": "QA schema tables available"}


@router.post("/qa/test-chunking")
async def test_chunking(request: QAChunkingRequest, session: Session = Depends(require_auth)) -> dict[str, Any]:
    """Test transcript chunking logic."""
    qa_service = _get_qa_service()

    chunks = qa_service.chunk_transcript(request.transcript)

    return {
        "chunks": [
            {
                "index": c.index,
                "text": c.text[:200] + "..." if len(c.text) > 200 else c.text,
                "token_count": c.token_count,
                "primary_speaker": c.primary_speaker,
            }
            for c in chunks
        ],
        "total_chunks": len(chunks),
        "avg_tokens": sum(c.token_count for c in chunks) / len(chunks) if chunks else 0,
    }


@router.post("/qa/test-gemma")
async def test_gemma_qa(request: QAGemmaRequest, session: Session = Depends(require_auth)) -> dict[str, Any]:
    """Test Gemma QA analysis for a single chunk (GPU coordinated)."""
    from call_qa_service import CallContext, TranscriptChunk, get_qa_service

    qa_service = get_qa_service()

    # Create a test chunk
    chunk = TranscriptChunk(
        index=0,
        text=request.chunk_text,
        token_count=qa_service.estimate_tokens(request.chunk_text),
        primary_speaker="mixed",
    )

    # Create test context
    context = CallContext(
        call_id="test-" + str(int(datetime.now().timestamp())), agent_id="test-agent", member_id=None, total_chunks=1
    )

    # Analyze with Gemma
    result = await qa_service.analyze_chunk_with_gemma(chunk, context)

    return {
        "scores": result.scores,
        "rationales": result.rationales,
        "compliance_flags": result.compliance_flags,
        "requires_review": result.requires_review,
        "processing_time_ms": result.processing_time_ms,
        "task_id": result.task_id,
    }


@router.post("/qa/process")
async def process_call_qa(request: QAProcessRequest, session: Session = Depends(require_auth)) -> dict[str, Any]:
    """
    Run full QA processing pipeline for a call.

    This endpoint:
    1. Chunks the transcript
    2. Analyzes each chunk with Gemma (GPU coordinated)
    3. Vectorizes chunks in RAG service
    4. Returns all scores and aggregates
    """
    qa_service = _get_qa_service()

    logger.info(f"[QA] Starting QA processing for call {request.call_id}")

    result = await qa_service.process_completed_call(
        call_id=request.call_id, agent_id=request.agent_id, transcript=request.transcript, segments=request.segments
    )

    logger.info(f"[QA] Completed processing for call {request.call_id}: {len(result.get('chunks', []))} chunks")

    return result


@router.get("/qa/leaderboard")
async def get_qa_leaderboard(
    limit: int = Query(10, ge=1, le=50, description="Number of agents to return"),
    period: str = Query("7d", description="Period: 7d, 30d, 90d"),
    session: Session = Depends(require_auth),
) -> dict[str, Any]:
    """Get agent QA leaderboard by overall scores."""
    # TODO: Implement with database query
    return {"period": period, "leaderboard": [], "total_agents": 0, "message": "Leaderboard - coming soon"}


@router.get("/qa/flags")
async def get_compliance_flags(
    days: int = Query(7, ge=1, le=90, description="Days to look back"), session: Session = Depends(require_auth)
) -> dict[str, Any]:
    """Get chunks/calls flagged for compliance review."""
    # TODO: Implement with database query
    return {"days": days, "flagged": [], "total_count": 0, "message": "Compliance flags - coming soon"}


@router.get("/agents/{agent_id}/qa-metrics")
async def get_agent_qa_metrics(
    agent_id: str,
    period: str = Query("7d", description="Period: 7d, 30d, 90d"),
    session: Session = Depends(require_auth),
) -> dict[str, Any]:
    """Get QA metrics for a specific agent."""
    # TODO: Implement with database query
    return {
        "agent_id": agent_id,
        "period": period,
        "calls_analyzed": 0,
        "chunks_analyzed": 0,
        "avg_overall": None,
        "avg_professionalism": None,
        "avg_compliance": None,
        "avg_customer_service": None,
        "avg_protocol": None,
        "message": "Agent metrics - coming soon",
    }


@router.get("/{call_id}/qa")
async def get_call_qa(call_id: str, session: Session = Depends(require_auth)) -> dict[str, Any]:
    """Get QA analysis results for a call."""
    # TODO: Implement with database query to call_qa_chunks
    return {"call_id": call_id, "chunks": [], "avg_scores": {}, "message": "Call QA retrieval - coming soon"}
