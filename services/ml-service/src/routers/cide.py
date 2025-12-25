"""
CIDE Router - Contextual Insight Discovery Engine Endpoints
============================================================

API endpoints for the enhanced vectorization and insight discovery features.

Endpoints:
- POST /enriched: Enhanced vectorization with temporal context
- POST /discover: Run chaotic insight discovery
- GET /temporal-events: Fetch world events for a date
- POST /generate-context: Generate Gemma context for a section

Author: NeMo Server Team
Version: 1.0.0
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/cide", tags=["CIDE - Insight Discovery"])

# =============================================================================
# Request/Response Models
# =============================================================================


class TemporalContextRequest(BaseModel):
    """Request for temporal context enrichment."""
    
    recording_date: str = Field(..., description="ISO format date string")
    meeting_type: str = Field(default="general", description="Type of meeting")
    business_events: list[str] = Field(default_factory=list)
    fetch_world_events: bool = Field(default=True)


class TemporalContextResponse(BaseModel):
    """Response with temporal context data."""
    
    recording_date: str
    day_of_week: str
    quarter: int
    fiscal_quarter: str
    notable_events: list[str]
    event_categories: list[str]
    context_summary: str


class SectionInput(BaseModel):
    """Input for a transcription section."""
    
    text: str
    index: int = 0
    speaker: str = "unknown"
    start_time_sec: float | None = None
    end_time_sec: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EnrichedVectorizeRequest(BaseModel):
    """Request for enhanced vectorization."""
    
    sections: list[SectionInput] = Field(..., description="Sections to vectorize")
    recording_date: str = Field(..., description="ISO format recording date")
    meeting_type: str = Field(default="general")
    business_context: dict[str, Any] = Field(default_factory=dict)
    fetch_world_events: bool = Field(default=True)
    combination_mode: str = Field(
        default="concatenate",
        description="Embedding combination mode: concatenate, weighted_average, or hybrid"
    )
    generate_gemma_context: bool = Field(
        default=True,
        description="Whether to generate Gemma context descriptions"
    )


class EnrichedVectorizeResponse(BaseModel):
    """Response from enhanced vectorization."""
    
    job_id: str
    sections_processed: int
    embedding_dimension: int
    temporal_context: dict[str, Any]
    sections: list[dict[str, Any]]


class DiscoveryRequest(BaseModel):
    """Request for insight discovery."""
    
    sections: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Sections with embeddings and metadata"
    )
    discovery_modes: list[str] = Field(
        default=["temporal_bridging", "semantic_outliers", "contradiction"],
        description="Discovery modes to run"
    )
    max_insights: int = Field(default=10, ge=1, le=50)
    min_chaos_score: float = Field(default=0.3, ge=0.0, le=1.0)
    min_temporal_gap_days: int = Field(default=7, ge=0)


class DiscoveryResponse(BaseModel):
    """Response from insight discovery."""
    
    transcription_count: int
    section_count: int
    pairs_analyzed: int
    insights: list[dict[str, Any]]
    processing_time_sec: float
    discovery_modes: list[str]


class GemmaContextRequest(BaseModel):
    """Request for Gemma context generation."""
    
    section_text: str
    temporal_context: dict[str, Any] = Field(default_factory=dict)
    business_context: dict[str, Any] = Field(default_factory=dict)
    max_tokens: int = Field(default=300, ge=50, le=500)


class GemmaContextResponse(BaseModel):
    """Response with Gemma-generated context."""
    
    context: str
    token_count: int
    processing_time_ms: int


# =============================================================================
# V2 Scoring Models (Quantitative Business Analysis)
# =============================================================================


class V2ScoreSegmentRequest(BaseModel):
    """Request for V2 quantitative segment scoring."""
    
    segment_text: str = Field(..., description="Transcript segment to score")
    speaker: str = Field(default="Unknown")
    meeting_type: str = Field(default="general")
    company_context: str = Field(default="", description="Context about the company")
    recording_date: str = Field(default="", description="ISO format date")
    transcription_id: str = Field(default="", description="Parent transcription ID")


class V2ScoreSegmentResponse(BaseModel):
    """Response with quantitative scores."""
    
    segment_id: str
    scores: dict[str, int]
    extracted: dict[str, list[str]]
    overall_tone: str
    health_score: float
    model_confidence: float
    justifications: dict[str, str]


class V2ScoreBatchRequest(BaseModel):
    """Request for batch scoring multiple segments."""
    
    segments: list[dict[str, Any]] = Field(..., description="Segments to score")
    company_context: str = Field(default="")
    store_results: bool = Field(default=True, description="Store scores in database")


class V2ScoreBatchResponse(BaseModel):
    """Response from batch scoring."""
    
    segments_scored: int
    segments_failed: int
    avg_health_score: float
    high_stress_count: int
    results: list[dict[str, Any]]


class V2CorrelationRequest(BaseModel):
    """Request for correlation data retrieval."""
    
    start_date: str = Field(default="", description="ISO format start date")
    end_date: str = Field(default="", description="ISO format end date")
    stress_threshold: int = Field(default=7, ge=1, le=10)


class V2CorrelationResponse(BaseModel):
    """Response with correlation-ready data."""
    
    date_range: dict[str, str]
    aggregated_data: list[dict[str, Any]]
    high_stress_segments: list[dict[str, Any]]
    summary_stats: dict[str, Any]


# =============================================================================
# Endpoint Implementations
# =============================================================================


@router.post("/temporal-context", response_model=TemporalContextResponse)
async def get_temporal_context(request: TemporalContextRequest):
    """
    Get enriched temporal context for a recording date.
    
    Fetches world events and generates temporal features.
    """
    try:
        # Import here to avoid circular imports
        from src.temporal_context import get_temporal_enricher
        
        enricher = get_temporal_enricher()
        
        # Parse date
        try:
            recording_date = datetime.fromisoformat(request.recording_date)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid date format: {request.recording_date}. Use ISO format."
            )
        
        # Enrich context
        context = await enricher.enrich(
            recording_date=recording_date,
            meeting_type=request.meeting_type,
            business_events=request.business_events,
            fetch_world_events=request.fetch_world_events,
        )
        
        return TemporalContextResponse(
            recording_date=context.recording_date.isoformat(),
            day_of_week=context.day_of_week,
            quarter=context.quarter,
            fiscal_quarter=context.fiscal_quarter,
            notable_events=context.notable_events[:10],
            event_categories=context.event_categories,
            context_summary=context.get_context_summary(),
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting temporal context: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/vectorize/enriched", response_model=EnrichedVectorizeResponse)
async def vectorize_enriched(request: EnrichedVectorizeRequest):
    """
    Enhanced vectorization with temporal and contextual enrichment.
    
    Pipeline:
    1. Enrich with temporal context (dates, world events)
    2. Generate Gemma context descriptions for each section
    3. Create hybrid embeddings (raw + context + temporal)
    4. Return enriched sections ready for storage
    """
    import uuid
    
    try:
        from src.temporal_context import get_temporal_enricher
        from src.hybrid_embedder import get_hybrid_embedder, Section, EmbeddingConfig
        
        job_id = f"cide_{uuid.uuid4().hex[:12]}"
        
        # Parse recording date
        try:
            recording_date = datetime.fromisoformat(request.recording_date)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid date format: {request.recording_date}"
            )
        
        # Get temporal context
        enricher = get_temporal_enricher()
        temporal_context = await enricher.enrich(
            recording_date=recording_date,
            meeting_type=request.meeting_type,
            business_events=request.business_context.get("events", []),
            fetch_world_events=request.fetch_world_events,
            market_context=request.business_context.get("market", {}),
        )
        
        # Encode temporal features
        temporal_features = enricher.encode_context(temporal_context)
        
        # Convert input sections to Section objects
        sections = []
        for sec_input in request.sections:
            section = Section(
                index=sec_input.index,
                text=sec_input.text,
                speaker=sec_input.speaker,
                start_time_sec=sec_input.start_time_sec,
                end_time_sec=sec_input.end_time_sec,
                metadata=sec_input.metadata,
            )
            sections.append(section)
        
        # Generate Gemma context descriptions if requested
        if request.generate_gemma_context:
            for section in sections:
                try:
                    gemma_context = await _generate_section_context(
                        section.text,
                        temporal_context,
                        request.business_context,
                    )
                    section.gemma_context = gemma_context
                except Exception as e:
                    logger.warning(f"Gemma context generation failed for section {section.index}: {e}")
                    section.gemma_context = ""
        
        # Generate hybrid embeddings
        embedder = get_hybrid_embedder()
        sections = embedder.embed_sections(
            sections,
            temporal_features,
            combination_mode=request.combination_mode,
        )
        
        # Prepare response
        output_sections = []
        for section in sections:
            output_sections.append({
                "index": section.index,
                "text_preview": section.text[:200],
                "gemma_context": section.gemma_context[:300] if section.gemma_context else "",
                "speaker": section.speaker,
                "has_embedding": section.hybrid_embedding is not None,
                "embedding": section.hybrid_embedding.tolist() if section.hybrid_embedding is not None else None,
                "metadata": {
                    **section.metadata,
                    "recording_date": recording_date.isoformat(),
                    "day_of_week": temporal_context.day_of_week,
                    "quarter": temporal_context.quarter,
                    "fiscal_quarter": temporal_context.fiscal_quarter,
                },
            })
        
        # Calculate embedding dimension
        emb_dim = 0
        if sections and sections[0].hybrid_embedding is not None:
            emb_dim = len(sections[0].hybrid_embedding)
        
        return EnrichedVectorizeResponse(
            job_id=job_id,
            sections_processed=len(sections),
            embedding_dimension=emb_dim,
            temporal_context=temporal_context.to_dict(),
            sections=output_sections,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in enriched vectorization: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/discover", response_model=DiscoveryResponse)
async def discover_insights(request: DiscoveryRequest):
    """
    Run chaotic insight discovery across sections.
    
    Discovery modes:
    - temporal_bridging: Find patterns across similar time periods
    - semantic_outliers: Connect unrelated topics with hidden links
    - contradiction: Find evolving or conflicting viewpoints
    - emergent_patterns: Detect topic drift over time
    """
    try:
        import numpy as np
        from src.chaos_comparator import (
            get_discovery_engine,
            DiscoveryConfig,
        )
        
        if not request.sections:
            raise HTTPException(
                status_code=400,
                detail="No sections provided for discovery"
            )
        
        # Configure discovery
        config = DiscoveryConfig(
            modes=request.discovery_modes,
            max_insights=request.max_insights,
            min_chaos_score=request.min_chaos_score,
            min_temporal_gap=request.min_temporal_gap_days,
        )
        
        # Get discovery engine
        engine = get_discovery_engine()
        
        # Convert embeddings from lists to numpy arrays
        sections = []
        for sec in request.sections:
            section_copy = dict(sec)
            if "embedding" in section_copy and section_copy["embedding"]:
                section_copy["embedding"] = np.array(
                    section_copy["embedding"], dtype=np.float32
                )
            sections.append(section_copy)
        
        # Run discovery
        report = await engine.discover_insights(sections, config=config)
        
        return DiscoveryResponse(
            transcription_count=report.transcription_count,
            section_count=report.section_count,
            pairs_analyzed=report.pairs_analyzed,
            insights=report.insights,
            processing_time_sec=report.processing_time_sec,
            discovery_modes=request.discovery_modes,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in insight discovery: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-context", response_model=GemmaContextResponse)
async def generate_context(request: GemmaContextRequest):
    """
    Generate a Gemma context description for a section.
    
    Useful for individual section processing or testing.
    """
    from datetime import datetime
    
    start_time = datetime.now()
    
    try:
        context = await _generate_section_context(
            request.section_text,
            request.temporal_context,
            request.business_context,
            max_tokens=request.max_tokens,
        )
        
        elapsed_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        
        return GemmaContextResponse(
            context=context,
            token_count=len(context.split()),  # Approximate
            processing_time_ms=elapsed_ms,
        )
        
    except Exception as e:
        logger.error(f"Error generating context: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def cide_health():
    """Check CIDE module health."""
    health = {
        "status": "healthy",
        "modules": {},
    }
    
    # Check temporal context module
    try:
        from src.temporal_context import get_temporal_enricher
        enricher = get_temporal_enricher()
        health["modules"]["temporal_context"] = "loaded"
    except Exception as e:
        health["modules"]["temporal_context"] = f"error: {e}"
        health["status"] = "degraded"
    
    # Check hybrid embedder
    try:
        from src.hybrid_embedder import get_hybrid_embedder
        embedder = get_hybrid_embedder()
        health["modules"]["hybrid_embedder"] = "loaded"
    except Exception as e:
        health["modules"]["hybrid_embedder"] = f"error: {e}"
        health["status"] = "degraded"
    
    # Check chaos comparator
    try:
        from src.chaos_comparator import get_discovery_engine
        engine = get_discovery_engine()
        health["modules"]["chaos_comparator"] = "loaded"
    except Exception as e:
        health["modules"]["chaos_comparator"] = f"error: {e}"
        health["status"] = "degraded"
    
    return health


# =============================================================================
# Helper Functions
# =============================================================================


async def _generate_section_context(
    section_text: str,
    temporal_context: Any,
    business_context: dict[str, Any],
    max_tokens: int = 300,
) -> str:
    """
    Use Gemma to generate a rich contextual description of a section.
    
    This description captures:
    - Main topics discussed
    - Decisions made or action items
    - Emotional undertones
    - Connection to temporal events
    - Semantic categories
    """
    import httpx
    
    GEMMA_SERVICE_URL = os.getenv("GEMMA_SERVICE_URL", "http://gemma-service:8001")
    
    # Build temporal context string
    if hasattr(temporal_context, 'get_context_summary'):
        temporal_str = temporal_context.get_context_summary()
    elif isinstance(temporal_context, dict):
        temporal_str = f"""
Date: {temporal_context.get('recording_date', 'Unknown')}
Quarter: Q{temporal_context.get('quarter', '?')} {temporal_context.get('year', '')}
Notable Events: {', '.join(temporal_context.get('notable_events', [])[:3])}
        """.strip()
    else:
        temporal_str = "No temporal context available"
    
    # Build business context string
    business_str = ""
    if business_context:
        items = [f"{k}: {v}" for k, v in list(business_context.items())[:5]]
        business_str = "\n".join(items)
    
    prompt = f"""Analyze this business meeting section and create a rich contextual description.

SECTION TEXT:
{section_text[:1500]}

TEMPORAL CONTEXT:
{temporal_str}

BUSINESS CONTEXT:
{business_str if business_str else 'General business discussion'}

Create a dense contextual description (one paragraph) that:
1. Summarizes the key points discussed
2. Notes any decisions, action items, or concerns
3. Identifies emotional undertones (urgency, optimism, caution, etc.)
4. Connects discussion to external events if relevant
5. Tags with semantic categories (e.g., strategy, operations, finance, risk, planning)

Write a paragraph optimized for semantic embedding that captures the full meaning of this section in context.

CONTEXTUAL DESCRIPTION:"""
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{GEMMA_SERVICE_URL}/chat",
                json={
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": 0.3,  # Lower temperature for consistent descriptions
                },
            )
            
            if response.status_code == 200:
                data = response.json()
                context = data.get("message", "") or data.get("response", "")
                return context.strip()
            else:
                logger.warning(f"Gemma call failed: {response.status_code}")
                return ""
                
    except Exception as e:
        logger.error(f"Error calling Gemma: {e}")
        return ""


# =============================================================================
# V2 Scoring Endpoints (Quantitative Business Analysis)
# =============================================================================


@router.post("/v2/score", response_model=V2ScoreSegmentResponse)
async def score_segment_v2(request: V2ScoreSegmentRequest):
    """
    Score a single transcript segment using Gemma.
    
    Returns quantitative scores (1-10) for:
    - Business practice adherence
    - Industry best practices
    - Deadline stress indicators
    - Emotional conflict levels
    - Decision clarity
    - Speaker confidence
    - Action orientation
    - Risk awareness
    """
    import uuid
    
    try:
        from src.gemma_scorer import get_gemma_scorer
        
        scorer = get_gemma_scorer()
        
        scores = await scorer.score_segment(
            segment_text=request.segment_text,
            speaker=request.speaker,
            meeting_type=request.meeting_type,
            company_context=request.company_context,
            recording_date=request.recording_date,
        )
        
        return V2ScoreSegmentResponse(
            segment_id=str(uuid.uuid4()),
            scores={
                "business_practice_adherence": scores.business_practice_adherence,
                "industry_best_practices": scores.industry_best_practices,
                "deadline_stress": scores.deadline_stress,
                "emotional_conflict": scores.emotional_conflict,
                "decision_clarity": scores.decision_clarity,
                "speaker_confidence": scores.speaker_confidence,
                "action_orientation": scores.action_orientation,
                "risk_awareness": scores.risk_awareness,
            },
            extracted={
                "topics": scores.topics,
                "decisions": scores.decisions,
                "action_items": scores.action_items,
                "concerns": scores.concerns,
            },
            overall_tone=scores.overall_tone,
            health_score=scores.health_score(),
            model_confidence=scores.model_confidence,
            justifications=scores.justifications,
        )
        
    except Exception as e:
        logger.error(f"Error scoring segment: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v2/score/batch", response_model=V2ScoreBatchResponse)
async def score_batch_v2(request: V2ScoreBatchRequest):
    """
    Score multiple transcript segments in batch.
    
    Optionally stores results in the score database for correlation analysis.
    """
    try:
        from src.gemma_scorer import get_gemma_scorer, get_score_database
        
        scorer = get_gemma_scorer()
        
        # Score all segments
        scored_segments = await scorer.score_segments(
            segments=request.segments,
            company_context=request.company_context,
            batch_delay=0.2,  # Small delay to avoid overwhelming Gemma
        )
        
        # Store if requested
        if request.store_results and scored_segments:
            try:
                db = get_score_database()
                db.store_scored_segments(scored_segments)
                logger.info(f"Stored {len(scored_segments)} scored segments")
            except Exception as e:
                logger.warning(f"Failed to store scores: {e}")
        
        # Calculate summary stats
        health_scores = [s.scores.health_score() for s in scored_segments]
        avg_health = sum(health_scores) / len(health_scores) if health_scores else 0
        high_stress_count = sum(
            1 for s in scored_segments if s.scores.deadline_stress >= 7
        )
        
        # Prepare results
        results = []
        for seg in scored_segments:
            results.append({
                "segment_id": seg.segment_id,
                "speaker": seg.speaker,
                "health_score": seg.scores.health_score(),
                "deadline_stress": seg.scores.deadline_stress,
                "emotional_conflict": seg.scores.emotional_conflict,
                "overall_tone": seg.scores.overall_tone,
                "topics": seg.scores.topics,
            })
        
        return V2ScoreBatchResponse(
            segments_scored=len(scored_segments),
            segments_failed=len(request.segments) - len(scored_segments),
            avg_health_score=round(avg_health, 2),
            high_stress_count=high_stress_count,
            results=results,
        )
        
    except Exception as e:
        logger.error(f"Error in batch scoring: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v2/correlation", response_model=V2CorrelationResponse)
async def get_correlation_data_v2(request: V2CorrelationRequest):
    """
    Get aggregated score data for correlation analysis.
    
    Returns data suitable for comparing against business databases
    (Salesforce, banking, operations) to find predictive patterns.
    """
    try:
        from src.gemma_scorer import get_score_database
        
        db = get_score_database()
        
        # Get aggregated data
        aggregated = db.get_correlation_data()
        
        # Get high stress segments
        high_stress = db.get_high_stress_segments(
            threshold=request.stress_threshold
        )
        
        # Filter by date range if provided
        if request.start_date and request.end_date:
            aggregated = [
                d for d in aggregated
                if request.start_date <= d.get("date", "") <= request.end_date
            ]
            high_stress = [
                s for s in high_stress
                if request.start_date <= s.get("recording_date", "")[:10] <= request.end_date
            ]
        
        # Calculate summary stats
        summary = {
            "total_segments": sum(d.get("segment_count", 0) for d in aggregated),
            "avg_stress": round(
                sum(d.get("avg_stress", 0) for d in aggregated) / len(aggregated), 2
            ) if aggregated else 0,
            "avg_health": round(
                sum(d.get("avg_health", 0) for d in aggregated) / len(aggregated), 2
            ) if aggregated else 0,
            "high_stress_segment_count": len(high_stress),
            "date_count": len(set(d.get("date") for d in aggregated)),
        }
        
        return V2CorrelationResponse(
            date_range={
                "start": request.start_date or "all",
                "end": request.end_date or "all",
            },
            aggregated_data=aggregated[:100],  # Limit for response size
            high_stress_segments=high_stress[:50],
            summary_stats=summary,
        )
        
    except Exception as e:
        logger.error(f"Error getting correlation data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/v2/health")
async def v2_health():
    """Check V2 scoring module health."""
    health = {
        "status": "healthy",
        "modules": {},
        "database": {},
    }
    
    # Check gemma scorer
    try:
        from src.gemma_scorer import get_gemma_scorer
        scorer = get_gemma_scorer()
        health["modules"]["gemma_scorer"] = "loaded"
    except Exception as e:
        health["modules"]["gemma_scorer"] = f"error: {e}"
        health["status"] = "degraded"
    
    # Check score database
    try:
        from src.gemma_scorer import get_score_database
        db = get_score_database()
        health["modules"]["score_database"] = "loaded"
        health["database"]["path"] = db.db_path
    except Exception as e:
        health["modules"]["score_database"] = f"error: {e}"
        health["status"] = "degraded"
    
    return health

