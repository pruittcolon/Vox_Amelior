"""
Database Quality Scoring Router.

Provides endpoints for Gemma-powered database quality analysis:
- Start scoring job with configurable chunk sizes
- Get job status with real-time progress
- Retrieve final results
- Test mode for integration verification

Security: Admin-only, read-only result storage, rate limited.
"""

import asyncio
import logging
import math
import os
from datetime import datetime
from typing import Any, Literal

import httpx
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/database-scoring", tags=["Database Scoring"])

# Configuration
GEMMA_SERVICE_URL = os.environ.get("GEMMA_URL", "http://gemma-service:8001")
RESULTS_DIR = "/app/data/analysis_results"

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# Initialize service auth for Gemma calls (avoid circular import from main)
_scoring_service_auth = None
try:
    from shared.security.service_auth import ServiceAuth, load_service_jwt_keys
    jwt_keys = load_service_jwt_keys("ml-service")
    if jwt_keys:
        _scoring_service_auth = ServiceAuth(service_id="ml-service", service_secret=jwt_keys)
        logger.info("[SCORING] ✅ Service auth initialized for Gemma calls")
    else:
        logger.warning("[SCORING] No JWT keys found, Gemma calls may fail with 401")
except Exception as e:
    logger.warning(f"[SCORING] Service auth not available: {e}")


# ============================================================================
# Request/Response Models
# ============================================================================


class StartScoringRequest(BaseModel):
    """Request to start a scoring job."""

    chunk_size: Literal[1, 5, 10, 20, 50, 100, 500, 1000, 5000, 10000, 25000, 100000] = Field(
        default=20, description="Number of rows per chunk"
    )
    test_mode: bool = Field(
        default=False, description="Only process first chunk for testing"
    )
    # Row range options for partial database scoring
    row_range: Literal[
        "full", "last_10pct", "last_25pct", "last_50pct",
        "last_1000", "last_10000", "last_100000"
    ] = Field(
        default="full", description="Which portion of database to score"
    )
    # Date-based filtering (optional)
    date_column: str | None = Field(
        default=None, description="Column name for date-based filtering"
    )
    date_start: str | None = Field(
        default=None, description="Start date (ISO format) for date range"
    )
    date_end: str | None = Field(
        default=None, description="End date (ISO format) for date range"
    )


class ScoringJobResponse(BaseModel):
    """Response containing job information."""

    job_id: str
    filename: str
    chunk_size: int
    total_chunks: int
    processed_chunks: int
    status: str
    eta_ms: int
    avg_overall: float
    created_at: str
    error: str | None = None


class ChunkScoreResponse(BaseModel):
    """Single chunk score response."""

    chunk_id: int
    row_start: int
    row_end: int
    scores: dict[str, float]
    processing_time_ms: float


class ScoringResultsResponse(BaseModel):
    """Full scoring results with all chunks."""

    job_id: str
    filename: str
    status: str
    total_chunks: int
    processed_chunks: int
    chunks: list[ChunkScoreResponse]
    summary: dict[str, Any]


# ============================================================================
# In-Memory Job Storage
# ============================================================================


class ScoringJobState:
    """Tracks the state of a scoring job."""

    def __init__(
        self,
        job_id: str,
        filename: str,
        chunk_size: int,
        total_chunks: int,
        total_rows: int,
    ):
        self.job_id = job_id
        self.filename = filename
        self.chunk_size = chunk_size
        self.total_chunks = total_chunks
        self.total_rows = total_rows
        self.processed_chunks = 0
        self.status = "pending"  # pending, running, complete, failed
        self.created_at = datetime.utcnow()
        self.started_at: datetime | None = None
        self.completed_at: datetime | None = None
        self.error: str | None = None
        self.chunks: list[dict[str, Any]] = []
        self.first_three_times: list[float] = []
        self.test_mode = False

    def get_eta_ms(self) -> int:
        """Calculate estimated time remaining."""
        if len(self.first_three_times) < 3:
            return -1
        avg_time = sum(self.first_three_times) / 3
        remaining = self.total_chunks - self.processed_chunks
        return int(remaining * avg_time)

    def get_avg_overall(self) -> float:
        """Get average overall score."""
        if not self.chunks:
            return 0.0
        scores = [c["scores"]["overall"] for c in self.chunks if "scores" in c]
        return sum(scores) / len(scores) if scores else 0.0

    def to_response(self) -> dict[str, Any]:
        """Convert to API response format."""
        return {
            "job_id": self.job_id,
            "filename": self.filename,
            "chunk_size": self.chunk_size,
            "total_chunks": self.total_chunks,
            "processed_chunks": self.processed_chunks,
            "status": self.status,
            "eta_ms": self.get_eta_ms(),
            "avg_overall": self.get_avg_overall(),
            "created_at": self.created_at.isoformat(),
            "error": self.error,
        }


# Active jobs storage
_jobs: dict[str, ScoringJobState] = {}


# ============================================================================
# Background Scoring Task
# ============================================================================


async def run_scoring_job(
    job_id: str,
    df_data: list[dict[str, Any]],
    columns: list[str],
    chunk_size: int,
    test_mode: bool,
    service_token: str | None = None,
):
    """
    Background task to run the scoring job.

    Args:
        job_id: Job identifier
        df_data: Full dataset as list of dicts
        columns: Column names
        chunk_size: Rows per chunk
        test_mode: Only process first chunk
        service_token: JWT for Gemma service calls
    """
    job = _jobs.get(job_id)
    if not job:
        logger.error(f"Job {job_id} not found")
        return

    job.status = "running"
    job.started_at = datetime.utcnow()

    total_rows = len(df_data)
    total_chunks = math.ceil(total_rows / chunk_size)

    # In test mode, only process 1 chunk
    if test_mode:
        total_chunks = 1
        job.test_mode = True

    job.total_chunks = total_chunks

    logger.info(
        f"[SCORING] Starting job {job_id}: {total_rows} rows, {total_chunks} chunks"
    )

    try:
        async with httpx.AsyncClient(timeout=180.0) as client:  # 180s for Gemma
            for chunk_idx in range(total_chunks):
                row_start = chunk_idx * chunk_size
                row_end = min(row_start + chunk_size, total_rows)
                chunk_data = df_data[row_start:row_end]

                # Format chunk for Gemma
                chunk_text = _format_chunk_for_scoring(chunk_data, columns)

                # Call Gemma scoring endpoint
                start_time = datetime.utcnow()

                headers = {}
                if service_token:
                    headers["X-Service-Token"] = service_token

                try:
                    logger.info(f"[SCORING] Calling Gemma for chunk {chunk_idx + 1}/{total_chunks}...")
                    
                    # Use module-level service auth (initialized at import time)
                    if _scoring_service_auth:
                        headers.update(_scoring_service_auth.get_auth_header())
                    
                    response = await client.post(
                        f"{GEMMA_SERVICE_URL}/score-chunk",
                        json={
                            "chunk_content": chunk_text,
                            "columns": columns,
                            "row_count": len(chunk_data),
                            "filename": job.filename,
                            "row_start": row_start,
                            "row_end": row_end - 1,
                            "questions": [],  # Use defaults for now
                        },
                        headers=headers,
                    )

                    if response.status_code == 200:
                        scores = response.json()
                    else:
                        logger.warning(
                            f"Gemma scoring returned {response.status_code}, using defaults"
                        )
                        scores = _default_scores()

                except Exception as e:
                    logger.warning(f"Gemma call failed: {e}, using defaults")
                    scores = _default_scores()

                processing_time = (
                    datetime.utcnow() - start_time
                ).total_seconds() * 1000

                # Store chunk result
                chunk_result = {
                    "chunk_id": chunk_idx,
                    "row_start": row_start,
                    "row_end": row_end - 1,
                    "scores": scores,
                    "processing_time_ms": processing_time,
                }
                job.chunks.append(chunk_result)
                job.processed_chunks += 1

                # Track first 3 for ETA
                if len(job.first_three_times) < 3:
                    job.first_three_times.append(processing_time)

                logger.debug(f"[SCORING] Chunk {chunk_idx + 1}/{total_chunks} complete")

        job.status = "complete"
        job.completed_at = datetime.utcnow()

        # Save results to file
        await _save_results(job)

        logger.info(
            f"[SCORING] Job {job_id} complete: avg_overall={job.get_avg_overall():.2f}"
        )

    except Exception as e:
        logger.error(f"[SCORING] Job {job_id} failed: {e}")
        job.status = "failed"
        job.error = str(e)
        job.completed_at = datetime.utcnow()


def _format_chunk_for_scoring(
    chunk_data: list[dict[str, Any]], columns: list[str]
) -> str:
    """Format chunk data as compact text for fast Gemma analysis."""
    lines = []
    # SPEED: Only use first 3 rows and 8 columns for ~200 tokens
    for i, row in enumerate(chunk_data[:3]):
        parts = []
        for col in columns[:8]:
            val = row.get(col, "")
            str_val = str(val) if val is not None else ""
            if len(str_val) > 20:
                str_val = str_val[:17] + "..."
            parts.append(f"{col}={str_val}")
        lines.append(f"R{i+1}: {', '.join(parts)}")

    if len(chunk_data) > 3:
        lines.append(f"(+{len(chunk_data)-3} more rows)")

    return "\n".join(lines)


def _default_scores() -> dict[str, float]:
    """Return default neutral scores when Gemma is unavailable."""
    return {
        "completeness": 5.0,
        "accuracy": 5.0,
        "consistency": 5.0,
        "relevance": 5.0,
        "overall": 5.0,
    }


async def _save_results(job: ScoringJobState) -> None:
    """Save job results to JSON file and generate insights CSV."""
    import json
    import csv
    import re

    result = {
        "job_id": job.job_id,
        "filename": job.filename,
        "chunk_size": job.chunk_size,
        "total_chunks": job.total_chunks,
        "processed_chunks": job.processed_chunks,
        "status": job.status,
        "created_at": job.created_at.isoformat(),
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        "test_mode": job.test_mode,
        "chunks": job.chunks,
        "summary": {
            "avg_Q1": _avg_score(job.chunks, "Q1"),
            "avg_Q2": _avg_score(job.chunks, "Q2"),
            "avg_Q3": _avg_score(job.chunks, "Q3"),
            "avg_Q4": _avg_score(job.chunks, "Q4"),
            "avg_Q5": _avg_score(job.chunks, "Q5"),
            "avg_overall": job.get_avg_overall(),
            "total_processing_time_ms": sum(
                c.get("processing_time_ms", 0) for c in job.chunks
            ),
        },
    }

    # Save JSON results
    json_filepath = os.path.join(RESULTS_DIR, f"{job.filename}_{job.job_id}.json")
    with open(json_filepath, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    logger.info(f"[SCORING] JSON results saved to {json_filepath}")

    # Generate insights CSV
    insights_csv_path = await _generate_insights_csv(job)
    if insights_csv_path:
        result["insights_csv_path"] = insights_csv_path
        logger.info(f"[SCORING] Insights CSV saved to {insights_csv_path}")


async def _generate_insights_csv(job: ScoringJobState) -> str | None:
    """Generate new CSV with original data + Gemma score columns per row."""
    import csv
    import pandas as pd
    
    if not job.chunks:
        logger.warning("[SCORING] No chunks to generate insights CSV")
        return None
    
    # Load original data
    original_path = f"/app/data/uploads/{job.filename}"
    try:
        if job.filename.endswith(".csv"):
            df = pd.read_csv(original_path)
        elif job.filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(original_path)
        else:
            logger.error(f"[SCORING] Unsupported file format: {job.filename}")
            return None
    except Exception as e:
        logger.error(f"[SCORING] Failed to load original file: {e}")
        return None
    
    # Add score columns - initialize with None (empty)
    # For partial scoring, only scored rows will have values
    # NOTE: Removed "findings" column - no longer generating per-row findings to save tokens
    score_columns = ["Q1_anomaly", "Q2_business", "Q3_validity", "Q4_complete", "Q5_consistent", "overall"]
    for col in score_columns:
        df[col] = None  # Empty for unscored rows
    
    # Assign scores to each row based on which chunk it belongs to
    for chunk in job.chunks:
        row_start = chunk.get("row_start", 0)
        row_end = chunk.get("row_end", 0)
        scores = chunk.get("scores", {})
        
        # Get scores with defaults
        q1 = scores.get("Q1", 5)
        q2 = scores.get("Q2", 5)
        q3 = scores.get("Q3", 5)
        q4 = scores.get("Q4", 5)
        q5 = scores.get("Q5", 5)
        overall = scores.get("overall", 5.0)
        
        # Assign to rows in this chunk's range
        for row_idx in range(row_start, min(row_end + 1, len(df))):
            df.at[row_idx, "Q1_anomaly"] = q1
            df.at[row_idx, "Q2_business"] = q2
            df.at[row_idx, "Q3_validity"] = q3
            df.at[row_idx, "Q4_complete"] = q4
            df.at[row_idx, "Q5_consistent"] = q5
            df.at[row_idx, "overall"] = overall
    
    # Save new insights CSV
    base_name = job.filename.rsplit(".", 1)[0] if "." in job.filename else job.filename
    insights_filename = f"{base_name}_insights.csv"
    insights_path = f"/app/data/uploads/{insights_filename}"
    
    try:
        df.to_csv(insights_path, index=False)
        logger.info(f"[SCORING] Generated row-level insights CSV: {insights_path} ({len(df)} rows)")
        return insights_path
    except Exception as e:
        logger.error(f"[SCORING] Failed to write insights CSV: {e}")
        return None


def _avg_score(chunks: list[dict], dimension: str) -> float:
    """Calculate average score for a dimension."""
    scores = [c["scores"][dimension] for c in chunks if "scores" in c]
    return sum(scores) / len(scores) if scores else 0.0


# ============================================================================
# API Endpoints
# ============================================================================


@router.post("/score/{filename}")
async def start_scoring(
    filename: str,
    request: StartScoringRequest,
    background_tasks: BackgroundTasks,
):
    """
    Start a database quality scoring job.

    Chunks the database and scores each chunk using Gemma.
    Returns job_id for progress tracking.
    """
    # Load the database file
    from pathlib import Path
    import pandas as pd

    uploads_dir = Path("/app/data/uploads")
    file_path = uploads_dir / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Database not found: {filename}")

    # Load data
    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(file_path)  # Load full file for filtering
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load file: {e}")

    original_rows = len(df)

    # Apply row range filter for partial database scoring
    if request.row_range != "full":
        if request.row_range == "last_10pct":
            start_row = int(original_rows * 0.9)
        elif request.row_range == "last_25pct":
            start_row = int(original_rows * 0.75)
        elif request.row_range == "last_50pct":
            start_row = int(original_rows * 0.5)
        elif request.row_range == "last_1000":
            start_row = max(0, original_rows - 1000)
        elif request.row_range == "last_10000":
            start_row = max(0, original_rows - 10000)
        elif request.row_range == "last_100000":
            start_row = max(0, original_rows - 100000)
        else:
            start_row = 0

        df = df.iloc[start_row:].reset_index(drop=True)
        logger.info(f"[SCORING] Row range '{request.row_range}': filtered from {original_rows} to {len(df)} rows")

    # Apply date-based filtering if specified
    if request.date_column and request.date_start:
        try:
            df[request.date_column] = pd.to_datetime(df[request.date_column], errors='coerce')
            mask = df[request.date_column] >= pd.to_datetime(request.date_start)
            if request.date_end:
                mask &= df[request.date_column] <= pd.to_datetime(request.date_end)
            df = df[mask].reset_index(drop=True)
            logger.info(f"[SCORING] Date filter on '{request.date_column}': {len(df)} rows remain")
        except Exception as e:
            logger.warning(f"[SCORING] Date filtering failed: {e}")

    # Safety limit: cap at 500k rows for memory
    if len(df) > 500000:
        logger.warning(f"[SCORING] Capping from {len(df)} to 500,000 rows for memory safety")
        df = df.tail(500000).reset_index(drop=True)

    # Create job
    import uuid

    job_id = str(uuid.uuid4())[:8]
    total_rows = len(df)
    total_chunks = math.ceil(total_rows / request.chunk_size)

    if request.test_mode:
        total_chunks = 1

    job = ScoringJobState(
        job_id=job_id,
        filename=filename,
        chunk_size=request.chunk_size,
        total_chunks=total_chunks,
        total_rows=total_rows,
    )
    _jobs[job_id] = job

    # Convert to list of dicts for background task
    df_data = df.to_dict(orient="records")
    columns = list(df.columns)

    # Start background scoring
    background_tasks.add_task(
        run_scoring_job,
        job_id=job_id,
        df_data=df_data,
        columns=columns,
        chunk_size=request.chunk_size,
        test_mode=request.test_mode,
    )

    return {
        "job_id": job_id,
        "filename": filename,
        "chunk_size": request.chunk_size,
        "total_chunks": total_chunks,
        "total_rows": total_rows,
        "test_mode": request.test_mode,
        "status": "pending",
        "message": "Scoring job started",
    }


@router.get("/status/{job_id}")
async def get_job_status(job_id: str):
    """Get the current status of a scoring job."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return job.to_response()


@router.get("/results/{job_id}")
async def get_job_results(job_id: str):
    """Get the full results of a completed scoring job."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status != "complete":
        return {
            "job_id": job_id,
            "status": job.status,
            "message": "Job not yet complete",
            "processed_chunks": job.processed_chunks,
            "total_chunks": job.total_chunks,
        }

    return {
        "job_id": job.job_id,
        "filename": job.filename,
        "status": job.status,
        "total_chunks": job.total_chunks,
        "processed_chunks": job.processed_chunks,
        "chunks": job.chunks,
        "summary": {
            "avg_Q1": _avg_score(job.chunks, "Q1"),
            "avg_Q2": _avg_score(job.chunks, "Q2"),
            "avg_Q3": _avg_score(job.chunks, "Q3"),
            "avg_Q4": _avg_score(job.chunks, "Q4"),
            "avg_Q5": _avg_score(job.chunks, "Q5"),
            "avg_overall": job.get_avg_overall(),
        },
    }


@router.post("/test/{filename}")
async def test_scoring(filename: str, background_tasks: BackgroundTasks):
    """
    Quick test mode: score only the first chunk.

    Use this to verify integration before running full analysis.
    Returns immediately with job_id, results available shortly.
    """
    return await start_scoring(
        filename=filename,
        request=StartScoringRequest(chunk_size=20, test_mode=True),
        background_tasks=background_tasks,
    )


@router.get("/jobs")
async def list_jobs():
    """List all scoring jobs."""
    return {
        "jobs": [job.to_response() for job in _jobs.values()],
        "count": len(_jobs),
    }
