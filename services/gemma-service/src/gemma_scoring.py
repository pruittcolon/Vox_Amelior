"""
Gemma Database Quality Scoring Module.

Analyzes database content chunks using Gemma LLM and provides quality scores
across 5 dimensions: Completeness, Accuracy, Consistency, Relevance, Overall.

Follows OWASP security practices for LLM input/output handling.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable
from uuid import uuid4

logger = logging.getLogger(__name__)


@dataclass
class ChunkScore:
    """Quality scores for a single chunk of data."""

    chunk_id: int
    row_start: int
    row_end: int
    completeness: float
    accuracy: float
    consistency: float
    relevance: float
    overall: float
    processing_time_ms: float
    raw_response: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "chunk_id": self.chunk_id,
            "row_start": self.row_start,
            "row_end": self.row_end,
            "scores": {
                "completeness": self.completeness,
                "accuracy": self.accuracy,
                "consistency": self.consistency,
                "relevance": self.relevance,
                "overall": self.overall,
            },
            "processing_time_ms": self.processing_time_ms,
        }


@dataclass
class ScoringJob:
    """Tracks the progress of a scoring job."""

    job_id: str = field(default_factory=lambda: str(uuid4())[:8])
    filename: str = ""
    chunk_size: int = 20
    total_chunks: int = 0
    processed_chunks: int = 0
    status: str = "pending"  # pending, running, complete, failed, cancelled
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error: str | None = None
    chunks: list[ChunkScore] = field(default_factory=list)

    # ETA calculation
    first_three_times_ms: list[float] = field(default_factory=list)

    def get_eta_ms(self) -> int:
        """Calculate estimated time remaining in milliseconds."""
        if len(self.first_three_times_ms) < 3:
            return -1  # Not enough data
        avg_time = sum(self.first_three_times_ms) / 3
        remaining = self.total_chunks - self.processed_chunks
        return int(remaining * avg_time)

    def get_avg_overall_score(self) -> float:
        """Calculate average overall score across all chunks."""
        if not self.chunks:
            return 0.0
        return sum(c.overall for c in self.chunks) / len(self.chunks)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "job_id": self.job_id,
            "filename": self.filename,
            "chunk_size": self.chunk_size,
            "total_chunks": self.total_chunks,
            "processed_chunks": self.processed_chunks,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error": self.error,
            "eta_ms": self.get_eta_ms(),
            "avg_overall": self.get_avg_overall_score(),
            "chunks": [c.to_dict() for c in self.chunks],
        }


class GemmaScoringAnalyzer:
    """
    Score database chunks using Gemma LLM with structured prompts.

    Each chunk is analyzed for 5 quality dimensions on a 1-10 scale:
    - Completeness: Are all expected fields populated?
    - Accuracy: Does the data appear correct and valid?
    - Consistency: Are formats and values internally consistent?
    - Relevance: Is the data relevant to the expected schema?
    - Overall: Holistic quality assessment

    Usage:
        analyzer = GemmaScoringAnalyzer(llm_callable)
        score = await analyzer.score_chunk(chunk_data, columns)
    """

    SCORING_PROMPT = """You are a data quality analyst. Analyze this database section and provide quality scores from 1 to 10.

COLUMNS: {columns}

DATA SAMPLE (showing {row_count} rows):
{chunk_content}

Rate each dimension:
1. COMPLETENESS (1-10): What percentage of fields have values? Are there many nulls/blanks?
2. ACCURACY (1-10): Do values appear valid? Are there obvious typos or impossible values?
3. CONSISTENCY (1-10): Are formats uniform? Are similar values formatted the same way?
4. RELEVANCE (1-10): Does the data match what the column names suggest it should contain?
5. OVERALL (1-10): Your holistic quality assessment considering all factors.

IMPORTANT: Respond ONLY with valid JSON in this exact format:
{{"completeness": X, "accuracy": X, "consistency": X, "relevance": X, "overall": X}}

Replace X with numbers 1-10. No other text."""

    def __init__(
        self,
        llm_callable: Callable | None = None,
        default_temperature: float = 0.1,
        default_max_tokens: int = 256,
    ):
        """
        Initialize the scoring analyzer.

        Args:
            llm_callable: Async function to call Gemma (signature: prompt, max_tokens, temperature)
            default_temperature: Lower = more deterministic scores
            default_max_tokens: Keep small for structured output
        """
        self.llm_callable = llm_callable
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens

    def format_chunk_for_analysis(
        self,
        chunk_data: list[dict[str, Any]],
        columns: list[str],
        max_display_rows: int = 10,
    ) -> str:
        """
        Format chunk data as readable text for Gemma.

        Args:
            chunk_data: List of row dictionaries
            columns: Column names
            max_display_rows: Limit rows shown to avoid token overflow

        Returns:
            Formatted string representation
        """
        # Limit display for very large chunks
        display_data = chunk_data[:max_display_rows]

        lines = []
        for i, row in enumerate(display_data):
            row_parts = []
            for col in columns:
                value = row.get(col, "")
                # Truncate very long values
                str_val = str(value) if value is not None else ""
                if len(str_val) > 50:
                    str_val = str_val[:47] + "..."
                row_parts.append(f"{col}={str_val}")
            lines.append(f"Row {i + 1}: {', '.join(row_parts)}")

        if len(chunk_data) > max_display_rows:
            lines.append(f"... and {len(chunk_data) - max_display_rows} more rows")

        return "\n".join(lines)

    def parse_score_response(self, response: str) -> dict[str, float]:
        """
        Parse Gemma's JSON response into scores.

        Args:
            response: Raw LLM response

        Returns:
            Dictionary with score values, defaults to 5.0 on parse failure
        """
        default_scores = {
            "completeness": 5.0,
            "accuracy": 5.0,
            "consistency": 5.0,
            "relevance": 5.0,
            "overall": 5.0,
        }

        # Try to find JSON in response
        try:
            # Look for JSON pattern
            json_match = re.search(r"\{[^}]+\}", response)
            if json_match:
                json_str = json_match.group()
                parsed = json.loads(json_str)

                # Validate and clamp scores
                for key in default_scores:
                    if key in parsed:
                        try:
                            value = float(parsed[key])
                            # Clamp to 1-10 range
                            default_scores[key] = max(1.0, min(10.0, value))
                        except (ValueError, TypeError):
                            pass

                return default_scores

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse score JSON: {e}")

        return default_scores

    async def score_chunk(
        self,
        chunk_data: list[dict[str, Any]],
        columns: list[str],
        chunk_id: int = 0,
        row_start: int = 0,
        row_end: int = 0,
    ) -> ChunkScore:
        """
        Score a single chunk of data.

        Args:
            chunk_data: List of row dictionaries
            columns: Column names
            chunk_id: Chunk identifier
            row_start: Starting row number in original data
            row_end: Ending row number in original data

        Returns:
            ChunkScore with all dimension scores
        """
        import time

        start_time = time.time()

        # Format content for analysis
        chunk_content = self.format_chunk_for_analysis(chunk_data, columns)

        # Build prompt
        prompt = self.SCORING_PROMPT.format(
            columns=", ".join(columns),
            row_count=len(chunk_data),
            chunk_content=chunk_content,
        )

        # Call LLM
        raw_response = ""
        if self.llm_callable:
            try:
                raw_response = await self.llm_callable(
                    prompt,
                    max_tokens=self.default_max_tokens,
                    temperature=self.default_temperature,
                )
            except Exception as e:
                logger.error(f"LLM call failed for chunk {chunk_id}: {e}")
                raw_response = '{"completeness": 5, "accuracy": 5, "consistency": 5, "relevance": 5, "overall": 5}'
        else:
            # No LLM - return neutral scores for testing
            raw_response = '{"completeness": 5, "accuracy": 5, "consistency": 5, "relevance": 5, "overall": 5}'

        # Parse response
        scores = self.parse_score_response(raw_response)

        processing_time_ms = (time.time() - start_time) * 1000

        return ChunkScore(
            chunk_id=chunk_id,
            row_start=row_start,
            row_end=row_end,
            completeness=scores["completeness"],
            accuracy=scores["accuracy"],
            consistency=scores["consistency"],
            relevance=scores["relevance"],
            overall=scores["overall"],
            processing_time_ms=processing_time_ms,
            raw_response=raw_response[:500],  # Truncate for storage
        )


# Active jobs storage (in-memory for now, could be Redis)
_active_jobs: dict[str, ScoringJob] = {}


def get_job(job_id: str) -> ScoringJob | None:
    """Get a scoring job by ID."""
    return _active_jobs.get(job_id)


def create_job(filename: str, chunk_size: int, total_chunks: int) -> ScoringJob:
    """Create and register a new scoring job."""
    job = ScoringJob(
        filename=filename,
        chunk_size=chunk_size,
        total_chunks=total_chunks,
    )
    _active_jobs[job.job_id] = job
    return job


def update_job_chunk(job_id: str, chunk_score: ChunkScore) -> None:
    """Update a job with a new chunk score."""
    job = _active_jobs.get(job_id)
    if job:
        job.chunks.append(chunk_score)
        job.processed_chunks += 1

        # Track first 3 for ETA
        if len(job.first_three_times_ms) < 3:
            job.first_three_times_ms.append(chunk_score.processing_time_ms)
