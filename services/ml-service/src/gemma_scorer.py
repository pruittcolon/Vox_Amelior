"""
Gemma Business Scorer - Quantitative Transcript Analysis
=========================================================

Uses Gemma to analyze transcript segments and produce structured
quantitative scores for business-relevant dimensions.

This module implements the V2 Database approach:
1. Gemma reads each segment
2. Produces structured JSON with scores (1-10)
3. Scores are stored in SQLite for correlation analysis

Author: NeMo Server Team
Version: 1.0.0
"""

import asyncio
import json
import logging
import os
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

# Try to import shared auth (available in container)
try:
    from shared.security.service_auth import get_service_auth, load_service_jwt_keys
    SHARED_AUTH_AVAILABLE = True
except ImportError:
    SHARED_AUTH_AVAILABLE = False


logger = logging.getLogger(__name__)

# Configuration
GEMMA_SERVICE_URL = os.getenv("GEMMA_SERVICE_URL", "http://gemma-service:8001")
SCORES_DB_PATH = os.getenv("SCORES_DB_PATH", "/app/data/transcript_scores.db")


@dataclass
class SegmentScores:
    """Quantitative scores for a transcript segment."""
    
    # Core business metrics (1-10)
    business_practice_adherence: int = 5
    industry_best_practices: int = 5
    deadline_stress: int = 1
    emotional_conflict: int = 1
    decision_clarity: int = 5
    speaker_confidence: int = 5
    action_orientation: int = 5
    risk_awareness: int = 5
    
    # Metadata
    overall_tone: str = "neutral"
    model_confidence: float = 0.5
    
    # Extracted content
    topics: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    action_items: list[str] = field(default_factory=list)
    concerns: list[str] = field(default_factory=list)
    
    # Justifications (optional)
    justifications: dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "scores": {
                "business_practice_adherence": self.business_practice_adherence,
                "industry_best_practices": self.industry_best_practices,
                "deadline_stress": self.deadline_stress,
                "emotional_conflict": self.emotional_conflict,
                "decision_clarity": self.decision_clarity,
                "speaker_confidence": self.speaker_confidence,
                "action_orientation": self.action_orientation,
                "risk_awareness": self.risk_awareness,
            },
            "overall_tone": self.overall_tone,
            "model_confidence": self.model_confidence,
            "extracted": {
                "topics": self.topics,
                "decisions": self.decisions,
                "action_items": self.action_items,
                "concerns": self.concerns,
            },
            "justifications": self.justifications,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "SegmentScores":
        """Create from dictionary (Gemma response)."""
        scores = data.get("scores", {})
        extracted = data.get("extracted", {})
        
        return cls(
            business_practice_adherence=scores.get("business_practice_adherence", 5),
            industry_best_practices=scores.get("industry_best_practices", 5),
            deadline_stress=scores.get("deadline_stress", 1),
            emotional_conflict=scores.get("emotional_conflict", 1),
            decision_clarity=scores.get("decision_clarity", 5),
            speaker_confidence=scores.get("speaker_confidence", 5),
            action_orientation=scores.get("action_orientation", 5),
            risk_awareness=scores.get("risk_awareness", 5),
            overall_tone=data.get("overall_tone", "neutral"),
            model_confidence=data.get("confidence", 0.5),
            topics=extracted.get("topics", []),
            decisions=extracted.get("decisions_made", []),
            action_items=extracted.get("action_items", []),
            concerns=extracted.get("concerns_raised", []),
            justifications=data.get("justifications", {}),
        )
    
    def health_score(self) -> float:
        """Calculate overall meeting health score (0-10)."""
        # Weighted average - low stress/conflict is good
        positive_avg = (
            self.business_practice_adherence +
            self.industry_best_practices +
            self.decision_clarity +
            self.speaker_confidence +
            self.action_orientation +
            self.risk_awareness
        ) / 6
        
        # Invert stress/conflict (high = bad)
        negative_penalty = (
            self.deadline_stress + self.emotional_conflict
        ) / 2
        
        # Health = positive factors - penalty (bounded 0-10)
        health = positive_avg - (negative_penalty * 0.3)
        return max(0, min(10, health))


@dataclass
class ScoredSegment:
    """A transcript segment with its scores."""
    
    id: str
    segment_id: str
    transcription_id: str
    recording_date: datetime
    speaker: str
    meeting_type: str
    text: str
    scores: SegmentScores
    created_at: datetime = field(default_factory=datetime.now)


class GemmaScorer:
    """
    Uses Gemma to score transcript segments quantitatively.
    
    Usage:
        scorer = GemmaScorer()
        scores = await scorer.score_segment(
            segment_text="We need to accelerate the timeline...",
            speaker="CEO",
            meeting_type="strategy",
            company_context="Tech startup in growth phase"
        )
    """
    
    def __init__(
        self,
        gemma_url: str = GEMMA_SERVICE_URL,
        service_auth_getter: callable = None
    ):
        """Initialize scorer."""
        self.gemma_url = gemma_url
        self._get_service_headers = service_auth_getter or (lambda: {})

        # Auto-initialize auth if not provided and available
        if not service_auth_getter and SHARED_AUTH_AVAILABLE:
            try:
                # Load keys and init auth - strict security check
                # Note: In production, secrets are in /run/secrets
                jwt_keys = load_service_jwt_keys("ml-service")
                self.auth = get_service_auth(service_id="ml-service", service_secret=jwt_keys)
                
                # Define getter to generate fresh token on each call
                self._get_service_headers = lambda: {
                    "X-Service-Token": self.auth.create_token(
                        aud="internal", 
                        expires_in=60  # Short expiry for security
                    )
                }
                logger.info("✅ GemmaScorer initialized with shared.security.service_auth")
            except Exception as e:
                logger.warning(f"⚠️ Failed to auto-initialize GemmaScorer auth: {e}")

    
    def _build_prompt(
        self,
        segment_text: str,
        speaker: str = "Unknown",
        meeting_type: str = "general",
        company_context: str = "",
        recording_date: str = ""
    ) -> str:
        """Build the scoring prompt for Gemma."""
        
        return f"""You are an expert business analyst evaluating internal meeting transcripts to quantify business health indicators.

CONTEXT:
- Company: {company_context or "Business organization"}
- Meeting Type: {meeting_type}
- Date: {recording_date or "Not specified"}
- Speaker: {speaker}

TRANSCRIPT SEGMENT:
\"\"\"
{segment_text[:2000]}
\"\"\"

Evaluate this segment and return a JSON object with the following structure:

{{
  "scores": {{
    "business_practice_adherence": <1-10>,
    "industry_best_practices": <1-10>,
    "deadline_stress": <1-10>,
    "emotional_conflict": <1-10>,
    "decision_clarity": <1-10>,
    "speaker_confidence": <1-10>,
    "action_orientation": <1-10>,
    "risk_awareness": <1-10>
  }},
  "justifications": {{
    "business_practice_adherence": "<one sentence>",
    "deadline_stress": "<one sentence>",
    "emotional_conflict": "<one sentence>"
  }},
  "extracted": {{
    "topics": ["<topic1>", "<topic2>"],
    "decisions_made": ["<decision or empty>"],
    "action_items": ["<action or empty>"],
    "concerns_raised": ["<concern or empty>"]
  }},
  "overall_tone": "<positive|neutral|negative|mixed>",
  "confidence": <0.0-1.0>
}}

SCORING GUIDELINES:
- 1-3: Poor/Absent - Clear problems or missing entirely
- 4-6: Moderate - Present but with room for improvement  
- 7-9: Good - Solid execution with minor gaps
- 10: Excellent - Best-in-class execution

For deadline_stress and emotional_conflict:
- 1-3: Calm, no issues
- 4-6: Some pressure or tension
- 7-10: High stress or significant conflict

Return ONLY the JSON object, no other text."""
    
    async def score_segment(
        self,
        segment_text: str,
        speaker: str = "Unknown",
        meeting_type: str = "general",
        company_context: str = "",
        recording_date: str = ""
    ) -> SegmentScores:
        """
        Score a single transcript segment using Gemma.
        
        Args:
            segment_text: The transcript text to analyze
            speaker: Who said this
            meeting_type: Type of meeting (strategy, operations, etc.)
            company_context: Optional context about the company
            recording_date: When this was recorded
            
        Returns:
            SegmentScores with quantitative assessments
        """
        prompt = self._build_prompt(
            segment_text=segment_text,
            speaker=speaker,
            meeting_type=meeting_type,
            company_context=company_context,
            recording_date=recording_date,
        )
        
        try:
            import httpx
            
            async with httpx.AsyncClient(timeout=90.0) as client:
                response = await client.post(
                    f"{self.gemma_url}/chat",
                    json={
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 800,
                        "temperature": 0.1,  # Low temp for consistent scoring
                    },
                    headers=self._get_service_headers(),
                )
                
                if response.status_code != 200:
                    logger.warning(f"Gemma call failed: {response.status_code}")
                    return SegmentScores()  # Return defaults
                
                data = response.json()
                response_text = data.get("message", "") or data.get("response", "")
                
                # Parse JSON from response
                scores_data = self._parse_json_response(response_text)
                
                if scores_data:
                    return SegmentScores.from_dict(scores_data)
                else:
                    logger.warning("Failed to parse Gemma response as JSON")
                    return SegmentScores()
                    
        except Exception as e:
            logger.error(f"Error scoring segment: {e}")
            return SegmentScores()
    
    def _parse_json_response(self, response_text: str) -> dict | None:
        """Extract and parse JSON from Gemma response."""
        # Try direct parse
        try:
            return json.loads(response_text.strip())
        except json.JSONDecodeError:
            pass
        
        # Try to find JSON in response
        import re
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        return None
    
    async def score_segments(
        self,
        segments: list[dict[str, Any]],
        company_context: str = "",
        batch_delay: float = 0.1
    ) -> list[ScoredSegment]:
        """
        Score multiple segments.
        
        Args:
            segments: List of segment dicts with text, speaker, etc.
            company_context: Context about the company
            batch_delay: Delay between requests to avoid overload
            
        Returns:
            List of ScoredSegment objects
        """
        results = []
        
        for seg in segments:
            segment_text = seg.get("text", "")
            if not segment_text or len(segment_text.strip()) < 10:
                continue
            
            scores = await self.score_segment(
                segment_text=segment_text,
                speaker=seg.get("speaker", "Unknown"),
                meeting_type=seg.get("meeting_type", "general"),
                company_context=company_context,
                recording_date=str(seg.get("recording_date", "")),
            )
            
            scored = ScoredSegment(
                id=str(uuid.uuid4()),
                segment_id=seg.get("id", str(uuid.uuid4())),
                transcription_id=seg.get("transcription_id", ""),
                recording_date=seg.get("recording_date", datetime.now()),
                speaker=seg.get("speaker", "Unknown"),
                meeting_type=seg.get("meeting_type", "general"),
                text=segment_text,
                scores=scores,
            )
            results.append(scored)
            
            if batch_delay > 0:
                await asyncio.sleep(batch_delay)
        
        return results


class ScoreDatabase:
    """
    SQLite storage for transcript scores.
    
    Enables correlation analysis against business databases.
    """
    
    def __init__(self, db_path: str = SCORES_DB_PATH):
        """Initialize database connection."""
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_tables()
    
    def _init_tables(self):
        """Create tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS transcript_scores (
                    id TEXT PRIMARY KEY,
                    segment_id TEXT NOT NULL,
                    transcription_id TEXT NOT NULL,
                    recording_date TEXT,
                    speaker TEXT,
                    meeting_type TEXT,
                    text TEXT,
                    
                    -- Core Scores
                    business_practice_adherence INTEGER,
                    industry_best_practices INTEGER,
                    deadline_stress INTEGER,
                    emotional_conflict INTEGER,
                    decision_clarity INTEGER,
                    speaker_confidence INTEGER,
                    action_orientation INTEGER,
                    risk_awareness INTEGER,
                    
                    -- Derived
                    health_score REAL,
                    
                    -- Metadata
                    overall_tone TEXT,
                    model_confidence REAL,
                    topics TEXT,
                    decisions TEXT,
                    action_items TEXT,
                    concerns TEXT,
                    
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_scores_date 
                ON transcript_scores(recording_date)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_scores_transcription 
                ON transcript_scores(transcription_id)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_scores_stress 
                ON transcript_scores(deadline_stress)
            """)
            
            # Aggregated meeting scores
            conn.execute("""
                CREATE TABLE IF NOT EXISTS meeting_scores (
                    id TEXT PRIMARY KEY,
                    transcription_id TEXT UNIQUE NOT NULL,
                    recording_date TEXT,
                    meeting_type TEXT,
                    segment_count INTEGER,
                    
                    avg_business_practice REAL,
                    avg_deadline_stress REAL,
                    avg_emotional_conflict REAL,
                    avg_decision_clarity REAL,
                    avg_confidence REAL,
                    
                    max_stress_score INTEGER,
                    conflict_segment_count INTEGER,
                    decisions_made_count INTEGER,
                    action_items_count INTEGER,
                    
                    meeting_health_score REAL,
                    
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
    
    def store_scored_segment(self, segment: ScoredSegment):
        """Store a scored segment."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO transcript_scores (
                    id, segment_id, transcription_id, recording_date,
                    speaker, meeting_type, text,
                    business_practice_adherence, industry_best_practices,
                    deadline_stress, emotional_conflict, decision_clarity,
                    speaker_confidence, action_orientation, risk_awareness,
                    health_score, overall_tone, model_confidence,
                    topics, decisions, action_items, concerns, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                segment.id,
                segment.segment_id,
                segment.transcription_id,
                segment.recording_date.isoformat() if isinstance(segment.recording_date, datetime) else str(segment.recording_date),
                segment.speaker,
                segment.meeting_type,
                segment.text[:500],  # Truncate for storage
                segment.scores.business_practice_adherence,
                segment.scores.industry_best_practices,
                segment.scores.deadline_stress,
                segment.scores.emotional_conflict,
                segment.scores.decision_clarity,
                segment.scores.speaker_confidence,
                segment.scores.action_orientation,
                segment.scores.risk_awareness,
                segment.scores.health_score(),
                segment.scores.overall_tone,
                segment.scores.model_confidence,
                json.dumps(segment.scores.topics),
                json.dumps(segment.scores.decisions),
                json.dumps(segment.scores.action_items),
                json.dumps(segment.scores.concerns),
                segment.created_at.isoformat(),
            ))
            conn.commit()
    
    def store_scored_segments(self, segments: list[ScoredSegment]):
        """Store multiple scored segments."""
        for segment in segments:
            self.store_scored_segment(segment)
    
    def calculate_meeting_scores(self, transcription_id: str) -> dict[str, Any]:
        """Calculate aggregated scores for a meeting."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as segment_count,
                    AVG(business_practice_adherence) as avg_business_practice,
                    AVG(deadline_stress) as avg_deadline_stress,
                    AVG(emotional_conflict) as avg_emotional_conflict,
                    AVG(decision_clarity) as avg_decision_clarity,
                    AVG(speaker_confidence) as avg_confidence,
                    MAX(deadline_stress) as max_stress_score,
                    SUM(CASE WHEN emotional_conflict > 6 THEN 1 ELSE 0 END) as conflict_segment_count,
                    MIN(recording_date) as recording_date,
                    meeting_type,
                    AVG(health_score) as meeting_health_score
                FROM transcript_scores
                WHERE transcription_id = ?
                GROUP BY transcription_id
            """, (transcription_id,))
            
            row = cursor.fetchone()
            
            if row:
                return dict(row)
            return {}
    
    def get_scores_by_date_range(
        self,
        start_date: str,
        end_date: str
    ) -> list[dict]:
        """Get all scores within a date range."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            cursor = conn.execute("""
                SELECT *
                FROM transcript_scores
                WHERE recording_date BETWEEN ? AND ?
                ORDER BY recording_date
            """, (start_date, end_date))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_high_stress_segments(self, threshold: int = 7) -> list[dict]:
        """Get segments with high deadline stress."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            cursor = conn.execute("""
                SELECT *
                FROM transcript_scores
                WHERE deadline_stress >= ?
                ORDER BY deadline_stress DESC, recording_date DESC
            """, (threshold,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_correlation_data(self) -> list[dict]:
        """Get aggregated data suitable for correlation analysis."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            cursor = conn.execute("""
                SELECT 
                    DATE(recording_date) as date,
                    meeting_type,
                    AVG(deadline_stress) as avg_stress,
                    AVG(emotional_conflict) as avg_conflict,
                    AVG(speaker_confidence) as avg_confidence,
                    AVG(health_score) as avg_health,
                    COUNT(*) as segment_count
                FROM transcript_scores
                GROUP BY DATE(recording_date), meeting_type
                ORDER BY date
            """)
            
            return [dict(row) for row in cursor.fetchall()]


# =============================================================================
# Singleton Instances
# =============================================================================

_gemma_scorer: GemmaScorer | None = None
_score_database: ScoreDatabase | None = None


def get_gemma_scorer(
    service_auth_getter: callable = None
) -> GemmaScorer:
    """Get or create the Gemma scorer singleton."""
    global _gemma_scorer
    if _gemma_scorer is None:
        _gemma_scorer = GemmaScorer(service_auth_getter=service_auth_getter)
    return _gemma_scorer


def get_score_database(db_path: str = None) -> ScoreDatabase:
    """Get or create the score database singleton."""
    global _score_database
    if _score_database is None:
        _score_database = ScoreDatabase(db_path or SCORES_DB_PATH)
    return _score_database
