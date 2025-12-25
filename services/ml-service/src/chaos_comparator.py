"""
Chaos Comparator - Unexpected Insight Discovery Engine
=======================================================

Finds unexpected insights by comparing semantically distant sections
that share hidden patterns (temporal, categorical, or emergent).

Based on the principle that valuable insights often come from
connecting dots that aren't obviously related - the "chaos" in
chaotic comparison refers to finding order in apparent disorder.

Discovery Modes:
1. Temporal Bridging: Compare sections from similar temporal contexts
   across different meetings (e.g., all Q4 discussions over 3 years)
2. Semantic Outliers: Find sections that are semantically distant
   but share categorical tags
3. Contradiction Detection: Find conflicting statements across time
4. Emergent Patterns: Detect topic drift and evolution over time

Author: NeMo Server Team
Version: 1.0.0
"""

import asyncio
import hashlib
import json
import logging
import os
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)

# Configuration
GEMMA_SERVICE_URL = os.getenv("GEMMA_SERVICE_URL", "http://gemma-service:8001")
RAG_SERVICE_URL = os.getenv("RAG_SERVICE_URL", "http://rag-service:8004")


@dataclass
class ChaoticPair:
    """
    A pair of sections discovered through chaotic comparison.
    
    Represents two sections that are semantically distant but
    connected through temporal, categorical, or other hidden patterns.
    """
    
    section_a_idx: int
    section_b_idx: int
    discovery_mode: str              # Which mode found this pair
    connection_type: str             # Specific connection type
    chaos_score: float               # Lower semantic similarity = higher chaos
    temporal_similarity: float       # How similar in time context
    insight: str = ""                # Gemma-generated insight
    confidence: float = 0.0          # Confidence in the connection
    
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "section_a_idx": self.section_a_idx,
            "section_b_idx": self.section_b_idx,
            "discovery_mode": self.discovery_mode,
            "connection_type": self.connection_type,
            "chaos_score": self.chaos_score,
            "temporal_similarity": self.temporal_similarity,
            "insight": self.insight,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


@dataclass
class DiscoveryConfig:
    """Configuration for insight discovery."""
    
    modes: list[str] = field(default_factory=lambda: [
        "temporal_bridging",
        "semantic_outliers",
        "contradiction"
    ])
    max_insights: int = 10
    min_chaos_score: float = 0.3     # Minimum semantic distance
    max_chaos_score: float = 0.9     # Maximum (beyond this, likely unrelated)
    min_temporal_gap: int = 7        # Minimum days between sections
    max_temporal_gap: int = 365      # Maximum days (1 year)
    
    # Mode-specific settings
    temporal_window_days: int = 30   # For temporal bridging
    outlier_threshold: float = 0.4   # For semantic outliers


@dataclass
class InsightReport:
    """
    Report generated from chaotic insight discovery.
    """
    
    transcription_count: int
    section_count: int
    pairs_analyzed: int
    insights: list[dict[str, Any]]
    generated_at: datetime
    config: DiscoveryConfig = field(default_factory=DiscoveryConfig)
    processing_time_sec: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "transcription_count": self.transcription_count,
            "section_count": self.section_count,
            "pairs_analyzed": self.pairs_analyzed,
            "insights": self.insights,
            "generated_at": self.generated_at.isoformat(),
            "processing_time_sec": self.processing_time_sec,
            "config": {
                "modes": self.config.modes,
                "max_insights": self.config.max_insights,
                "min_chaos_score": self.config.min_chaos_score,
            },
        }


class ChaosComparator:
    """
    Finds unexpected connections between sections through chaotic comparison.
    
    The algorithm:
    1. Instead of finding similar sections (like traditional search),
       find sections that are semantically DISTANT but share hidden patterns
    2. Hidden patterns include:
       - Same temporal context (quarter, month-end, year)
       - Same speaker discussing different topics
       - Evolution of stance on a topic over time
       - Contradiction between past and present statements
    3. Use Gemma to synthesize insights from these chaotic pairs
    """
    
    def __init__(
        self,
        gemma_url: str = GEMMA_SERVICE_URL,
        service_auth_getter: Callable[[], dict] | None = None
    ):
        """
        Initialize comparator.
        
        Args:
            gemma_url: URL for Gemma service
            service_auth_getter: Function that returns auth headers
        """
        self.gemma_url = gemma_url
        self._get_service_headers = service_auth_getter or (lambda: {})
    
    def compute_semantic_similarity(
        self,
        embedding_a: np.ndarray,
        embedding_b: np.ndarray
    ) -> float:
        """Compute cosine similarity between embeddings."""
        dot = np.dot(embedding_a, embedding_b)
        norm_a = np.linalg.norm(embedding_a)
        norm_b = np.linalg.norm(embedding_b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return float(dot / (norm_a * norm_b))
    
    def compute_temporal_similarity(
        self,
        meta_a: dict[str, Any],
        meta_b: dict[str, Any]
    ) -> float:
        """
        Compute how similar two sections are temporally.
        
        Returns higher value if:
        - Same day of week
        - Same time of year (month)
        - Same quarter
        - Same fiscal period patterns
        """
        score = 0.0
        
        # Same day of week
        if meta_a.get("day_of_week") == meta_b.get("day_of_week"):
            score += 0.15
        
        # Same month
        if meta_a.get("month") == meta_b.get("month"):
            score += 0.25
        
        # Same quarter
        if meta_a.get("quarter") == meta_b.get("quarter"):
            score += 0.20
        
        # Same fiscal quarter
        if meta_a.get("fiscal_quarter") == meta_b.get("fiscal_quarter"):
            score += 0.15
        
        # Both month-end
        if meta_a.get("is_month_end") and meta_b.get("is_month_end"):
            score += 0.15
        
        # Both quarter-end
        if meta_a.get("is_quarter_end") and meta_b.get("is_quarter_end"):
            score += 0.10
        
        return min(score, 1.0)
    
    def find_chaotic_pairs(
        self,
        section_embeddings: list[np.ndarray],
        metadata: list[dict[str, Any]],
        config: DiscoveryConfig
    ) -> list[ChaoticPair]:
        """
        Find all chaotic pairs across discovery modes.
        
        Args:
            section_embeddings: List of hybrid embeddings
            metadata: List of metadata dicts for each section
            config: Discovery configuration
            
        Returns:
            List of ChaoticPair objects, sorted by chaos_score
        """
        all_pairs = []
        
        for mode in config.modes:
            if mode == "temporal_bridging":
                pairs = self._temporal_bridging_discovery(
                    section_embeddings, metadata, config
                )
            elif mode == "semantic_outliers":
                pairs = self._semantic_outlier_discovery(
                    section_embeddings, metadata, config
                )
            elif mode == "contradiction":
                pairs = self._contradiction_discovery(
                    section_embeddings, metadata, config
                )
            elif mode == "emergent_patterns":
                pairs = self._emergent_pattern_discovery(
                    section_embeddings, metadata, config
                )
            else:
                logger.warning(f"Unknown discovery mode: {mode}")
                continue
            
            all_pairs.extend(pairs)
        
        # Remove duplicates (same pair found by different modes)
        unique_pairs = self._deduplicate_pairs(all_pairs)
        
        # Sort by chaos score (higher = more interesting)
        unique_pairs.sort(key=lambda p: p.chaos_score, reverse=True)
        
        return unique_pairs[:config.max_insights * 2]  # Extra for filtering
    
    def _temporal_bridging_discovery(
        self,
        embeddings: list[np.ndarray],
        metadata: list[dict[str, Any]],
        config: DiscoveryConfig
    ) -> list[ChaoticPair]:
        """
        Temporal Bridging: Find sections from different meetings but
        similar temporal contexts (same quarter, season) that discuss
        different topics.
        
        The insight: How did perspective change on similar issues
        across the same time of year in different years?
        """
        pairs = []
        n = len(embeddings)
        
        for i in range(n):
            for j in range(i + 1, n):
                meta_a = metadata[i]
                meta_b = metadata[j]
                
                # Must be from different transcriptions
                if meta_a.get("transcription_id") == meta_b.get("transcription_id"):
                    continue
                
                # Check temporal gap
                date_a = self._parse_date(meta_a.get("recording_date"))
                date_b = self._parse_date(meta_b.get("recording_date"))
                
                if date_a and date_b:
                    gap_days = abs((date_a - date_b).days)
                    if gap_days < config.min_temporal_gap:
                        continue
                    if gap_days > config.max_temporal_gap:
                        continue
                
                # Compute similarities
                semantic_sim = self.compute_semantic_similarity(
                    embeddings[i], embeddings[j]
                )
                temporal_sim = self.compute_temporal_similarity(meta_a, meta_b)
                
                # Chaos score: semantically distant but temporally similar
                chaos_score = (1 - semantic_sim) * temporal_sim
                
                if (config.min_chaos_score <= chaos_score <= config.max_chaos_score
                    and temporal_sim >= 0.3):
                    
                    connection_type = self._determine_temporal_connection(
                        meta_a, meta_b
                    )
                    
                    pairs.append(ChaoticPair(
                        section_a_idx=i,
                        section_b_idx=j,
                        discovery_mode="temporal_bridging",
                        connection_type=connection_type,
                        chaos_score=chaos_score,
                        temporal_similarity=temporal_sim,
                        metadata={
                            "semantic_similarity": semantic_sim,
                            "temporal_gap_days": gap_days if date_a and date_b else None,
                        }
                    ))
        
        return pairs
    
    def _semantic_outlier_discovery(
        self,
        embeddings: list[np.ndarray],
        metadata: list[dict[str, Any]],
        config: DiscoveryConfig
    ) -> list[ChaoticPair]:
        """
        Semantic Outliers: Find sections with low semantic similarity
        but shared categorical tags or speakers.
        
        The insight: How does the same person or category show up
        in unexpectedly different contexts?
        """
        pairs = []
        n = len(embeddings)
        
        for i in range(n):
            for j in range(i + 1, n):
                meta_a = metadata[i]
                meta_b = metadata[j]
                
                # Look for shared attributes
                shared_speaker = (
                    meta_a.get("speaker") == meta_b.get("speaker") and
                    meta_a.get("speaker") not in [None, "unknown", ""]
                )
                
                shared_categories = set(meta_a.get("categories", [])) & \
                                   set(meta_b.get("categories", []))
                
                shared_tags = set(meta_a.get("tags", [])) & \
                             set(meta_b.get("tags", []))
                
                # Need at least one shared attribute
                if not (shared_speaker or shared_categories or shared_tags):
                    continue
                
                # Compute semantic similarity
                semantic_sim = self.compute_semantic_similarity(
                    embeddings[i], embeddings[j]
                )
                
                # We want LOW semantic similarity (outliers)
                if semantic_sim > config.outlier_threshold:
                    continue
                
                # Chaos score based on how different yet connected they are
                connection_strength = (
                    (0.5 if shared_speaker else 0) +
                    (0.3 if shared_categories else 0) +
                    (0.2 if shared_tags else 0)
                )
                
                chaos_score = (1 - semantic_sim) * connection_strength
                
                if chaos_score >= config.min_chaos_score:
                    connection_type = self._determine_outlier_connection(
                        meta_a, meta_b, shared_speaker, 
                        shared_categories, shared_tags
                    )
                    
                    pairs.append(ChaoticPair(
                        section_a_idx=i,
                        section_b_idx=j,
                        discovery_mode="semantic_outliers",
                        connection_type=connection_type,
                        chaos_score=chaos_score,
                        temporal_similarity=0.0,  # Not relevant for this mode
                        metadata={
                            "semantic_similarity": semantic_sim,
                            "shared_speaker": shared_speaker,
                            "shared_categories": list(shared_categories),
                            "shared_tags": list(shared_tags),
                        }
                    ))
        
        return pairs
    
    def _contradiction_discovery(
        self,
        embeddings: list[np.ndarray],
        metadata: list[dict[str, Any]],
        config: DiscoveryConfig
    ) -> list[ChaoticPair]:
        """
        Contradiction Detection: Find sections that might represent
        evolving or conflicting viewpoints on the same topic.
        
        Uses moderate semantic similarity (same topic) but looks
        for sentiment or stance differences.
        """
        pairs = []
        n = len(embeddings)
        
        for i in range(n):
            for j in range(i + 1, n):
                # Moderate similarity suggests same topic
                semantic_sim = self.compute_semantic_similarity(
                    embeddings[i], embeddings[j]
                )
                
                # Looking for similar topics (0.4-0.7 range)
                if semantic_sim < 0.4 or semantic_sim > 0.7:
                    continue
                
                meta_a = metadata[i]
                meta_b = metadata[j]
                
                # Need temporal difference to detect evolution
                date_a = self._parse_date(meta_a.get("recording_date"))
                date_b = self._parse_date(meta_b.get("recording_date"))
                
                if not date_a or not date_b:
                    continue
                
                gap_days = abs((date_a - date_b).days)
                if gap_days < config.min_temporal_gap:
                    continue
                
                # Check for sentiment/emotion differences
                emotion_a = meta_a.get("emotion", "neutral")
                emotion_b = meta_b.get("emotion", "neutral")
                
                sentiment_a = meta_a.get("sentiment", 0.0)
                sentiment_b = meta_b.get("sentiment", 0.0)
                
                # Detect potential contradiction indicators
                emotion_shift = emotion_a != emotion_b
                sentiment_shift = abs(sentiment_a - sentiment_b) > 0.3
                
                if not (emotion_shift or sentiment_shift):
                    continue
                
                # Chaos score for contradictions
                chaos_score = semantic_sim * (0.5 + 0.5 * int(sentiment_shift))
                
                if chaos_score >= config.min_chaos_score:
                    pairs.append(ChaoticPair(
                        section_a_idx=i,
                        section_b_idx=j,
                        discovery_mode="contradiction",
                        connection_type="potential_evolution" if gap_days > 30 else "potential_conflict",
                        chaos_score=chaos_score,
                        temporal_similarity=0.0,
                        metadata={
                            "semantic_similarity": semantic_sim,
                            "temporal_gap_days": gap_days,
                            "emotion_a": emotion_a,
                            "emotion_b": emotion_b,
                            "sentiment_shift": abs(sentiment_a - sentiment_b),
                            "earlier_section": i if date_a < date_b else j,
                        }
                    ))
        
        return pairs
    
    def _emergent_pattern_discovery(
        self,
        embeddings: list[np.ndarray],
        metadata: list[dict[str, Any]],
        config: DiscoveryConfig
    ) -> list[ChaoticPair]:
        """
        Emergent Patterns: Detect topic drift and evolution over time.
        
        Looks for chains of sections that show gradual semantic drift
        while maintaining topical coherence.
        """
        # This is more complex - simplified version for now
        # Full implementation would use clustering and time-series analysis
        pairs = []
        
        # Sort by date
        dated_sections = [
            (i, self._parse_date(metadata[i].get("recording_date")))
            for i in range(len(embeddings))
            if self._parse_date(metadata[i].get("recording_date"))
        ]
        dated_sections.sort(key=lambda x: x[1])
        
        # Look for drift patterns (compare sections N apart)
        for step in [3, 5, 10]:
            for i in range(len(dated_sections) - step):
                idx_a, date_a = dated_sections[i]
                idx_b, date_b = dated_sections[i + step]
                
                semantic_sim = self.compute_semantic_similarity(
                    embeddings[idx_a], embeddings[idx_b]
                )
                
                gap_days = (date_b - date_a).days
                
                # Look for gradual drift (moderate similarity over time)
                if 0.3 <= semantic_sim <= 0.6 and gap_days > 30:
                    chaos_score = (1 - semantic_sim) * 0.7
                    
                    if chaos_score >= config.min_chaos_score:
                        pairs.append(ChaoticPair(
                            section_a_idx=idx_a,
                            section_b_idx=idx_b,
                            discovery_mode="emergent_patterns",
                            connection_type="topic_drift",
                            chaos_score=chaos_score,
                            temporal_similarity=0.5,
                            metadata={
                                "semantic_similarity": semantic_sim,
                                "temporal_gap_days": gap_days,
                                "step_size": step,
                            }
                        ))
        
        return pairs
    
    def _determine_temporal_connection(
        self,
        meta_a: dict,
        meta_b: dict
    ) -> str:
        """Determine the type of temporal connection."""
        connections = []
        
        if meta_a.get("quarter") == meta_b.get("quarter"):
            connections.append("same_quarter")
        
        if meta_a.get("month") == meta_b.get("month"):
            connections.append("same_month")
        
        if meta_a.get("is_quarter_end") and meta_b.get("is_quarter_end"):
            connections.append("quarter_end_pattern")
        
        if meta_a.get("is_month_end") and meta_b.get("is_month_end"):
            connections.append("month_end_pattern")
        
        if meta_a.get("day_of_week") == meta_b.get("day_of_week"):
            connections.append("same_weekday")
        
        return "+".join(connections) if connections else "temporal_pattern"
    
    def _determine_outlier_connection(
        self,
        meta_a: dict,
        meta_b: dict,
        shared_speaker: bool,
        shared_categories: set,
        shared_tags: set
    ) -> str:
        """Determine the type of outlier connection."""
        if shared_speaker:
            return f"same_speaker_{meta_a.get('speaker', 'unknown')}"
        elif shared_categories:
            return f"shared_category_{list(shared_categories)[0]}"
        elif shared_tags:
            return f"shared_tag_{list(shared_tags)[0]}"
        return "unknown_connection"
    
    def _deduplicate_pairs(
        self,
        pairs: list[ChaoticPair]
    ) -> list[ChaoticPair]:
        """Remove duplicate pairs, keeping highest chaos score."""
        seen = {}
        
        for pair in pairs:
            key = tuple(sorted([pair.section_a_idx, pair.section_b_idx]))
            
            if key not in seen or pair.chaos_score > seen[key].chaos_score:
                seen[key] = pair
        
        return list(seen.values())
    
    def _parse_date(self, date_val: Any) -> datetime | None:
        """Parse a date value from various formats."""
        if date_val is None:
            return None
        
        if isinstance(date_val, datetime):
            return date_val
        
        if isinstance(date_val, str):
            try:
                return datetime.fromisoformat(date_val.replace("Z", "+00:00"))
            except ValueError:
                pass
            
            try:
                return datetime.strptime(date_val, "%Y-%m-%d")
            except ValueError:
                pass
        
        return None
    
    async def generate_insight(
        self,
        section_a: dict[str, Any],
        section_b: dict[str, Any],
        pair: ChaoticPair
    ) -> str:
        """
        Use Gemma to synthesize an insight from the chaotic pair.
        
        Args:
            section_a: First section with text and metadata
            section_b: Second section with text and metadata
            pair: ChaoticPair with discovery details
            
        Returns:
            Generated insight string
        """
        # Build context-aware prompt based on discovery mode
        if pair.discovery_mode == "temporal_bridging":
            mode_context = """
            These sections are from different meetings but share similar temporal patterns
            (same time of year, same day of week, or similar business cycle position).
            Despite being recorded at different times, they occupy similar "positions" in
            the business calendar.
            """
        elif pair.discovery_mode == "semantic_outliers":
            mode_context = f"""
            These sections share {pair.connection_type} but discuss very different topics.
            This suggests an unexpected link between seemingly unrelated discussions.
            """
        elif pair.discovery_mode == "contradiction":
            mode_context = """
            These sections discuss similar topics but show signs of different stances
            or evolving viewpoints. They might represent a shift in perspective over time.
            """
        else:
            mode_context = """
            These sections show an interesting pattern worth exploring.
            """
        
        date_a = section_a.get("recording_date", "unknown date")
        date_b = section_b.get("recording_date", "unknown date")
        
        prompt = f"""You are an expert business analyst discovering hidden insights by comparing 
business discussions that aren't obviously related.

{mode_context}

SECTION A ({date_a}):
"{section_a.get('text', '')[:500]}"

SECTION B ({date_b}):
"{section_b.get('text', '')[:500]}"

CONNECTION TYPE: {pair.connection_type}
CHAOS SCORE: {pair.chaos_score:.2f} (higher = more unexpected connection)

Generate a business insight that:
1. Explains what's interesting about comparing these two sections
2. Identifies any hidden patterns, contradictions, or evolutions
3. Provides a strategic implication or question worth investigating
4. Would surprise and inform a business executive

Write a concise insight (2-3 sentences) that wouldn't be obvious from reading either section alone.
Focus on what emerges from the COMPARISON, not from the individual sections.

INSIGHT:"""
        
        try:
            import httpx
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.gemma_url}/chat",
                    json={
                        "messages": [
                            {"role": "user", "content": prompt}
                        ],
                        "max_tokens": 200,
                        "temperature": 0.7,  # Some creativity for insights
                    },
                    headers=self._get_service_headers()
                )
                
                if response.status_code == 200:
                    data = response.json()
                    insight = data.get("message", "") or data.get("response", "")
                    return insight.strip()
                else:
                    logger.warning(f"Gemma call failed: {response.status_code}")
                    return f"Connection found ({pair.connection_type}) - Gemma unavailable for synthesis."
                    
        except Exception as e:
            logger.error(f"Error generating insight: {e}")
            return f"Interesting connection: {pair.connection_type}"


class InsightDiscoveryEngine:
    """
    Main orchestrator for chaotic insight discovery.
    
    Coordinates the full pipeline:
    1. Load sections from storage
    2. Build temporal and semantic indices
    3. Run chaos comparator in multiple modes
    4. Generate Gemma insights for top pairs
    5. Compile into structured report
    """
    
    def __init__(
        self,
        rag_url: str = RAG_SERVICE_URL,
        gemma_url: str = GEMMA_SERVICE_URL,
        service_auth_getter: Callable[[], dict] | None = None
    ):
        """Initialize discovery engine."""
        self.rag_url = rag_url
        self.gemma_url = gemma_url
        self._get_service_headers = service_auth_getter or (lambda: {})
        
        self.comparator = ChaosComparator(
            gemma_url=gemma_url,
            service_auth_getter=service_auth_getter
        )
    
    async def discover_insights(
        self,
        sections: list[dict[str, Any]],
        embeddings: list[np.ndarray] | None = None,
        config: DiscoveryConfig | None = None
    ) -> InsightReport:
        """
        Run full insight discovery pipeline.
        
        Args:
            sections: List of section dicts with text, metadata, and embeddings
            embeddings: Optional pre-computed embeddings (extracted from sections if not provided)
            config: Discovery configuration
            
        Returns:
            InsightReport with discovered insights
        """
        start_time = datetime.now()
        config = config or DiscoveryConfig()
        
        if not sections:
            return InsightReport(
                transcription_count=0,
                section_count=0,
                pairs_analyzed=0,
                insights=[],
                generated_at=start_time,
                config=config,
            )
        
        logger.info(f"Starting insight discovery with {len(sections)} sections")
        
        # Extract embeddings if not provided
        if embeddings is None:
            embeddings = []
            for section in sections:
                emb = section.get("hybrid_embedding") or \
                      section.get("embedding") or \
                      section.get("raw_embedding")
                
                if emb is not None:
                    if isinstance(emb, list):
                        emb = np.array(emb, dtype=np.float32)
                    embeddings.append(emb)
                else:
                    # Placeholder if no embedding
                    embeddings.append(np.zeros(384, dtype=np.float32))
        
        # Extract metadata
        metadata = [section.get("metadata", section) for section in sections]
        
        # Find chaotic pairs
        logger.info(f"Finding chaotic pairs with modes: {config.modes}")
        pairs = self.comparator.find_chaotic_pairs(embeddings, metadata, config)
        
        logger.info(f"Found {len(pairs)} candidate pairs")
        
        # Generate insights for top pairs
        insights = []
        for pair in pairs[:config.max_insights]:
            section_a = sections[pair.section_a_idx]
            section_b = sections[pair.section_b_idx]
            
            # Generate Gemma insight
            insight_text = await self.comparator.generate_insight(
                section_a, section_b, pair
            )
            pair.insight = insight_text
            
            insights.append({
                "section_a_idx": pair.section_a_idx,
                "section_b_idx": pair.section_b_idx,
                "section_a_text": section_a.get("text", "")[:200],
                "section_b_text": section_b.get("text", "")[:200],
                "section_a_date": str(section_a.get("recording_date", "")),
                "section_b_date": str(section_b.get("recording_date", "")),
                "discovery_mode": pair.discovery_mode,
                "connection_type": pair.connection_type,
                "chaos_score": pair.chaos_score,
                "insight": pair.insight,
                "metadata": pair.metadata,
            })
            
            # Small delay to avoid overwhelming Gemma
            await asyncio.sleep(0.1)
        
        # Count unique transcriptions
        transcription_ids = set()
        for section in sections:
            tid = section.get("transcription_id") or \
                  section.get("metadata", {}).get("transcription_id")
            if tid:
                transcription_ids.add(tid)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        report = InsightReport(
            transcription_count=len(transcription_ids) or 1,
            section_count=len(sections),
            pairs_analyzed=len(pairs),
            insights=insights,
            generated_at=datetime.now(),
            config=config,
            processing_time_sec=processing_time,
        )
        
        logger.info(
            f"Insight discovery complete: {len(insights)} insights in "
            f"{processing_time:.1f}s"
        )
        
        return report
    
    async def discover_from_transcription_ids(
        self,
        transcription_ids: list[str],
        config: DiscoveryConfig | None = None
    ) -> InsightReport:
        """
        Discover insights from transcriptions stored in RAG service.
        
        Args:
            transcription_ids: List of transcription IDs to analyze
            config: Discovery configuration
            
        Returns:
            InsightReport with discovered insights
        """
        # Fetch sections from RAG service
        sections = await self._fetch_sections(transcription_ids)
        
        return await self.discover_insights(sections, config=config)
    
    async def _fetch_sections(
        self,
        transcription_ids: list[str]
    ) -> list[dict[str, Any]]:
        """Fetch sections from RAG service."""
        import httpx
        
        sections = []
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                for tid in transcription_ids:
                    # Assuming RAG service has an endpoint to get sections by transcription
                    response = await client.get(
                        f"{self.rag_url}/sections/{tid}",
                        headers=self._get_service_headers()
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        sections.extend(data.get("sections", []))
                        
        except Exception as e:
            logger.error(f"Error fetching sections: {e}")
        
        return sections


# =============================================================================
# Module-level convenience functions
# =============================================================================

_discovery_engine: InsightDiscoveryEngine | None = None


def get_discovery_engine(
    service_auth_getter: Callable[[], dict] | None = None
) -> InsightDiscoveryEngine:
    """Get or create the insight discovery engine singleton."""
    global _discovery_engine
    if _discovery_engine is None:
        _discovery_engine = InsightDiscoveryEngine(
            service_auth_getter=service_auth_getter
        )
    return _discovery_engine
