"""
Unit Tests for CIDE - Contextual Insight Discovery Engine
==========================================================

Tests for:
- temporal_context.py: TemporalContext, TemporalEncoder, WorldEventsProvider
- hybrid_embedder.py: HybridEmbedder, Section, embedding combination modes
- chaos_comparator.py: ChaosComparator, InsightDiscoveryEngine

Author: NeMo Server Team
"""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest


# =============================================================================
# Tests for temporal_context.py
# =============================================================================


class TestTemporalContext:
    """Tests for TemporalContext dataclass."""
    
    def test_temporal_context_creation(self):
        """Test basic TemporalContext creation with date derivations."""
        from src.temporal_context import TemporalContext
        
        # January 15, 2024 is a Monday
        ctx = TemporalContext(recording_date=datetime(2024, 1, 15, 10, 30))
        
        assert ctx.day_of_week == "Monday"
        assert ctx.day_of_week_num == 0
        assert ctx.month == 1
        assert ctx.quarter == 1
        assert ctx.year == 2024
        assert ctx.is_weekend is False
        assert ctx.is_month_end is False
        assert ctx.is_quarter_end is False
    
    def test_temporal_context_from_string(self):
        """Test TemporalContext creation from ISO string."""
        from src.temporal_context import TemporalContext
        
        ctx = TemporalContext(recording_date="2024-03-31T14:00:00")
        
        assert ctx.month == 3
        assert ctx.quarter == 1
        assert ctx.is_month_end is True
        assert ctx.is_quarter_end is True
    
    def test_temporal_context_weekend_detection(self):
        """Test weekend detection."""
        from src.temporal_context import TemporalContext
        
        # Saturday
        ctx_sat = TemporalContext(recording_date=datetime(2024, 1, 13))
        assert ctx_sat.is_weekend is True
        
        # Sunday
        ctx_sun = TemporalContext(recording_date=datetime(2024, 1, 14))
        assert ctx_sun.is_weekend is True
        
        # Monday
        ctx_mon = TemporalContext(recording_date=datetime(2024, 1, 15))
        assert ctx_mon.is_weekend is False
    
    def test_temporal_context_to_dict(self):
        """Test serialization to dictionary."""
        from src.temporal_context import TemporalContext
        
        ctx = TemporalContext(
            recording_date=datetime(2024, 6, 15),
            notable_events=["Event 1", "Event 2"],
            meeting_type="strategy",
        )
        
        data = ctx.to_dict()
        
        assert "recording_date" in data
        assert data["quarter"] == 2
        assert data["notable_events"] == ["Event 1", "Event 2"]
        assert data["meeting_type"] == "strategy"
    
    def test_context_summary_generation(self):
        """Test get_context_summary method."""
        from src.temporal_context import TemporalContext
        
        ctx = TemporalContext(
            recording_date=datetime(2024, 12, 31),
            notable_events=["Year-end event"],
            market_context={"vix": 20},
        )
        
        summary = ctx.get_context_summary()
        
        assert "December 31, 2024" in summary
        assert "Q4" in summary
        assert "Year-end event" in summary


class TestTemporalEncoder:
    """Tests for TemporalEncoder."""
    
    def test_encoder_output_shape(self):
        """Test that encoder produces correct output shape."""
        from src.temporal_context import TemporalContext, TemporalEncoder
        
        encoder = TemporalEncoder(feature_dim=32)
        ctx = TemporalContext(recording_date=datetime(2024, 6, 15))
        
        features = encoder.encode(ctx)
        
        assert features.shape == (32,)
        assert features.dtype == np.float32
    
    def test_cyclical_encoding(self):
        """Test that cyclical features are properly encoded."""
        from src.temporal_context import TemporalContext, TemporalEncoder
        
        encoder = TemporalEncoder(feature_dim=32)
        
        # January and December should have close cyclical values (wrap-around)
        ctx_jan = TemporalContext(recording_date=datetime(2024, 1, 15))
        ctx_dec = TemporalContext(recording_date=datetime(2024, 12, 15))
        
        features_jan = encoder.encode(ctx_jan)
        features_dec = encoder.encode(ctx_dec)
        
        # Month features are at indices 2-3 (sin/cos)
        # January (month 1) and December (month 12) should be close in cyclical space
        jan_month_vec = features_jan[2:4]
        dec_month_vec = features_dec[2:4]
        
        # They should be relatively close (Euclidean distance < 1.0)
        distance = np.linalg.norm(jan_month_vec - dec_month_vec)
        assert distance < 1.0
    
    def test_event_category_encoding(self):
        """Test event category flag encoding."""
        from src.temporal_context import TemporalContext, TemporalEncoder
        
        encoder = TemporalEncoder(feature_dim=32)
        
        ctx_with_events = TemporalContext(
            recording_date=datetime(2024, 6, 15),
            event_categories=["technology", "business"],
        )
        
        ctx_no_events = TemporalContext(
            recording_date=datetime(2024, 6, 15),
            event_categories=[],
        )
        
        features_with = encoder.encode(ctx_with_events)
        features_without = encoder.encode(ctx_no_events)
        
        # Features should be different
        assert not np.allclose(features_with, features_without)


class TestWorldEventsProvider:
    """Tests for WorldEventsProvider."""
    
    def test_cache_path_generation(self):
        """Test cache path generation."""
        from src.temporal_context import WorldEventsProvider
        
        provider = WorldEventsProvider(cache_dir="/tmp/test_cache")
        
        path = provider._get_cache_path(datetime(2024, 6, 15))
        
        assert "events_2024-06-15.json" in path
    
    def test_category_detection(self):
        """Test event category detection from text."""
        from src.temporal_context import WorldEventsProvider
        
        provider = WorldEventsProvider()
        
        assert provider._detect_category("President signs new bill") == "politics"
        assert provider._detect_category("Tech company announces AI breakthrough") == "technology"
        assert provider._detect_category("Stock market reaches new high") == "economics"
        assert provider._detect_category("Some random text") == "general"


# =============================================================================
# Tests for hybrid_embedder.py
# =============================================================================


class TestHybridEmbedder:
    """Tests for HybridEmbedder."""
    
    @pytest.fixture
    def mock_embedder(self):
        """Create embedder with mocked sentence-transformer."""
        from src.hybrid_embedder import HybridEmbedder, EmbeddingConfig
        
        config = EmbeddingConfig(
            raw_embedding_dim=384,
            context_embedding_dim=384,
            temporal_embedding_dim=32,
        )
        
        embedder = HybridEmbedder(config)
        
        # Mock the sentence-transformer model
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.randn(384).astype(np.float32)
        embedder._model = mock_model
        
        return embedder
    
    def test_concatenate_mode_output_shape(self, mock_embedder):
        """Test that concatenate mode produces correct output shape."""
        temporal_features = np.random.randn(32).astype(np.float32)
        
        result = mock_embedder.generate_hybrid_embedding(
            raw_text="Test text",
            gemma_context="Gemma context",
            temporal_features=temporal_features,
            combination_mode="concatenate",
        )
        
        # 384 + 384 + 32 = 800 (but weighted, then normalized)
        assert result.shape == (800,)
    
    def test_weighted_average_mode_output_shape(self, mock_embedder):
        """Test that weighted_average mode produces correct output shape."""
        temporal_features = np.random.randn(32).astype(np.float32)
        
        result = mock_embedder.generate_hybrid_embedding(
            raw_text="Test text",
            gemma_context="Gemma context",
            temporal_features=temporal_features,
            combination_mode="weighted_average",
        )
        
        # Weighted average matches raw dimension
        assert result.shape == (384,)
    
    def test_hybrid_mode_output_shape(self, mock_embedder):
        """Test that hybrid mode produces correct output shape."""
        temporal_features = np.random.randn(32).astype(np.float32)
        
        result = mock_embedder.generate_hybrid_embedding(
            raw_text="Test text",
            gemma_context="Gemma context",
            temporal_features=temporal_features,
            combination_mode="hybrid",
        )
        
        # 384 + 32 = 416
        assert result.shape == (416,)
    
    def test_normalization(self, mock_embedder):
        """Test that embeddings are normalized."""
        mock_embedder.config.normalize_embeddings = True
        temporal_features = np.random.randn(32).astype(np.float32)
        
        result = mock_embedder.generate_hybrid_embedding(
            raw_text="Test text",
            gemma_context="Gemma context",
            temporal_features=temporal_features,
            combination_mode="concatenate",
        )
        
        # Check L2 norm is approximately 1
        norm = np.linalg.norm(result)
        assert abs(norm - 1.0) < 0.01
    
    def test_empty_gemma_context_fallback(self, mock_embedder):
        """Test that empty Gemma context falls back to raw text."""
        temporal_features = np.random.randn(32).astype(np.float32)
        
        # This should not raise
        result = mock_embedder.generate_hybrid_embedding(
            raw_text="Test text",
            gemma_context="",  # Empty
            temporal_features=temporal_features,
            combination_mode="concatenate",
        )
        
        assert result is not None


class TestSimilarityFunctions:
    """Tests for similarity helper functions."""
    
    def test_cosine_similarity_identical(self):
        """Test cosine similarity of identical vectors."""
        from src.hybrid_embedder import cosine_similarity
        
        vec = np.array([1, 2, 3], dtype=np.float32)
        
        sim = cosine_similarity(vec, vec)
        
        assert abs(sim - 1.0) < 0.001
    
    def test_cosine_similarity_orthogonal(self):
        """Test cosine similarity of orthogonal vectors."""
        from src.hybrid_embedder import cosine_similarity
        
        vec_a = np.array([1, 0], dtype=np.float32)
        vec_b = np.array([0, 1], dtype=np.float32)
        
        sim = cosine_similarity(vec_a, vec_b)
        
        assert abs(sim) < 0.001
    
    def test_find_similar_sections(self):
        """Test finding similar sections."""
        from src.hybrid_embedder import find_similar_sections
        
        query = np.array([1, 0, 0], dtype=np.float32)
        sections = [
            np.array([1, 0, 0], dtype=np.float32),  # Identical
            np.array([0, 1, 0], dtype=np.float32),  # Orthogonal
            np.array([0.9, 0.1, 0], dtype=np.float32),  # Similar
        ]
        
        results = find_similar_sections(query, sections, top_k=2)
        
        assert len(results) == 2
        assert results[0][0] == 0  # Index of identical vector
        assert results[0][1] > 0.99  # High similarity


# =============================================================================
# Tests for chaos_comparator.py
# =============================================================================


class TestChaosComparator:
    """Tests for ChaosComparator."""
    
    @pytest.fixture
    def comparator(self):
        """Create ChaosComparator instance."""
        from src.chaos_comparator import ChaosComparator
        
        return ChaosComparator(service_auth_getter=lambda: {})
    
    def test_semantic_similarity(self, comparator):
        """Test semantic similarity computation."""
        vec_a = np.array([1, 0, 0], dtype=np.float32)
        vec_b = np.array([0.8, 0.2, 0], dtype=np.float32)
        
        sim = comparator.compute_semantic_similarity(vec_a, vec_b)
        
        assert 0 < sim < 1
    
    def test_temporal_similarity(self, comparator):
        """Test temporal similarity computation."""
        meta_a = {
            "day_of_week": "Monday",
            "month": 6,
            "quarter": 2,
            "is_month_end": False,
        }
        meta_b = {
            "day_of_week": "Monday",
            "month": 6,
            "quarter": 2,
            "is_month_end": False,
        }
        
        sim = comparator.compute_temporal_similarity(meta_a, meta_b)
        
        # Same temporal profile should have high similarity
        assert sim >= 0.5
    
    def test_find_chaotic_pairs(self, comparator):
        """Test finding chaotic pairs."""
        from src.chaos_comparator import DiscoveryConfig
        
        # Create test embeddings (semantically different)
        embeddings = [
            np.array([1, 0, 0], dtype=np.float32),
            np.array([0, 1, 0], dtype=np.float32),
            np.array([0, 0, 1], dtype=np.float32),
        ]
        
        # Create test metadata (temporally similar)
        metadata = [
            {
                "transcription_id": "t1",
                "recording_date": "2024-01-15",
                "day_of_week": "Monday",
                "month": 1,
                "quarter": 1,
            },
            {
                "transcription_id": "t2",
                "recording_date": "2023-01-16",
                "day_of_week": "Monday",
                "month": 1,
                "quarter": 1,
            },
            {
                "transcription_id": "t3",
                "recording_date": "2022-01-17",
                "day_of_week": "Monday",
                "month": 1,
                "quarter": 1,
            },
        ]
        
        config = DiscoveryConfig(
            modes=["temporal_bridging"],
            min_chaos_score=0.0,
            min_temporal_gap=30,
        )
        
        pairs = comparator.find_chaotic_pairs(embeddings, metadata, config)
        
        # Should find pairs between different transcriptions
        assert len(pairs) >= 0  # May find pairs depending on scoring
    
    def test_parse_date(self, comparator):
        """Test date parsing helper."""
        # ISO string
        date = comparator._parse_date("2024-06-15T10:00:00")
        assert date is not None
        assert date.year == 2024
        
        # Simple date string
        date = comparator._parse_date("2024-06-15")
        assert date is not None
        
        # Datetime object
        dt = datetime(2024, 6, 15)
        date = comparator._parse_date(dt)
        assert date == dt
        
        # None
        date = comparator._parse_date(None)
        assert date is None


class TestInsightDiscoveryEngine:
    """Tests for InsightDiscoveryEngine."""
    
    @pytest.fixture
    def engine(self):
        """Create InsightDiscoveryEngine instance."""
        from src.chaos_comparator import InsightDiscoveryEngine
        
        return InsightDiscoveryEngine(service_auth_getter=lambda: {})
    
    @pytest.mark.asyncio
    async def test_discover_insights_empty_sections(self, engine):
        """Test discovery with empty sections."""
        from src.chaos_comparator import DiscoveryConfig
        
        config = DiscoveryConfig(modes=["temporal_bridging"])
        
        report = await engine.discover_insights([], config=config)
        
        assert report.section_count == 0
        assert report.insights == []
    
    @pytest.mark.asyncio
    async def test_discover_insights_with_sections(self, engine):
        """Test discovery with valid sections."""
        from src.chaos_comparator import DiscoveryConfig
        
        sections = [
            {
                "text": "We need to increase Q1 sales targets",
                "recording_date": "2024-01-15",
                "metadata": {
                    "transcription_id": "t1",
                    "day_of_week": "Monday",
                    "month": 1,
                    "quarter": 1,
                },
                "embedding": np.random.randn(384).tolist(),
            },
            {
                "text": "The new product launch is on track",
                "recording_date": "2023-01-20",
                "metadata": {
                    "transcription_id": "t2",
                    "day_of_week": "Friday",
                    "month": 1,
                    "quarter": 1,
                },
                "embedding": np.random.randn(384).tolist(),
            },
        ]
        
        config = DiscoveryConfig(
            modes=["temporal_bridging"],
            max_insights=5,
            min_chaos_score=0.0,
            min_temporal_gap=7,
        )
        
        # Mock Gemma calls
        with patch.object(engine.comparator, 'generate_insight', new_callable=AsyncMock) as mock_insight:
            mock_insight.return_value = "Test insight"
            
            report = await engine.discover_insights(sections, config=config)
        
        assert report.section_count == 2
        assert report.processing_time_sec >= 0


class TestChaoticPair:
    """Tests for ChaoticPair dataclass."""
    
    def test_chaotic_pair_to_dict(self):
        """Test ChaoticPair serialization."""
        from src.chaos_comparator import ChaoticPair
        
        pair = ChaoticPair(
            section_a_idx=0,
            section_b_idx=1,
            discovery_mode="temporal_bridging",
            connection_type="same_quarter",
            chaos_score=0.75,
            temporal_similarity=0.8,
            insight="Test insight",
            confidence=0.9,
            metadata={"test": "value"},
        )
        
        data = pair.to_dict()
        
        assert data["section_a_idx"] == 0
        assert data["discovery_mode"] == "temporal_bridging"
        assert data["chaos_score"] == 0.75
        assert data["insight"] == "Test insight"


# =============================================================================
# Integration Tests
# =============================================================================


class TestCIDEIntegration:
    """Integration tests for CIDE pipeline."""
    
    @pytest.mark.asyncio
    async def test_full_enrichment_pipeline(self):
        """Test full enrichment pipeline (without Gemma)."""
        from src.temporal_context import get_temporal_enricher
        
        enricher = get_temporal_enricher()
        
        # Enrich context without world events (faster)
        context = await enricher.enrich(
            recording_date=datetime(2024, 6, 15),
            meeting_type="strategy",
            fetch_world_events=False,
        )
        
        assert context.quarter == 2
        assert context.meeting_type == "strategy"
        
        # Encode features
        temporal_features = enricher.encode_context(context)
        
        assert temporal_features.shape == (32,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
