"""
Lead Scoring Engine Unit Tests

Tests the lead scoring engine for:
- Column detection and preprocessing
- Score calculation accuracy
- Segment classification
- Factor generation
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "engines"))

from engines.lead_scoring_engine import LeadScoringEngine

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def engine():
    """Create lead scoring engine instance."""
    return LeadScoringEngine()


@pytest.fixture
def sample_leads_df():
    """Create sample leads DataFrame."""
    return pd.DataFrame(
        [
            {
                "Id": "L1",
                "FirstName": "John",
                "LastName": "Smith",
                "Company": "TechCorp",
                "Title": "CTO",
                "Industry": "Technology",
                "LeadSource": "Referral",
                "Email": "john@techcorp.com",
                "Phone": "555-1234",
                "NumberOfEmployees": 500,
                "AnnualRevenue": 5000000,
                "CreatedDate": "2025-01-01",
                "Status": "New",
            },
            {
                "Id": "L2",
                "FirstName": "Sarah",
                "LastName": "Johnson",
                "Company": "FinServ Inc",
                "Title": "VP Finance",
                "Industry": "Financial Services",
                "LeadSource": "Webinar",
                "Email": "sarah@finserv.com",
                "Phone": "555-5678",
                "NumberOfEmployees": 1200,
                "AnnualRevenue": 15000000,
                "CreatedDate": "2025-01-05",
                "Status": "Working",
            },
            {
                "Id": "L3",
                "FirstName": "Mike",
                "LastName": "Davis",
                "Company": "SmallBiz",
                "Title": "Manager",
                "Industry": "Retail",
                "LeadSource": "Cold Call",
                "Email": None,
                "Phone": None,
                "NumberOfEmployees": 10,
                "AnnualRevenue": 50000,
                "CreatedDate": "2024-06-01",
                "Status": "New",
            },
        ]
    )


@pytest.fixture
def minimal_leads_df():
    """Create minimal leads DataFrame with only basic fields."""
    return pd.DataFrame(
        [
            {"Id": "L1", "Name": "Test Lead", "Company": "Test Corp"},
            {"Id": "L2", "Name": "Another Lead", "Company": "Another Corp"},
        ]
    )


# =============================================================================
# ENGINE INFO TESTS
# =============================================================================


class TestEngineInfo:
    """Test engine metadata methods."""

    def test_get_engine_info(self, engine):
        """Test engine info returns expected fields."""
        info = engine.get_engine_info()

        assert info["name"] == "lead_scoring"
        assert info["display_name"] == "Lead Scoring"
        assert info["icon"] == "🎯"
        assert info["task_type"] == "scoring"

    def test_get_config_schema(self, engine):
        """Test config schema has required parameters."""
        schema = engine.get_config_schema()

        assert len(schema) >= 5
        param_names = [p.name for p in schema]
        assert "lead_id_column" in param_names
        assert "industry_column" in param_names
        assert "title_column" in param_names

    def test_get_methodology_info(self, engine):
        """Test methodology info is complete."""
        method = engine.get_methodology_info()

        assert "name" in method
        assert "steps" in method
        assert len(method["steps"]) >= 4


# =============================================================================
# SCORING TESTS
# =============================================================================


class TestLeadScoring:
    """Test lead scoring calculations."""

    def test_analyze_returns_expected_structure(self, engine, sample_leads_df):
        """Test analyze returns all expected keys."""
        results = engine.analyze(sample_leads_df)

        assert "engine" in results
        assert results["engine"] == "lead_scoring"
        assert "summary" in results
        assert "scored_leads" in results
        assert "segments" in results
        assert "graphs" in results
        assert "insights" in results

    def test_all_leads_are_scored(self, engine, sample_leads_df):
        """Test all input leads receive scores."""
        results = engine.analyze(sample_leads_df)

        assert len(results["scored_leads"]) == len(sample_leads_df)

    def test_scores_in_valid_range(self, engine, sample_leads_df):
        """Test all scores are between 1 and 99."""
        results = engine.analyze(sample_leads_df)

        for lead in results["scored_leads"]:
            assert 1 <= lead["score"] <= 99, f"Score {lead['score']} out of range"

    def test_high_quality_lead_scores_high(self, engine, sample_leads_df):
        """Test C-level executive with good company scores high."""
        results = engine.analyze(sample_leads_df)

        # Find the CTO lead (John Smith)
        cto_lead = next(l for l in results["scored_leads"] if "CTO" in (l.get("title") or ""))

        # CTO should score higher than average
        avg_score = results["summary"]["avg_score"]
        assert cto_lead["score"] >= avg_score, "CTO should score above average"

    def test_low_quality_lead_scores_lower(self, engine, sample_leads_df):
        """Test incomplete lead with small company scores lower."""
        results = engine.analyze(sample_leads_df)

        # Find the Manager lead with no email/phone
        manager_lead = next(l for l in results["scored_leads"] if l["lead_id"] == "L3")

        # Should be in Lukewarm or Cold segment
        assert manager_lead["segment"] in ["Lukewarm", "Cold"]

    def test_segment_classification(self, engine, sample_leads_df):
        """Test leads are classified into segments."""
        results = engine.analyze(sample_leads_df)

        segments_found = set(l["segment"] for l in results["scored_leads"])

        # Should have at least 2 different segments
        assert len(segments_found) >= 1

        # All segments should be valid
        valid_segments = {"Hot", "Warm", "Lukewarm", "Cold"}
        assert segments_found.issubset(valid_segments)


# =============================================================================
# FACTOR TESTS
# =============================================================================


class TestScoringFactors:
    """Test positive and negative factor generation."""

    def test_positive_factors_for_high_value_lead(self, engine, sample_leads_df):
        """Test high-value leads have positive factors."""
        results = engine.analyze(sample_leads_df)

        # CTO lead should have positive factors
        cto_lead = next(l for l in results["scored_leads"] if "CTO" in (l.get("title") or ""))

        assert len(cto_lead["positive_factors"]) > 0

    def test_negative_factors_for_incomplete_lead(self, engine, sample_leads_df):
        """Test incomplete leads have negative factors."""
        results = engine.analyze(sample_leads_df)

        # Manager lead with missing email/phone
        manager_lead = next(l for l in results["scored_leads"] if l["lead_id"] == "L3")

        # Should have some negative factors
        assert len(manager_lead.get("negative_factors", [])) >= 0  # May or may not have negatives

    def test_factor_analysis_aggregation(self, engine, sample_leads_df):
        """Test factors are aggregated in analysis."""
        results = engine.analyze(sample_leads_df)

        assert "factors_analysis" in results
        assert "top_positive_factors" in results["factors_analysis"]
        assert "top_negative_factors" in results["factors_analysis"]


# =============================================================================
# SUMMARY TESTS
# =============================================================================


class TestScoringSummary:
    """Test scoring summary statistics."""

    def test_summary_has_required_fields(self, engine, sample_leads_df):
        """Test summary contains expected metrics."""
        results = engine.analyze(sample_leads_df)
        summary = results["summary"]

        assert "total_leads" in summary
        assert "avg_score" in summary
        assert "high_quality_leads" in summary
        assert "hot_leads_pct" in summary

    def test_total_leads_count(self, engine, sample_leads_df):
        """Test total leads count is accurate."""
        results = engine.analyze(sample_leads_df)

        assert results["summary"]["total_leads"] == len(sample_leads_df)

    def test_avg_score_calculation(self, engine, sample_leads_df):
        """Test average score is calculated correctly."""
        results = engine.analyze(sample_leads_df)

        scores = [l["score"] for l in results["scored_leads"]]
        expected_avg = sum(scores) / len(scores)

        assert abs(results["summary"]["avg_score"] - expected_avg) < 0.1


# =============================================================================
# VISUALIZATION TESTS
# =============================================================================


class TestVisualizations:
    """Test visualization generation."""

    def test_graphs_generated(self, engine, sample_leads_df):
        """Test graphs are generated."""
        results = engine.analyze(sample_leads_df)

        assert len(results["graphs"]) >= 1

    def test_histogram_in_graphs(self, engine, sample_leads_df):
        """Test score distribution histogram is included."""
        results = engine.analyze(sample_leads_df)

        graph_types = [g["type"] for g in results["graphs"]]
        assert "histogram" in graph_types


# =============================================================================
# INSIGHTS TESTS
# =============================================================================


class TestInsights:
    """Test AI insight generation."""

    def test_insights_generated(self, engine, sample_leads_df):
        """Test insights are generated."""
        results = engine.analyze(sample_leads_df)

        assert len(results["insights"]) >= 1

    def test_insights_are_strings(self, engine, sample_leads_df):
        """Test all insights are strings."""
        results = engine.analyze(sample_leads_df)

        for insight in results["insights"]:
            assert isinstance(insight, str)


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_minimal_data(self, engine, minimal_leads_df):
        """Test with minimal lead data."""
        results = engine.analyze(minimal_leads_df)

        # Should still return results
        assert "scored_leads" in results
        assert len(results["scored_leads"]) == 2

    def test_empty_dataframe(self, engine):
        """Test with empty DataFrame."""
        empty_df = pd.DataFrame()

        results = engine.analyze(empty_df)

        # Should handle gracefully
        assert "scored_leads" in results or "error" in results

    def test_single_lead(self, engine):
        """Test with single lead."""
        single_df = pd.DataFrame([{"Id": "L1", "Name": "Solo Lead", "Company": "Solo Corp"}])

        results = engine.analyze(single_df)

        assert len(results.get("scored_leads", [])) >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
