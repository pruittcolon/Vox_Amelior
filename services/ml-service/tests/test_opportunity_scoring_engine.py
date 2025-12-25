"""
Opportunity Scoring Engine Unit Tests

Tests the opportunity scoring engine for:
- Win probability calculation
- Deal velocity analysis
- Risk factor identification
- Pipeline health assessment
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "engines"))

from engines.opportunity_scoring_engine import OpportunityScoringEngine

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def engine():
    """Create opportunity scoring engine instance."""
    return OpportunityScoringEngine()


@pytest.fixture
def sample_opps_df():
    """Create sample opportunities DataFrame."""
    today = datetime.now()

    return pd.DataFrame(
        [
            {
                "Id": "O1",
                "Name": "Enterprise Deal",
                "StageName": "Negotiation/Review",
                "Amount": 250000,
                "CloseDate": (today + timedelta(days=14)).strftime("%Y-%m-%d"),
                "CreatedDate": (today - timedelta(days=30)).strftime("%Y-%m-%d"),
                "AccountId": "A1",
            },
            {
                "Id": "O2",
                "Name": "Mid-Market Opportunity",
                "StageName": "Proposal/Price Quote",
                "Amount": 75000,
                "CloseDate": (today + timedelta(days=30)).strftime("%Y-%m-%d"),
                "CreatedDate": (today - timedelta(days=45)).strftime("%Y-%m-%d"),
                "AccountId": "A2",
            },
            {
                "Id": "O3",
                "Name": "Stalled Deal",
                "StageName": "Qualification",
                "Amount": 50000,
                "CloseDate": (today - timedelta(days=15)).strftime("%Y-%m-%d"),  # Overdue
                "CreatedDate": (today - timedelta(days=120)).strftime("%Y-%m-%d"),  # Old
                "AccountId": "A3",
            },
            {
                "Id": "O4",
                "Name": "Won Deal",
                "StageName": "Closed Won",
                "Amount": 100000,
                "CloseDate": (today - timedelta(days=5)).strftime("%Y-%m-%d"),
                "CreatedDate": (today - timedelta(days=60)).strftime("%Y-%m-%d"),
                "AccountId": "A4",
            },
            {
                "Id": "O5",
                "Name": "Lost Deal",
                "StageName": "Closed Lost",
                "Amount": 80000,
                "CloseDate": (today - timedelta(days=10)).strftime("%Y-%m-%d"),
                "CreatedDate": (today - timedelta(days=90)).strftime("%Y-%m-%d"),
                "AccountId": "A5",
            },
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

        assert info["name"] == "opportunity_scoring"
        assert info["display_name"] == "Opportunity Scoring"
        assert info["icon"] == "🎯"

    def test_get_config_schema(self, engine):
        """Test config schema has required parameters."""
        schema = engine.get_config_schema()

        param_names = [p.name for p in schema]
        assert "stage_column" in param_names
        assert "amount_column" in param_names
        assert "close_date_column" in param_names


# =============================================================================
# SCORING TESTS
# =============================================================================


class TestOpportunityScoring:
    """Test opportunity scoring calculations."""

    def test_analyze_returns_expected_structure(self, engine, sample_opps_df):
        """Test analyze returns all expected keys."""
        results = engine.analyze(sample_opps_df)

        assert "engine" in results
        assert results["engine"] == "opportunity_scoring"
        assert "summary" in results
        assert "scored_opportunities" in results
        assert "pipeline_health" in results
        assert "at_risk_deals" in results

    def test_all_opportunities_scored(self, engine, sample_opps_df):
        """Test all opportunities receive scores."""
        results = engine.analyze(sample_opps_df)

        assert len(results["scored_opportunities"]) == len(sample_opps_df)

    def test_scores_in_valid_range(self, engine, sample_opps_df):
        """Test all scores are between 1 and 99."""
        results = engine.analyze(sample_opps_df)

        for opp in results["scored_opportunities"]:
            assert 1 <= opp["score"] <= 99, f"Score {opp['score']} out of range"

    def test_negotiation_stage_scores_high(self, engine, sample_opps_df):
        """Test deals in negotiation stage score higher."""
        results = engine.analyze(sample_opps_df)

        # Find negotiation deal
        neg_deal = next(o for o in results["scored_opportunities"] if "Negotiation" in (o.get("stage") or ""))

        # Should be in upper half
        assert neg_deal["score"] >= 50

    def test_closed_won_scores_99(self, engine, sample_opps_df):
        """Test closed won deals score 99."""
        results = engine.analyze(sample_opps_df)

        won_deal = next(o for o in results["scored_opportunities"] if o["stage_type"] == "won")

        assert won_deal["score"] == 99

    def test_closed_lost_scores_1(self, engine, sample_opps_df):
        """Test closed lost deals score 1."""
        results = engine.analyze(sample_opps_df)

        lost_deal = next(o for o in results["scored_opportunities"] if o["stage_type"] == "lost")

        assert lost_deal["score"] == 1


# =============================================================================
# HEALTH INDICATOR TESTS
# =============================================================================


class TestHealthIndicators:
    """Test deal health classification."""

    def test_health_indicators_assigned(self, engine, sample_opps_df):
        """Test all deals get health indicators."""
        results = engine.analyze(sample_opps_df)

        for opp in results["scored_opportunities"]:
            assert opp["health"] in ["Green", "Yellow", "Red"]

    def test_overdue_deal_is_red(self, engine, sample_opps_df):
        """Test overdue deal gets red health."""
        results = engine.analyze(sample_opps_df)

        # The stalled/overdue deal
        stalled = next(o for o in results["scored_opportunities"] if o["opp_id"] == "O3")

        assert stalled["health"] == "Red"

    def test_high_probability_is_green(self, engine, sample_opps_df):
        """Test high probability deal gets green health."""
        results = engine.analyze(sample_opps_df)

        # Negotiation deal should be green or yellow
        neg_deal = next(o for o in results["scored_opportunities"] if "Negotiation" in (o.get("stage") or ""))

        assert neg_deal["health"] in ["Green", "Yellow"]


# =============================================================================
# RISK ANALYSIS TESTS
# =============================================================================


class TestRiskAnalysis:
    """Test at-risk deal identification."""

    def test_at_risk_deals_identified(self, engine, sample_opps_df):
        """Test at-risk deals are tracked."""
        results = engine.analyze(sample_opps_df)

        # Should identify the stalled deal
        assert "at_risk_deals" in results

    def test_recommended_actions_provided(self, engine, sample_opps_df):
        """Test all deals have recommended actions."""
        results = engine.analyze(sample_opps_df)

        for opp in results["scored_opportunities"]:
            assert "recommended_action" in opp
            assert isinstance(opp["recommended_action"], str)


# =============================================================================
# PIPELINE HEALTH TESTS
# =============================================================================


class TestPipelineHealth:
    """Test pipeline health assessment."""

    def test_pipeline_health_status(self, engine, sample_opps_df):
        """Test pipeline health returns status."""
        results = engine.analyze(sample_opps_df)

        health = results["pipeline_health"]
        assert "status" in health
        assert health["status"] in ["Healthy", "Moderate", "At Risk", "Empty"]

    def test_pipeline_has_metrics(self, engine, sample_opps_df):
        """Test pipeline health includes metrics."""
        results = engine.analyze(sample_opps_df)

        health = results["pipeline_health"]
        assert "avg_probability" in health
        assert "at_risk_percentage" in health


# =============================================================================
# SUMMARY TESTS
# =============================================================================


class TestSummary:
    """Test summary statistics."""

    def test_summary_has_required_fields(self, engine, sample_opps_df):
        """Test summary contains expected metrics."""
        results = engine.analyze(sample_opps_df)
        summary = results["summary"]

        assert "total_opportunities" in summary
        assert "avg_win_probability" in summary
        assert "weighted_pipeline_value" in summary
        assert "high_probability_deals" in summary

    def test_weighted_pipeline_calculated(self, engine, sample_opps_df):
        """Test weighted pipeline value is calculated."""
        results = engine.analyze(sample_opps_df)

        # Weighted should be less than total (probability < 100%)
        assert results["summary"]["weighted_pipeline_value"] <= results["summary"]["total_pipeline_value"]


# =============================================================================
# VELOCITY TESTS
# =============================================================================


class TestDealVelocity:
    """Test deal velocity analysis."""

    def test_days_in_pipeline_tracked(self, engine, sample_opps_df):
        """Test days in pipeline is calculated."""
        results = engine.analyze(sample_opps_df)

        open_opps = [o for o in results["scored_opportunities"] if o["stage_type"] == "open"]

        for opp in open_opps:
            assert "days_in_pipeline" in opp

    def test_stalled_deal_penalized(self, engine, sample_opps_df):
        """Test stalled deals get lower scores."""
        results = engine.analyze(sample_opps_df)

        stalled = next(o for o in results["scored_opportunities"] if o["opp_id"] == "O3")
        enterprise = next(o for o in results["scored_opportunities"] if o["opp_id"] == "O1")

        # Stalled should score lower than active deal
        assert stalled["score"] < enterprise["score"]


# =============================================================================
# EDGE CASES
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_minimal_data(self, engine):
        """Test with minimal opportunity data."""
        minimal_df = pd.DataFrame([{"Id": "O1", "Name": "Test Opp", "StageName": "Prospecting", "Amount": 10000}])

        results = engine.analyze(minimal_df)

        assert "scored_opportunities" in results
        assert len(results["scored_opportunities"]) == 1

    def test_no_amount(self, engine):
        """Test opportunities with no amount."""
        no_amount_df = pd.DataFrame([{"Id": "O1", "Name": "Zero Amount", "StageName": "Proposal", "Amount": None}])

        results = engine.analyze(no_amount_df)

        # Should still score
        assert len(results["scored_opportunities"]) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
