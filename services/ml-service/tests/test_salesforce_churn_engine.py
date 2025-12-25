"""
Salesforce Churn Engine Unit Tests

Tests account health scoring and churn prediction for:
- Health score calculation
- Risk level classification
- Churn probability estimation
- Risk factor identification
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "engines"))

from engines.salesforce_churn_engine import SalesforceChurnEngine

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def engine():
    """Create churn engine instance."""
    return SalesforceChurnEngine()


@pytest.fixture
def sample_accounts_df():
    """Create sample accounts DataFrame."""
    today = datetime.now()

    return pd.DataFrame(
        [
            {
                "Id": "A1",
                "Name": "Healthy Corp",
                "Industry": "Technology",
                "AnnualRevenue": 500000,
                "CreatedDate": (today - timedelta(days=730)).strftime("%Y-%m-%d"),
                "LastActivityDate": (today - timedelta(days=3)).strftime("%Y-%m-%d"),
                "OpenCases": 0,
            },
            {
                "Id": "A2",
                "Name": "At Risk Inc",
                "Industry": "Retail",
                "AnnualRevenue": 150000,
                "CreatedDate": (today - timedelta(days=400)).strftime("%Y-%m-%d"),
                "LastActivityDate": (today - timedelta(days=95)).strftime("%Y-%m-%d"),
                "OpenCases": 6,
            },
            {
                "Id": "A3",
                "Name": "New Customer",
                "Industry": "Healthcare",
                "AnnualRevenue": 75000,
                "CreatedDate": (today - timedelta(days=45)).strftime("%Y-%m-%d"),
                "LastActivityDate": (today - timedelta(days=10)).strftime("%Y-%m-%d"),
                "OpenCases": 1,
            },
        ]
    )


# =============================================================================
# ENGINE INFO TESTS
# =============================================================================


class TestEngineInfo:
    """Test engine metadata."""

    def test_get_engine_info(self, engine):
        """Test engine info returns expected fields."""
        info = engine.get_engine_info()

        assert info["name"] == "salesforce_churn"
        assert info["display_name"] == "Churn Prediction"
        assert "icon" in info

    def test_get_config_schema(self, engine):
        """Test config schema has required parameters."""
        schema = engine.get_config_schema()

        param_names = [p.name for p in schema]
        assert "account_id_column" in param_names
        assert "revenue_column" in param_names


# =============================================================================
# SCORING TESTS
# =============================================================================


class TestChurnScoring:
    """Test churn/health scoring."""

    def test_analyze_returns_expected_structure(self, engine, sample_accounts_df):
        """Test analyze returns all expected keys."""
        results = engine.analyze(sample_accounts_df)

        assert "engine" in results
        assert results["engine"] == "salesforce_churn"
        assert "summary" in results
        assert "scored_accounts" in results
        assert "at_risk_accounts" in results
        assert "insights" in results

    def test_all_accounts_scored(self, engine, sample_accounts_df):
        """Test all accounts receive scores."""
        results = engine.analyze(sample_accounts_df)

        assert len(results["scored_accounts"]) == len(sample_accounts_df)

    def test_health_scores_in_valid_range(self, engine, sample_accounts_df):
        """Test health scores are 1-100."""
        results = engine.analyze(sample_accounts_df)

        for account in results["scored_accounts"]:
            assert 1 <= account["health_score"] <= 100

    def test_churn_probability_valid(self, engine, sample_accounts_df):
        """Test churn probability is 0-1."""
        results = engine.analyze(sample_accounts_df)

        for account in results["scored_accounts"]:
            assert 0 <= account["churn_probability"] <= 1


# =============================================================================
# RISK LEVEL TESTS
# =============================================================================


class TestRiskLevels:
    """Test risk level classification."""

    def test_risk_levels_assigned(self, engine, sample_accounts_df):
        """Test all accounts get risk levels."""
        results = engine.analyze(sample_accounts_df)

        valid_levels = {"Critical", "High", "Moderate", "Low"}
        for account in results["scored_accounts"]:
            assert account["risk_level"] in valid_levels

    def test_at_risk_accounts_identified(self, engine, sample_accounts_df):
        """Test at-risk accounts are tracked."""
        results = engine.analyze(sample_accounts_df)

        # Should identify the at-risk account
        assert "at_risk_accounts" in results

    def test_recommended_actions_provided(self, engine, sample_accounts_df):
        """Test all accounts have recommended actions."""
        results = engine.analyze(sample_accounts_df)

        for account in results["scored_accounts"]:
            assert "recommended_actions" in account
            assert len(account["recommended_actions"]) > 0


# =============================================================================
# SUMMARY TESTS
# =============================================================================


class TestSummary:
    """Test summary statistics."""

    def test_summary_has_required_fields(self, engine, sample_accounts_df):
        """Test summary contains expected metrics."""
        results = engine.analyze(sample_accounts_df)
        summary = results["summary"]

        assert "total_accounts" in summary
        assert "avg_health_score" in summary
        assert "at_risk_revenue" in summary

    def test_total_accounts_count(self, engine, sample_accounts_df):
        """Test total accounts count is accurate."""
        results = engine.analyze(sample_accounts_df)

        assert results["summary"]["total_accounts"] == len(sample_accounts_df)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
