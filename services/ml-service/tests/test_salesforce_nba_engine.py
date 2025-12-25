"""
Salesforce NBA Engine Unit Tests

Tests Next Best Action recommendation engine for:
- Action generation
- Priority classification
- Action type coverage
- Context integration
"""

import sys
from pathlib import Path

import pytest

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "engines"))

from engines.salesforce_nba_engine import ActionType, Priority, SalesforceNBAEngine

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def engine():
    """Create NBA engine instance."""
    return SalesforceNBAEngine()


@pytest.fixture
def sample_lead_scores():
    """Sample lead scores for testing."""
    return [
        {"lead_id": "L1", "name": "John Smith", "company": "TechCorp", "score": 87, "segment": "Hot"},
        {"lead_id": "L2", "name": "Sarah Johnson", "company": "FinServ", "score": 75, "segment": "Warm"},
        {"lead_id": "L3", "name": "Mike Davis", "company": "SmallBiz", "score": 45, "segment": "Lukewarm"},
    ]


@pytest.fixture
def sample_opp_scores():
    """Sample opportunity scores for testing."""
    return [
        {
            "opp_id": "O1",
            "name": "Enterprise Deal",
            "score": 78,
            "amount": 250000,
            "health": "Green",
            "stage_type": "open",
            "recommended_action": "Push to close",
        },
        {
            "opp_id": "O2",
            "name": "Stalled Deal",
            "score": 35,
            "amount": 50000,
            "health": "Red",
            "stage_type": "open",
            "days_in_pipeline": 60,
            "recommended_action": "Re-engage",
        },
    ]


@pytest.fixture
def sample_churn_scores():
    """Sample churn/account scores for testing."""
    return [
        {"account_id": "A1", "name": "Healthy Corp", "health_score": 85, "revenue": 200000, "risk_level": "Low"},
        {
            "account_id": "A2",
            "name": "At Risk Inc",
            "health_score": 25,
            "revenue": 150000,
            "risk_level": "Critical",
            "recommended_actions": ["Schedule executive review"],
        },
    ]


# =============================================================================
# ENGINE INFO TESTS
# =============================================================================


class TestEngineInfo:
    """Test engine metadata."""

    def test_get_engine_info(self, engine):
        """Test engine info returns expected fields."""
        info = engine.get_engine_info()

        assert info["name"] == "salesforce_nba"
        assert info["display_name"] == "Next Best Action"


# =============================================================================
# ACTION GENERATION TESTS
# =============================================================================


class TestActionGeneration:
    """Test action generation from CRM data."""

    def test_analyze_with_lead_scores(self, engine, sample_lead_scores):
        """Test actions generated from lead scores."""
        results = engine.analyze(lead_scores=sample_lead_scores)

        assert "actions" in results
        assert len(results["actions"]) > 0

    def test_analyze_with_opp_scores(self, engine, sample_opp_scores):
        """Test actions generated from opportunity scores."""
        results = engine.analyze(opp_scores=sample_opp_scores)

        assert "actions" in results
        assert len(results["actions"]) > 0

    def test_analyze_with_churn_scores(self, engine, sample_churn_scores):
        """Test actions generated from churn scores."""
        results = engine.analyze(churn_scores=sample_churn_scores)

        assert "actions" in results
        assert len(results["actions"]) > 0

    def test_analyze_with_all_context(self, engine, sample_lead_scores, sample_opp_scores, sample_churn_scores):
        """Test actions generated from all sources."""
        results = engine.analyze(
            lead_scores=sample_lead_scores, opp_scores=sample_opp_scores, churn_scores=sample_churn_scores
        )

        assert "actions" in results
        # Should have more actions with more context
        assert len(results["actions"]) > 3

    def test_demo_actions_when_no_data(self, engine):
        """Test demo actions generated when no data provided."""
        results = engine.analyze()

        assert "actions" in results
        assert len(results["actions"]) > 0


# =============================================================================
# ACTION STRUCTURE TESTS
# =============================================================================


class TestActionStructure:
    """Test action structure and fields."""

    def test_action_has_required_fields(self, engine, sample_lead_scores):
        """Test actions have all required fields."""
        results = engine.analyze(lead_scores=sample_lead_scores)

        for action in results["actions"]:
            assert "action_type" in action
            assert "priority" in action
            assert "title" in action
            assert "description" in action
            assert "target_type" in action
            assert "confidence" in action

    def test_action_types_valid(self, engine, sample_lead_scores):
        """Test action types are valid."""
        results = engine.analyze(lead_scores=sample_lead_scores)

        valid_types = {t.value for t in ActionType}
        for action in results["actions"]:
            assert action["action_type"] in valid_types

    def test_priorities_valid(self, engine, sample_lead_scores):
        """Test priorities are valid."""
        results = engine.analyze(lead_scores=sample_lead_scores)

        valid_priorities = {p.value for p in Priority}
        for action in results["actions"]:
            assert action["priority"] in valid_priorities

    def test_confidence_in_range(self, engine, sample_lead_scores):
        """Test confidence is 0-1."""
        results = engine.analyze(lead_scores=sample_lead_scores)

        for action in results["actions"]:
            assert 0 <= action["confidence"] <= 1


# =============================================================================
# PRIORITIZATION TESTS
# =============================================================================


class TestPrioritization:
    """Test action prioritization."""

    def test_actions_sorted_by_priority(self, engine, sample_lead_scores, sample_opp_scores, sample_churn_scores):
        """Test actions are sorted with urgent first."""
        results = engine.analyze(
            lead_scores=sample_lead_scores, opp_scores=sample_opp_scores, churn_scores=sample_churn_scores
        )

        actions = results["actions"]
        if len(actions) >= 2:
            priority_order = {"urgent": 0, "high": 1, "medium": 2, "low": 3}
            for i in range(len(actions) - 1):
                p1 = priority_order.get(actions[i]["priority"], 4)
                p2 = priority_order.get(actions[i + 1]["priority"], 4)
                # Allow same priority (sorted by confidence)
                assert p1 <= p2 or p1 == p2

    def test_max_actions_respected(self, engine, sample_lead_scores):
        """Test max_actions config is respected."""
        results = engine.analyze(lead_scores=sample_lead_scores, config={"max_actions": 2})

        assert len(results["actions"]) <= 2


# =============================================================================
# SUMMARY TESTS
# =============================================================================


class TestSummary:
    """Test summary generation."""

    def test_summary_has_counts(self, engine, sample_lead_scores):
        """Test summary has action counts."""
        results = engine.analyze(lead_scores=sample_lead_scores)

        assert "summary" in results
        assert "total_actions" in results["summary"]

    def test_insights_generated(self, engine, sample_lead_scores):
        """Test insights are generated."""
        results = engine.analyze(lead_scores=sample_lead_scores)

        assert "insights" in results
        assert len(results["insights"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
