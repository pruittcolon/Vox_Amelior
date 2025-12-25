"""
Salesforce Next Best Action Engine - Prescriptive AI Recommendations

Generates personalized, prioritized action recommendations:
- Cross-sell/upsell opportunities
- Engagement activities
- Risk mitigation actions
- Relationship building suggestions
- Time-optimized scheduling

Uses context from leads, opportunities, and account health.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import pandas as pd
from core import (
    ColumnProfiler,
    EngineRequirements,
)
from core.premium_models import ConfigParameter

logger = logging.getLogger(__name__)


# =============================================================================
# ACTION TYPES AND PRIORITIES
# =============================================================================


class ActionType(Enum):
    """Types of recommended actions."""

    CALL = "call"
    EMAIL = "email"
    MEETING = "meeting"
    DEMO = "demo"
    PROPOSAL = "proposal"
    FOLLOW_UP = "follow_up"
    ESCALATION = "escalation"
    CROSS_SELL = "cross_sell"
    UPSELL = "upsell"
    RETENTION = "retention"
    RENEWAL = "renewal"


class Priority(Enum):
    """Action priority levels."""

    URGENT = "urgent"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class RecommendedAction:
    """A single recommended action."""

    action_type: ActionType
    priority: Priority
    title: str
    description: str
    target_id: str
    target_name: str
    target_type: str  # lead, opportunity, account
    expected_impact: str
    confidence: float
    due_date: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_type": self.action_type.value,
            "priority": self.priority.value,
            "title": self.title,
            "description": self.description,
            "target_id": self.target_id,
            "target_name": self.target_name,
            "target_type": self.target_type,
            "expected_impact": self.expected_impact,
            "confidence": self.confidence,
            "due_date": self.due_date,
        }


class SalesforceNBAEngine:
    """
    Next Best Action Engine for Salesforce.

    Analyzes CRM data and generates prioritized recommendations:
    - What action to take
    - Why it's recommended
    - Expected impact
    - Confidence score

    Integrates context from leads, opportunities, and accounts.
    """

    def __init__(self):
        self.name = "Salesforce Next Best Action Engine"
        self.profiler = ColumnProfiler()

    def get_engine_info(self) -> dict[str, str]:
        """Get engine display information."""
        return {
            "name": "salesforce_nba",
            "display_name": "Next Best Action",
            "icon": "🎯",
            "task_type": "recommendation",
            "description": "AI-powered action recommendations for sales reps",
        }

    def get_config_schema(self) -> list[ConfigParameter]:
        """Get configuration schema for UI."""
        return [
            ConfigParameter(
                name="user_id",
                type="text",
                default=None,
                range=[],
                description="User/rep ID to personalize recommendations",
            ),
            ConfigParameter(
                name="max_actions",
                type="number",
                default=10,
                range=[1, 50],
                description="Maximum number of actions to recommend",
            ),
            ConfigParameter(
                name="include_cross_sell",
                type="boolean",
                default=True,
                range=[],
                description="Include cross-sell recommendations",
            ),
        ]

    def get_methodology_info(self) -> dict[str, Any]:
        """Get methodology documentation."""
        return {
            "name": "Prescriptive Next Best Action",
            "url": "https://www.salesforce.com/products/einstein/features/next-best-action/",
            "steps": [
                {
                    "step_number": 1,
                    "title": "Context Gathering",
                    "description": "Collect lead, opportunity, and account context",
                },
                {
                    "step_number": 2,
                    "title": "Opportunity Identification",
                    "description": "Identify engagement, sales, and retention opportunities",
                },
                {
                    "step_number": 3,
                    "title": "Action Generation",
                    "description": "Create specific actionable recommendations",
                },
                {"step_number": 4, "title": "Prioritization", "description": "Rank by urgency, impact, and confidence"},
                {"step_number": 5, "title": "Personalization", "description": "Tailor to user's role and capacity"},
            ],
        }

    def get_requirements(self) -> EngineRequirements:
        """Get engine data requirements."""
        return EngineRequirements(
            required_semantics=[],
            optional_semantics={},
            required_entities=[],
            preferred_domains=["crm", "sales"],
            applicable_tasks=["recommendation", "next_best_action"],
            min_rows=0,
            min_numeric_cols=0,
        )

    def analyze(
        self,
        leads_df: pd.DataFrame | None = None,
        opportunities_df: pd.DataFrame | None = None,
        accounts_df: pd.DataFrame | None = None,
        lead_scores: list[dict] | None = None,
        opp_scores: list[dict] | None = None,
        churn_scores: list[dict] | None = None,
        config: dict | None = None,
    ) -> dict[str, Any]:
        """
        Generate next best actions from CRM context.

        Args:
            leads_df: Optional leads DataFrame
            opportunities_df: Optional opportunities DataFrame
            accounts_df: Optional accounts DataFrame
            lead_scores: Optional pre-computed lead scores
            opp_scores: Optional pre-computed opportunity scores
            churn_scores: Optional pre-computed churn/health scores
            config: Configuration options

        Returns:
            Dict with:
            - actions: Prioritized list of recommended actions
            - summary: Action counts by type and priority
            - insights: Key observations
        """
        config = config or {}
        max_actions = config.get("max_actions", 10)

        results = {"engine": "salesforce_nba", "actions": [], "summary": {}, "insights": [], "action_stats": {}}

        try:
            all_actions: list[RecommendedAction] = []

            # Generate actions from leads
            if leads_df is not None or lead_scores:
                lead_actions = self._generate_lead_actions(leads_df, lead_scores)
                all_actions.extend(lead_actions)

            # Generate actions from opportunities
            if opportunities_df is not None or opp_scores:
                opp_actions = self._generate_opportunity_actions(opportunities_df, opp_scores)
                all_actions.extend(opp_actions)

            # Generate actions from accounts/churn
            if accounts_df is not None or churn_scores:
                account_actions = self._generate_account_actions(accounts_df, churn_scores)
                all_actions.extend(account_actions)

            # If no data provided, generate demo actions
            if not all_actions:
                all_actions = self._generate_demo_actions()

            # Sort by priority and confidence
            sorted_actions = self._prioritize_actions(all_actions)

            # Limit to max actions
            top_actions = sorted_actions[:max_actions]

            results["actions"] = [a.to_dict() for a in top_actions]

            # Build summary
            results["summary"] = self._build_summary(top_actions)

            # Generate insights
            results["insights"] = self._generate_insights(top_actions)

            # Action statistics
            results["action_stats"] = {
                "total_generated": len(all_actions),
                "returned": len(top_actions),
                "urgent_count": len([a for a in top_actions if a.priority == Priority.URGENT]),
                "high_count": len([a for a in top_actions if a.priority == Priority.HIGH]),
            }

        except Exception as e:
            logger.error(f"NBA engine failed: {e}")
            results["error"] = str(e)

        return results

    def _generate_lead_actions(
        self, leads_df: pd.DataFrame | None, lead_scores: list[dict] | None
    ) -> list[RecommendedAction]:
        """Generate actions for leads."""
        actions = []

        if lead_scores:
            # Hot leads - immediate follow up
            hot_leads = [l for l in lead_scores if l.get("segment") == "Hot"]
            for lead in hot_leads[:5]:
                actions.append(
                    RecommendedAction(
                        action_type=ActionType.CALL,
                        priority=Priority.URGENT,
                        title=f"Call hot lead: {lead.get('name', 'Unknown')}",
                        description=f"High-scoring lead (Score: {lead.get('score', 0)}) at {lead.get('company', 'N/A')}. Strike while hot!",
                        target_id=lead.get("lead_id", ""),
                        target_name=lead.get("name", "Unknown"),
                        target_type="lead",
                        expected_impact="High conversion potential",
                        confidence=0.85,
                        due_date=(datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"),
                    )
                )

            # Warm leads - schedule demo
            warm_leads = [l for l in lead_scores if l.get("segment") == "Warm"]
            for lead in warm_leads[:3]:
                actions.append(
                    RecommendedAction(
                        action_type=ActionType.DEMO,
                        priority=Priority.HIGH,
                        title=f"Schedule demo: {lead.get('name', 'Unknown')}",
                        description=f"Engaged lead ready for product demonstration. Company: {lead.get('company', 'N/A')}",
                        target_id=lead.get("lead_id", ""),
                        target_name=lead.get("name", "Unknown"),
                        target_type="lead",
                        expected_impact="Move to qualified opportunity",
                        confidence=0.72,
                        due_date=(datetime.now() + timedelta(days=3)).strftime("%Y-%m-%d"),
                    )
                )

        return actions

    def _generate_opportunity_actions(
        self, opps_df: pd.DataFrame | None, opp_scores: list[dict] | None
    ) -> list[RecommendedAction]:
        """Generate actions for opportunities."""
        actions = []

        if opp_scores:
            # High probability deals closing soon - push to close
            high_prob = [o for o in opp_scores if o.get("score", 0) >= 70 and o.get("stage_type") == "open"]
            for opp in high_prob[:3]:
                actions.append(
                    RecommendedAction(
                        action_type=ActionType.PROPOSAL,
                        priority=Priority.URGENT,
                        title=f"Close deal: {opp.get('name', 'Unknown')}",
                        description=f"High probability ({opp.get('score', 0)}%) deal worth ${opp.get('amount', 0):,.0f}. {opp.get('recommended_action', '')}",
                        target_id=opp.get("opp_id", ""),
                        target_name=opp.get("name", "Unknown"),
                        target_type="opportunity",
                        expected_impact=f"${opp.get('amount', 0):,.0f} revenue",
                        confidence=opp.get("score", 70) / 100,
                        due_date=(datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d"),
                    )
                )

            # At risk deals - intervention needed
            at_risk = [o for o in opp_scores if o.get("health") == "Red" and o.get("stage_type") == "open"]
            for opp in at_risk[:3]:
                actions.append(
                    RecommendedAction(
                        action_type=ActionType.MEETING,
                        priority=Priority.HIGH,
                        title=f"Rescue deal: {opp.get('name', 'Unknown')}",
                        description=f"At-risk deal needs attention. Current probability: {opp.get('score', 0)}%",
                        target_id=opp.get("opp_id", ""),
                        target_name=opp.get("name", "Unknown"),
                        target_type="opportunity",
                        expected_impact=f"Save ${opp.get('amount', 0):,.0f} pipeline",
                        confidence=0.65,
                        due_date=(datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d"),
                    )
                )

            # Stalled deals - re-engage
            stalled = [o for o in opp_scores if o.get("days_in_pipeline", 0) and o.get("days_in_pipeline") > 45]
            for opp in stalled[:2]:
                actions.append(
                    RecommendedAction(
                        action_type=ActionType.FOLLOW_UP,
                        priority=Priority.MEDIUM,
                        title=f"Re-engage: {opp.get('name', 'Unknown')}",
                        description=f"Deal stalled for {opp.get('days_in_pipeline', 0)} days. Time for fresh approach.",
                        target_id=opp.get("opp_id", ""),
                        target_name=opp.get("name", "Unknown"),
                        target_type="opportunity",
                        expected_impact="Revive stalled deal",
                        confidence=0.55,
                        due_date=(datetime.now() + timedelta(days=3)).strftime("%Y-%m-%d"),
                    )
                )

        return actions

    def _generate_account_actions(
        self, accounts_df: pd.DataFrame | None, churn_scores: list[dict] | None
    ) -> list[RecommendedAction]:
        """Generate actions for accounts based on churn risk."""
        actions = []

        if churn_scores:
            # Critical risk accounts - executive escalation
            critical = [a for a in churn_scores if a.get("risk_level") == "Critical"]
            for account in critical[:2]:
                actions.append(
                    RecommendedAction(
                        action_type=ActionType.ESCALATION,
                        priority=Priority.URGENT,
                        title=f"Executive escalation: {account.get('name', 'Unknown')}",
                        description=f"Critical churn risk (Health: {account.get('health_score', 0)}). Revenue at risk: ${account.get('revenue', 0):,.0f}",
                        target_id=account.get("account_id", ""),
                        target_name=account.get("name", "Unknown"),
                        target_type="account",
                        expected_impact="Prevent churn",
                        confidence=0.75,
                        due_date=(datetime.now()).strftime("%Y-%m-%d"),
                    )
                )

            # High risk - retention
            high_risk = [a for a in churn_scores if a.get("risk_level") == "High"]
            for account in high_risk[:3]:
                actions.append(
                    RecommendedAction(
                        action_type=ActionType.RETENTION,
                        priority=Priority.HIGH,
                        title=f"Retention outreach: {account.get('name', 'Unknown')}",
                        description=f"High churn risk. Recommended: {account.get('recommended_actions', ['Check in'])[0]}",
                        target_id=account.get("account_id", ""),
                        target_name=account.get("name", "Unknown"),
                        target_type="account",
                        expected_impact=f"Save ${account.get('revenue', 0):,.0f} ARR",
                        confidence=0.70,
                        due_date=(datetime.now() + timedelta(days=3)).strftime("%Y-%m-%d"),
                    )
                )

            # Healthy accounts - expansion
            healthy = [a for a in churn_scores if a.get("risk_level") == "Low" and a.get("revenue", 0) > 50000]
            for account in healthy[:2]:
                actions.append(
                    RecommendedAction(
                        action_type=ActionType.UPSELL,
                        priority=Priority.MEDIUM,
                        title=f"Expansion opportunity: {account.get('name', 'Unknown')}",
                        description=f"Healthy account with growth potential. Current value: ${account.get('revenue', 0):,.0f}",
                        target_id=account.get("account_id", ""),
                        target_name=account.get("name", "Unknown"),
                        target_type="account",
                        expected_impact="20-30% revenue expansion",
                        confidence=0.65,
                        due_date=(datetime.now() + timedelta(days=14)).strftime("%Y-%m-%d"),
                    )
                )

        return actions

    def _generate_demo_actions(self) -> list[RecommendedAction]:
        """Generate demo actions when no real data is available."""
        return [
            RecommendedAction(
                action_type=ActionType.CALL,
                priority=Priority.URGENT,
                title="Call hot lead: John Smith (CTO, TechCorp)",
                description="High-scoring lead (87) with executive decision-maker title. Referral source indicates high intent.",
                target_id="L1",
                target_name="John Smith",
                target_type="lead",
                expected_impact="High conversion potential",
                confidence=0.87,
                due_date=(datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"),
            ),
            RecommendedAction(
                action_type=ActionType.PROPOSAL,
                priority=Priority.URGENT,
                title="Close deal: Enterprise Generator Package",
                description="75% win probability deal worth $250,000 in negotiation stage. Push for close.",
                target_id="O1",
                target_name="Enterprise Generator Package",
                target_type="opportunity",
                expected_impact="$250,000 revenue",
                confidence=0.75,
                due_date=(datetime.now() + timedelta(days=3)).strftime("%Y-%m-%d"),
            ),
            RecommendedAction(
                action_type=ActionType.MEETING,
                priority=Priority.HIGH,
                title="Rescue stalled deal: Mid-Market Solution",
                description="Deal has been in qualification for 60 days. Schedule discovery call to re-engage.",
                target_id="O3",
                target_name="Mid-Market Solution",
                target_type="opportunity",
                expected_impact="Save $75,000 pipeline",
                confidence=0.60,
                due_date=(datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d"),
            ),
            RecommendedAction(
                action_type=ActionType.RETENTION,
                priority=Priority.HIGH,
                title="Retention outreach: HealthFirst Inc",
                description="Account health declining (42/100). No activity in 45 days. Proactive outreach needed.",
                target_id="A5",
                target_name="HealthFirst Inc",
                target_type="account",
                expected_impact="Prevent $150,000 churn",
                confidence=0.68,
                due_date=(datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"),
            ),
            RecommendedAction(
                action_type=ActionType.UPSELL,
                priority=Priority.MEDIUM,
                title="Expansion: United Oil & Gas",
                description="Healthy enterprise account ready for expansion. Recommend maintenance contract add-on.",
                target_id="A6",
                target_name="United Oil & Gas",
                target_type="account",
                expected_impact="25% revenue expansion",
                confidence=0.72,
                due_date=(datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d"),
            ),
        ]

    def _prioritize_actions(self, actions: list[RecommendedAction]) -> list[RecommendedAction]:
        """Sort actions by priority and confidence."""
        priority_order = {Priority.URGENT: 0, Priority.HIGH: 1, Priority.MEDIUM: 2, Priority.LOW: 3}

        return sorted(actions, key=lambda a: (priority_order[a.priority], -a.confidence))

    def _build_summary(self, actions: list[RecommendedAction]) -> dict[str, Any]:
        """Build action summary statistics."""
        by_type = {}
        by_priority = {}
        by_target = {}

        for action in actions:
            # By type
            type_name = action.action_type.value
            by_type[type_name] = by_type.get(type_name, 0) + 1

            # By priority
            pri_name = action.priority.value
            by_priority[pri_name] = by_priority.get(pri_name, 0) + 1

            # By target type
            tgt = action.target_type
            by_target[tgt] = by_target.get(tgt, 0) + 1

        return {"total_actions": len(actions), "by_type": by_type, "by_priority": by_priority, "by_target": by_target}

    def _generate_insights(self, actions: list[RecommendedAction]) -> list[str]:
        """Generate insights from recommended actions."""
        insights = []

        urgent = [a for a in actions if a.priority == Priority.URGENT]
        if urgent:
            insights.append(f"🚨 {len(urgent)} urgent actions require immediate attention")

        high_value = [a for a in actions if "$" in a.expected_impact]
        if high_value:
            insights.append("💰 High-value actions identified across your pipeline")

        leads = [a for a in actions if a.target_type == "lead"]
        opps = [a for a in actions if a.target_type == "opportunity"]
        accounts = [a for a in actions if a.target_type == "account"]

        if leads:
            insights.append(f"📞 {len(leads)} lead engagement actions recommended")
        if opps:
            insights.append(f"📈 {len(opps)} opportunity advancement actions")
        if accounts:
            insights.append(f"🤝 {len(accounts)} account relationship actions")

        return insights


__all__ = ["SalesforceNBAEngine", "ActionType", "Priority", "RecommendedAction"]
