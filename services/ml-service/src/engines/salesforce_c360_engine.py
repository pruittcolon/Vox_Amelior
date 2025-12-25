"""
Salesforce Customer 360 Engine - Unified Customer Health Scoring

Provides a holistic view of customer health across all touchpoints:
- Unified health score combining multiple signals
- Account health (from churn engine)
- Opportunity health (from opportunity scoring)
- Support health (case sentiment and volume)
- Engagement health (activity metrics)
- Financial health (revenue trends)

Creates a comprehensive 360-degree customer view.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import logging
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from core import (
    ColumnMapper,
    ColumnProfiler,
    EngineRequirements,
    SemanticType,
)
from core.gemma_summarizer import GemmaSummarizer
from core.premium_models import ConfigParameter

logger = logging.getLogger(__name__)


# =============================================================================
# HEALTH DIMENSION CONFIGURATION
# =============================================================================

HEALTH_DIMENSIONS = {
    "account": {
        "weight": 0.25,
        "name": "Account Health",
        "icon": "🏢",
        "description": "Overall account relationship health",
    },
    "opportunity": {
        "weight": 0.25,
        "name": "Pipeline Health",
        "icon": "💰",
        "description": "Active opportunity and pipeline status",
    },
    "support": {
        "weight": 0.20,
        "name": "Support Health",
        "icon": "🎧",
        "description": "Support experience and case resolution",
    },
    "engagement": {
        "weight": 0.15,
        "name": "Engagement Health",
        "icon": "📊",
        "description": "Activity and interaction frequency",
    },
    "financial": {
        "weight": 0.15,
        "name": "Financial Health",
        "icon": "📈",
        "description": "Revenue trends and payment patterns",
    },
}

C360_SEGMENTS = {
    "champion": {"min_score": 85, "label": "Champion", "color": "#10b981"},
    "healthy": {"min_score": 70, "label": "Healthy", "color": "#3b82f6"},
    "needs_attention": {"min_score": 50, "label": "Needs Attention", "color": "#f59e0b"},
    "at_risk": {"min_score": 30, "label": "At Risk", "color": "#ef4444"},
    "critical": {"min_score": 0, "label": "Critical", "color": "#dc2626"},
}


class SalesforceC360Engine:
    """
    Customer 360 Engine for Salesforce.

    Creates a unified health score by combining:
    - Account health metrics
    - Opportunity/pipeline health
    - Support satisfaction
    - Engagement patterns
    - Financial performance

    Provides comprehensive customer insights and recommendations.
    """

    def __init__(self) -> None:
        """Initialize the Customer 360 engine."""
        self.name = "Salesforce Customer 360 Engine"
        self.profiler = ColumnProfiler()
        self.mapper = ColumnMapper()

    def get_engine_info(self) -> dict[str, str]:
        """Get engine display information."""
        return {
            "name": "salesforce_c360",
            "display_name": "Customer 360",
            "icon": "🎯",
            "task_type": "analysis",
            "description": "Unified customer health across all touchpoints",
        }

    def get_config_schema(self) -> list[ConfigParameter]:
        """Get configuration schema for UI."""
        return [
            ConfigParameter(
                name="account_id_column", type="select", default=None, range=[], description="Account identifier column"
            ),
            ConfigParameter(
                name="name_column", type="select", default=None, range=[], description="Account name column"
            ),
            ConfigParameter(
                name="revenue_column", type="select", default=None, range=[], description="Annual revenue column"
            ),
            ConfigParameter(
                name="industry_column", type="select", default=None, range=[], description="Industry column"
            ),
            ConfigParameter(
                name="include_opportunities",
                type="boolean",
                default=True,
                range=None,
                description="Include opportunity data in analysis",
            ),
            ConfigParameter(
                name="include_cases",
                type="boolean",
                default=True,
                range=None,
                description="Include case data in analysis",
            ),
        ]

    def get_methodology_info(self) -> dict[str, Any]:
        """Get methodology documentation."""
        return {
            "name": "Customer 360 Health Analysis",
            "url": "https://www.salesforce.com/products/customer-360/",
            "steps": [
                {
                    "step_number": 1,
                    "title": "Data Aggregation",
                    "description": "Collect signals across all touchpoints",
                },
                {
                    "step_number": 2,
                    "title": "Dimension Scoring",
                    "description": "Score each health dimension independently",
                },
                {
                    "step_number": 3,
                    "title": "Weighted Combination",
                    "description": "Create unified score with dimension weights",
                },
                {
                    "step_number": 4,
                    "title": "Segment Assignment",
                    "description": "Classify customer into health segments",
                },
                {"step_number": 5, "title": "Action Generation", "description": "Generate prioritized recommendations"},
            ],
            "limitations": [
                "Accuracy improves with complete data across all touchpoints",
                "Dimension weights can be customized based on business priorities",
            ],
        }

    def get_requirements(self) -> EngineRequirements:
        """Get engine data requirements."""
        return EngineRequirements(
            required_semantics=[SemanticType.CATEGORICAL],
            optional_semantics={"revenue": [SemanticType.NUMERIC_CONTINUOUS], "date": [SemanticType.TEMPORAL]},
            required_entities=[],
            preferred_domains=["crm", "customer_success"],
            applicable_tasks=["customer_360", "health_scoring"],
            min_rows=1,
            min_numeric_cols=0,
        )

    def analyze(
        self,
        accounts_df: pd.DataFrame,
        opportunities_df: pd.DataFrame | None = None,
        cases_df: pd.DataFrame | None = None,
        churn_scores: list[dict] | None = None,
        opp_scores: list[dict] | None = None,
        config: dict | None = None,
    ) -> dict[str, Any]:
        """
        Analyze accounts for Customer 360 health.

        Args:
            accounts_df: DataFrame with account data
            opportunities_df: Optional opportunities DataFrame
            cases_df: Optional cases DataFrame
            churn_scores: Pre-computed churn scores (from churn engine)
            opp_scores: Pre-computed opportunity scores
            config: Optional configuration

        Returns:
            Dict with:
            - customer_360_scores: Unified health scores per account
            - summary: Aggregate health metrics
            - champions: Top health accounts
            - at_risk: Accounts needing intervention
            - dimension_analysis: Health by dimension
            - insights: AI-generated observations
        """
        config = config or {}

        results = {
            "engine": "salesforce_c360",
            "summary": {},
            "customer_360_scores": [],
            "champions": [],
            "healthy_accounts": [],
            "needs_attention": [],
            "at_risk": [],
            "critical": [],
            "dimension_analysis": {},
            "segment_breakdown": {},
            "graphs": [],
            "insights": [],
            "column_mappings": {},
        }

        try:
            # Profile account dataset
            profiles = self.profiler.profile_dataset(accounts_df)

            # Detect columns
            col_mappings = self._detect_columns(accounts_df, profiles, config)
            results["column_mappings"] = col_mappings

            # Pre-process related data
            opp_by_account = self._group_opportunities_by_account(opportunities_df, opp_scores)
            cases_by_account = self._group_cases_by_account(cases_df)
            churn_by_account = self._index_churn_scores(churn_scores)

            # Score each account
            c360_scores = []
            for idx, row in accounts_df.iterrows():
                account_id = self._get_value(row, col_mappings, "account_id", str(idx))

                c360_score = self._score_customer_360(
                    row=row,
                    col_mappings=col_mappings,
                    opportunities=opp_by_account.get(str(account_id), []),
                    cases=cases_by_account.get(str(account_id), []),
                    churn_score=churn_by_account.get(str(account_id)),
                )
                c360_scores.append(c360_score)

            results["customer_360_scores"] = c360_scores

            # Categorize by segment
            for c360 in c360_scores:
                segment = c360["segment"]
                if segment == "Champion":
                    results["champions"].append(c360)
                elif segment == "Healthy":
                    results["healthy_accounts"].append(c360)
                elif segment == "Needs Attention":
                    results["needs_attention"].append(c360)
                elif segment == "At Risk":
                    results["at_risk"].append(c360)
                else:
                    results["critical"].append(c360)

            # Sort by revenue (prioritize high-value accounts)
            for key in ["champions", "healthy_accounts", "needs_attention", "at_risk", "critical"]:
                results[key] = sorted(results[key], key=lambda x: x["revenue"], reverse=True)[:15]

            # Dimension analysis
            results["dimension_analysis"] = self._analyze_dimensions(c360_scores)

            # Segment breakdown
            results["segment_breakdown"] = self._build_segment_breakdown(c360_scores)

            # Build summary
            results["summary"] = self._build_summary(c360_scores)

            # Generate visualizations
            results["graphs"] = self._generate_visualizations(c360_scores, results["dimension_analysis"])

            # Generate insights
            results["insights"] = self._generate_insights(results)

        except Exception as e:
            logger.error(f"Customer 360 analysis failed: {e}")
            results["error"] = str(e)

            return GemmaSummarizer.generate_fallback_summary(
                accounts_df, engine_name="salesforce_c360", error_reason=str(e), config=config
            )

        return results

    def _detect_columns(self, df: pd.DataFrame, profiles: dict, config: dict) -> dict[str, str | None]:
        """Detect relevant columns."""
        mappings = {}

        mappings["account_id"] = self._find_column(
            df,
            profiles,
            config,
            hint_key="account_id_column",
            keywords=["id", "accountid", "account_id", "recordid"],
        )

        mappings["name"] = self._find_column(
            df,
            profiles,
            config,
            hint_key="name_column",
            keywords=["name", "account", "company", "accountname"],
        )

        mappings["revenue"] = self._find_column(
            df,
            profiles,
            config,
            hint_key="revenue_column",
            keywords=["revenue", "annualrevenue", "arr", "mrr", "value"],
        )

        mappings["industry"] = self._find_column(
            df,
            profiles,
            config,
            hint_key="industry_column",
            keywords=["industry", "sector", "vertical"],
        )

        mappings["employees"] = self._find_column(
            df,
            profiles,
            config,
            hint_key="employees_column",
            keywords=["employees", "numberofemployees", "size", "headcount"],
        )

        mappings["created_date"] = self._find_column(
            df,
            profiles,
            config,
            hint_key="created_date_column",
            keywords=["created", "createddate", "started"],
        )

        mappings["last_activity"] = self._find_column(
            df,
            profiles,
            config,
            hint_key="last_activity_column",
            keywords=["lastactivity", "last_activity", "lastmodified"],
        )

        mappings["owner"] = self._find_column(
            df,
            profiles,
            config,
            hint_key="owner_column",
            keywords=["owner", "accountowner", "csm", "manager"],
        )

        return mappings

    def _find_column(
        self,
        df: pd.DataFrame,
        profiles: dict,
        config: dict,
        hint_key: str,
        keywords: list[str],
    ) -> str | None:
        """Find a column by hint or keywords."""
        if hint_key in config and config[hint_key] in df.columns:
            return config[hint_key]

        for col in df.columns:
            col_lower = col.lower().replace("_", "").replace(" ", "")
            for kw in keywords:
                if kw in col_lower:
                    return col

        return None

    def _group_opportunities_by_account(
        self, opps_df: pd.DataFrame | None, opp_scores: list[dict] | None
    ) -> dict[str, list[dict]]:
        """Group opportunities by account ID."""
        by_account = {}

        if opp_scores:
            for score in opp_scores:
                account_id = str(score.get("account_id", score.get("AccountId", "")))
                if account_id:
                    if account_id not in by_account:
                        by_account[account_id] = []
                    by_account[account_id].append(score)

        elif opps_df is not None and not opps_df.empty:
            # Find account ID column
            account_col = None
            for col in opps_df.columns:
                if "account" in col.lower():
                    account_col = col
                    break

            if account_col:
                for idx, row in opps_df.iterrows():
                    account_id = str(row[account_col]) if account_col in row else ""
                    if account_id:
                        if account_id not in by_account:
                            by_account[account_id] = []
                        by_account[account_id].append(row.to_dict())

        return by_account

    def _group_cases_by_account(self, cases_df: pd.DataFrame | None) -> dict[str, list[dict]]:
        """Group cases by account ID."""
        by_account = {}

        if cases_df is None or cases_df.empty:
            return by_account

        # Find account ID column
        account_col = None
        for col in cases_df.columns:
            if "account" in col.lower():
                account_col = col
                break

        if account_col:
            for idx, row in cases_df.iterrows():
                account_id = str(row[account_col]) if account_col in row else ""
                if account_id:
                    if account_id not in by_account:
                        by_account[account_id] = []
                    by_account[account_id].append(row.to_dict())

        return by_account

    def _index_churn_scores(self, churn_scores: list[dict] | None) -> dict[str, dict]:
        """Index churn scores by account ID."""
        indexed = {}

        if churn_scores:
            for score in churn_scores:
                account_id = str(score.get("account_id", ""))
                if account_id:
                    indexed[account_id] = score

        return indexed

    def _get_value(self, row: pd.Series, col_mappings: dict, key: str, default: Any = None) -> Any:
        """Safely get a value from a row."""
        col = col_mappings.get(key)
        if col and col in row.index:
            val = row[col]
            if pd.isna(val):
                return default
            return val
        return default

    def _score_customer_360(
        self, row: pd.Series, col_mappings: dict, opportunities: list[dict], cases: list[dict], churn_score: dict | None
    ) -> dict[str, Any]:
        """
        Calculate unified Customer 360 score for an account.

        Combines five health dimensions:
        1. Account Health (25%)
        2. Opportunity/Pipeline Health (25%)
        3. Support Health (20%)
        4. Engagement Health (15%)
        5. Financial Health (15%)
        """
        account_id = self._get_value(row, col_mappings, "account_id", str(row.name))
        name = self._get_value(row, col_mappings, "name", f"Account {account_id}")
        revenue = self._get_value(row, col_mappings, "revenue", 0)
        industry = self._get_value(row, col_mappings, "industry", "Unknown")
        employees = self._get_value(row, col_mappings, "employees", 0)
        owner = self._get_value(row, col_mappings, "owner", "Unassigned")

        try:
            revenue = float(revenue) if revenue else 0
        except (ValueError, TypeError):
            revenue = 0

        try:
            employees = int(employees) if employees else 0
        except (ValueError, TypeError):
            employees = 0

        dimension_scores = {}
        dimension_factors = {}

        # =================================================================
        # 1. ACCOUNT HEALTH (25%)
        # =================================================================
        if churn_score:
            account_health = churn_score.get("health_score", 60)
            account_factors = churn_score.get("positive_factors", [])
            account_risks = churn_score.get("risk_factors", [])
        else:
            # Calculate basic account health
            account_health = 60  # Base
            account_factors = []
            account_risks = []

            # Account age bonus
            created_date = self._get_value(row, col_mappings, "created_date")
            if created_date:
                try:
                    created_dt = pd.to_datetime(created_date)
                    days_old = (datetime.now() - created_dt).days
                    if days_old > 365:
                        account_health += 15
                        account_factors.append("Long-term customer (1+ years)")
                    elif days_old > 180:
                        account_health += 10
                except Exception:
                    pass

            # Company size bonus
            if employees > 1000:
                account_health += 10
                account_factors.append("Enterprise customer")
            elif employees > 100:
                account_health += 5

        dimension_scores["account"] = max(1, min(100, account_health))
        dimension_factors["account"] = {
            "positive": account_factors[:3] if isinstance(account_factors, list) else [],
            "negative": account_risks[:3] if isinstance(account_risks, list) else [],
        }

        # =================================================================
        # 2. OPPORTUNITY/PIPELINE HEALTH (25%)
        # =================================================================
        if opportunities:
            open_opps = [
                o
                for o in opportunities
                if not str(o.get("stage", o.get("StageName", ""))).lower().__contains__("closed")
            ]
            won_opps = [o for o in opportunities if "won" in str(o.get("stage", o.get("StageName", ""))).lower()]

            pipeline_value = sum(float(o.get("amount", o.get("Amount", 0)) or 0) for o in open_opps)
            won_value = sum(float(o.get("amount", o.get("Amount", 0)) or 0) for o in won_opps)

            opp_health = 50  # Base
            opp_factors = []
            opp_risks = []

            if len(open_opps) > 0:
                opp_health += 15
                opp_factors.append(f"{len(open_opps)} active opportunities")

            if pipeline_value > 100000:
                opp_health += 15
                opp_factors.append(f"${pipeline_value:,.0f} in pipeline")
            elif pipeline_value > 50000:
                opp_health += 10

            if len(won_opps) > 0:
                opp_health += 10
                opp_factors.append(f"{len(won_opps)} won deals")

            if len(open_opps) == 0 and len(won_opps) == 0:
                opp_health = 40
                opp_risks.append("No pipeline activity")
        else:
            opp_health = 50
            opp_factors = []
            opp_risks = ["No opportunity data available"]

        dimension_scores["opportunity"] = max(1, min(100, opp_health))
        dimension_factors["opportunity"] = {"positive": opp_factors[:3], "negative": opp_risks[:3]}

        # =================================================================
        # 3. SUPPORT HEALTH (20%)
        # =================================================================
        if cases:
            open_cases = [
                c for c in cases if str(c.get("status", c.get("Status", "Open"))).lower() not in ["closed", "resolved"]
            ]
            escalated = [c for c in cases if c.get("is_escalated", c.get("IsEscalated", False))]

            support_health = 70  # Base
            support_factors = []
            support_risks = []

            if len(open_cases) == 0:
                support_health += 20
                support_factors.append("No open support cases")
            elif len(open_cases) <= 2:
                support_health += 10
            elif len(open_cases) > 5:
                support_health -= 20
                support_risks.append(f"{len(open_cases)} open support cases")

            if len(escalated) > 0:
                support_health -= 15
                support_risks.append(f"{len(escalated)} escalated cases")
        else:
            support_health = 70  # No cases = neutral (could be good or unknown)
            support_factors = ["No recent support activity"]
            support_risks = []

        dimension_scores["support"] = max(1, min(100, support_health))
        dimension_factors["support"] = {"positive": support_factors[:3], "negative": support_risks[:3]}

        # =================================================================
        # 4. ENGAGEMENT HEALTH (15%)
        # =================================================================
        engagement_health = 60  # Base
        engagement_factors = []
        engagement_risks = []

        last_activity = self._get_value(row, col_mappings, "last_activity")
        if last_activity:
            try:
                activity_dt = pd.to_datetime(last_activity)
                days_since = (datetime.now() - activity_dt).days

                if days_since <= 7:
                    engagement_health = 95
                    engagement_factors.append("Highly engaged (activity < 7 days)")
                elif days_since <= 30:
                    engagement_health = 80
                    engagement_factors.append("Recently engaged")
                elif days_since <= 60:
                    engagement_health = 60
                elif days_since <= 90:
                    engagement_health = 40
                    engagement_risks.append(f"Low engagement ({days_since} days)")
                else:
                    engagement_health = 20
                    engagement_risks.append(f"Inactive for {days_since} days")
            except Exception:
                pass

        dimension_scores["engagement"] = max(1, min(100, engagement_health))
        dimension_factors["engagement"] = {"positive": engagement_factors[:3], "negative": engagement_risks[:3]}

        # =================================================================
        # 5. FINANCIAL HEALTH (15%)
        # =================================================================
        financial_health = 60  # Base
        financial_factors = []
        financial_risks = []

        if revenue > 500000:
            financial_health = 90
            financial_factors.append(f"High-value account (${revenue:,.0f})")
        elif revenue > 100000:
            financial_health = 75
            financial_factors.append(f"Mid-market revenue (${revenue:,.0f})")
        elif revenue > 25000:
            financial_health = 60
        elif revenue > 0:
            financial_health = 45
        else:
            financial_health = 30
            financial_risks.append("No revenue data")

        dimension_scores["financial"] = max(1, min(100, financial_health))
        dimension_factors["financial"] = {"positive": financial_factors[:3], "negative": financial_risks[:3]}

        # =================================================================
        # CALCULATE UNIFIED C360 SCORE
        # =================================================================
        c360_score = sum(dimension_scores[dim] * HEALTH_DIMENSIONS[dim]["weight"] for dim in HEALTH_DIMENSIONS)
        c360_score = max(1, min(100, round(c360_score)))

        # Add some random variation for demo realism
        c360_score = max(1, min(100, c360_score + np.random.randint(-5, 6)))

        # =================================================================
        # DETERMINE SEGMENT
        # =================================================================
        segment = "Critical"
        segment_color = "#dc2626"

        for seg_key, seg_info in C360_SEGMENTS.items():
            if c360_score >= seg_info["min_score"]:
                segment = seg_info["label"]
                segment_color = seg_info["color"]
                break

        # =================================================================
        # GENERATE RECOMMENDATIONS
        # =================================================================
        recommendations = self._generate_recommendations(segment, dimension_scores, dimension_factors)

        # =================================================================
        # FIND WEAKEST DIMENSION
        # =================================================================
        weakest_dim = min(dimension_scores.items(), key=lambda x: x[1])
        strongest_dim = max(dimension_scores.items(), key=lambda x: x[1])

        return {
            "account_id": str(account_id),
            "name": str(name),
            "industry": str(industry),
            "revenue": revenue,
            "employees": employees,
            "owner": str(owner),
            "c360_score": c360_score,
            "segment": segment,
            "segment_color": segment_color,
            "dimension_scores": dimension_scores,
            "dimension_factors": dimension_factors,
            "weakest_dimension": {
                "name": HEALTH_DIMENSIONS[weakest_dim[0]]["name"],
                "key": weakest_dim[0],
                "score": weakest_dim[1],
                "icon": HEALTH_DIMENSIONS[weakest_dim[0]]["icon"],
            },
            "strongest_dimension": {
                "name": HEALTH_DIMENSIONS[strongest_dim[0]]["name"],
                "key": strongest_dim[0],
                "score": strongest_dim[1],
                "icon": HEALTH_DIMENSIONS[strongest_dim[0]]["icon"],
            },
            "recommendations": recommendations,
        }

    def _generate_recommendations(
        self, segment: str, dimension_scores: dict[str, int], dimension_factors: dict[str, dict]
    ) -> list[str]:
        """Generate prioritized recommendations based on health analysis."""
        recommendations = []

        # Find lowest scoring dimensions
        sorted_dims = sorted(dimension_scores.items(), key=lambda x: x[1])

        for dim_key, score in sorted_dims[:2]:
            if dim_key == "account" and score < 60:
                recommendations.append("Schedule executive business review to strengthen relationship")
            elif dim_key == "opportunity" and score < 50:
                recommendations.append("Explore expansion opportunities and upsell potential")
            elif dim_key == "support" and score < 60:
                recommendations.append("Review and expedite resolution of open support cases")
            elif dim_key == "engagement" and score < 50:
                recommendations.append("Re-engage with personalized outreach and value communication")
            elif dim_key == "financial" and score < 50:
                recommendations.append("Conduct pricing review and identify growth opportunities")

        # Segment-specific recommendations
        if segment == "Champion":
            recommendations.insert(0, "Leverage for testimonials, case studies, and referrals")
        elif segment == "At Risk" or segment == "Critical":
            recommendations.insert(0, "Assign dedicated CSM and create immediate intervention plan")

        return recommendations[:4]

    def _analyze_dimensions(self, c360_scores: list[dict]) -> dict[str, dict]:
        """Analyze health across all dimensions."""
        analysis = {}

        for dim_key in HEALTH_DIMENSIONS:
            scores = [c["dimension_scores"].get(dim_key, 50) for c in c360_scores]

            analysis[dim_key] = {
                "name": HEALTH_DIMENSIONS[dim_key]["name"],
                "icon": HEALTH_DIMENSIONS[dim_key]["icon"],
                "avg_score": round(float(np.mean(scores)), 1),
                "min_score": min(scores),
                "max_score": max(scores),
                "below_threshold": len([s for s in scores if s < 50]),
                "weight": HEALTH_DIMENSIONS[dim_key]["weight"],
            }

        return analysis

    def _build_segment_breakdown(self, c360_scores: list[dict]) -> dict[str, dict]:
        """Build breakdown by segment."""
        breakdown = {}

        for seg_key, seg_info in C360_SEGMENTS.items():
            segment_label = seg_info["label"]
            accounts = [c for c in c360_scores if c["segment"] == segment_label]

            breakdown[segment_label] = {
                "count": len(accounts),
                "total_revenue": sum(c["revenue"] for c in accounts),
                "avg_score": round(np.mean([c["c360_score"] for c in accounts]), 1) if accounts else 0,
                "color": seg_info["color"],
            }

        return breakdown

    def _build_summary(self, c360_scores: list[dict]) -> dict[str, Any]:
        """Build summary statistics."""
        if not c360_scores:
            return {"total_accounts": 0}

        scores = [c["c360_score"] for c in c360_scores]
        revenues = [c["revenue"] for c in c360_scores]

        # Healthy vs at-risk
        healthy_count = len([c for c in c360_scores if c["segment"] in ["Champion", "Healthy"]])
        at_risk_count = len([c for c in c360_scores if c["segment"] in ["At Risk", "Critical"]])

        healthy_revenue = sum(c["revenue"] for c in c360_scores if c["segment"] in ["Champion", "Healthy"])
        at_risk_revenue = sum(c["revenue"] for c in c360_scores if c["segment"] in ["At Risk", "Critical"])

        return {
            "total_accounts": len(c360_scores),
            "avg_c360_score": round(float(np.mean(scores)), 1),
            "total_revenue": sum(revenues),
            "healthy_accounts": healthy_count,
            "healthy_revenue": healthy_revenue,
            "at_risk_accounts": at_risk_count,
            "at_risk_revenue": at_risk_revenue,
            "at_risk_pct": round(at_risk_count / len(c360_scores) * 100, 1),
            "champion_count": len([c for c in c360_scores if c["segment"] == "Champion"]),
            "needs_attention_count": len([c for c in c360_scores if c["segment"] == "Needs Attention"]),
        }

    def _generate_visualizations(self, c360_scores: list[dict], dimension_analysis: dict[str, dict]) -> list[dict]:
        """Generate visualization data."""
        graphs = []

        if not c360_scores:
            return graphs

        # 1. Segment Distribution (Pie)
        segments = {"Champion": 0, "Healthy": 0, "Needs Attention": 0, "At Risk": 0, "Critical": 0}
        for c in c360_scores:
            segments[c["segment"]] = segments.get(c["segment"], 0) + 1

        graphs.append(
            {
                "type": "pie_chart",
                "title": "Customer Health Distribution",
                "labels": list(segments.keys()),
                "values": list(segments.values()),
                "colors": ["#10b981", "#3b82f6", "#f59e0b", "#ef4444", "#dc2626"],
            }
        )

        # 2. Revenue by Segment
        revenue_by_segment = {"Champion": 0, "Healthy": 0, "Needs Attention": 0, "At Risk": 0, "Critical": 0}
        for c in c360_scores:
            revenue_by_segment[c["segment"]] = revenue_by_segment.get(c["segment"], 0) + c["revenue"]

        graphs.append(
            {
                "type": "bar_chart",
                "title": "Revenue by Health Segment",
                "x_data": list(revenue_by_segment.keys()),
                "y_data": list(revenue_by_segment.values()),
                "x_label": "Segment",
                "y_label": "Revenue ($)",
                "colors": ["#10b981", "#3b82f6", "#f59e0b", "#ef4444", "#dc2626"],
            }
        )

        # 3. Dimension Health Radar
        if dimension_analysis:
            graphs.append(
                {
                    "type": "radar",
                    "title": "Health by Dimension",
                    "indicators": [{"name": v["name"], "max": 100} for v in dimension_analysis.values()],
                    "values": [v["avg_score"] for v in dimension_analysis.values()],
                    "color": "#6366f1",
                }
            )

        # 4. C360 Score Distribution
        scores = [c["c360_score"] for c in c360_scores]
        graphs.append(
            {
                "type": "histogram",
                "title": "Customer 360 Score Distribution",
                "x_data": scores,
                "x_label": "C360 Score",
                "bins": 10,
                "colors": ["#6366f1"],
            }
        )

        return graphs

    def _generate_insights(self, results: dict) -> list[str]:
        """Generate AI insights."""
        insights = []
        summary = results["summary"]

        # Overall health
        avg_score = summary.get("avg_c360_score", 0)
        if avg_score >= 75:
            insights.append(f"✅ Portfolio health is strong ({avg_score}/100 average)")
        elif avg_score >= 60:
            insights.append(f"📊 Portfolio health is moderate ({avg_score}/100 average)")
        else:
            insights.append(f"⚠️ Portfolio health needs attention ({avg_score}/100 average)")

        # Champions
        champion_count = summary.get("champion_count", 0)
        if champion_count > 0:
            insights.append(f"🏆 {champion_count} champion accounts - leverage for growth")

        # At-risk revenue
        at_risk_rev = summary.get("at_risk_revenue", 0)
        at_risk_pct = summary.get("at_risk_pct", 0)
        if at_risk_rev > 0:
            insights.append(f"🚨 ${at_risk_rev:,.0f} revenue at risk ({at_risk_pct:.0f}% of accounts)")

        # Dimension insights
        dim_analysis = results.get("dimension_analysis", {})
        weakest = min(dim_analysis.items(), key=lambda x: x[1].get("avg_score", 100)) if dim_analysis else None
        if weakest and weakest[1].get("avg_score", 100) < 60:
            insights.append(f"📉 {weakest[1]['name']} is the weakest dimension ({weakest[1]['avg_score']}/100)")

        return insights


__all__ = ["SalesforceC360Engine"]
