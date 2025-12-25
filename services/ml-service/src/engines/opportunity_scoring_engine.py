"""
Opportunity Scoring Engine - Win Probability Prediction

Predictive scoring for sales opportunities:
- Win probability (1-99 score)
- Deal velocity analysis
- Risk factor identification
- Stage progression likelihood

Provides explainable AI with positive/negative factors affecting deal health.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import logging
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from core import ColumnMapper, ColumnProfiler, EngineRequirements, SemanticType
from core.gemma_summarizer import GemmaSummarizer
from core.premium_models import ConfigParameter

logger = logging.getLogger(__name__)


# =============================================================================
# SCORING CONFIGURATION
# =============================================================================

STAGE_WIN_RATES = {
    # Stage name patterns -> base win probability
    "prospecting": 10,
    "qualification": 20,
    "needs_analysis": 30,
    "needs analysis": 30,
    "value_proposition": 40,
    "value proposition": 40,
    "id_decision_makers": 50,
    "id. decision makers": 50,
    "proposal": 60,
    "proposal/price quote": 60,
    "negotiation": 75,
    "negotiation/review": 75,
    "closed_won": 100,
    "closed won": 100,
    "closed_lost": 0,
    "closed lost": 0,
}

DEFAULT_AVG_CYCLE_DAYS = 45  # Average days to close


class OpportunityScoringEngine:
    """
    Win Probability Prediction Engine for Sales Opportunities.

    Produces scores 1-99 indicating likelihood of winning the deal.
    Analyzes deal velocity, risk factors, and provides recommendations.

    Features:
    - Stage-based win probability
    - Deal velocity analysis (comparing to average cycle time)
    - Amount/deal size scoring
    - Risk factor identification
    - Recommended actions per opportunity
    - Deal health indicators (Green/Yellow/Red)
    """

    def __init__(self):
        self.name = "Opportunity Scoring Engine (Win Probability)"
        self.profiler = ColumnProfiler()
        self.mapper = ColumnMapper()

    def get_engine_info(self) -> dict[str, str]:
        """Get engine display information."""
        return {
            "name": "opportunity_scoring",
            "display_name": "Opportunity Scoring",
            "icon": "🎯",
            "task_type": "scoring",
            "description": "AI-powered win probability scoring for deals",
        }

    def get_config_schema(self) -> list[ConfigParameter]:
        """Get configuration schema for UI."""
        return [
            ConfigParameter(
                name="opportunity_id_column",
                type="select",
                default=None,
                range=[],
                description="Opportunity identifier column",
            ),
            ConfigParameter(
                name="name_column", type="select", default=None, range=[], description="Opportunity name column"
            ),
            ConfigParameter(
                name="stage_column", type="select", default=None, range=[], description="Stage/status column"
            ),
            ConfigParameter(
                name="amount_column", type="select", default=None, range=[], description="Deal amount column"
            ),
            ConfigParameter(
                name="close_date_column",
                type="select",
                default=None,
                range=[],
                description="Expected close date column",
            ),
            ConfigParameter(
                name="created_date_column",
                type="select",
                default=None,
                range=[],
                description="Opportunity creation date column",
            ),
            ConfigParameter(
                name="probability_column",
                type="select",
                default=None,
                range=[],
                description="Existing probability field (optional)",
            ),
            ConfigParameter(
                name="avg_cycle_days",
                type="number",
                default=DEFAULT_AVG_CYCLE_DAYS,
                range=[7, 365],
                description="Average sales cycle in days",
            ),
        ]

    def get_methodology_info(self) -> dict[str, Any]:
        """Get methodology documentation."""
        return {
            "name": "Predictive Opportunity Scoring",
            "url": "https://www.salesforce.com/products/einstein/features/opportunity-scoring/",
            "steps": [
                {"step_number": 1, "title": "Stage Analysis", "description": "Base probability from current stage"},
                {
                    "step_number": 2,
                    "title": "Velocity Analysis",
                    "description": "Compare deal progress to typical cycle",
                },
                {"step_number": 3, "title": "Deal Size Scoring", "description": "Adjust for deal amount patterns"},
                {"step_number": 4, "title": "Risk Identification", "description": "Detect stalled deals and red flags"},
                {"step_number": 5, "title": "Factor Analysis", "description": "Generate explainable factors"},
            ],
            "limitations": [
                "Stage-based model - production may need historical win/loss training",
                "Assumes standard sales process stages",
            ],
            "assumptions": ["Later stages have higher win probability", "Deals past due date have lower probability"],
        }

    def get_requirements(self) -> EngineRequirements:
        """Get engine data requirements."""
        return EngineRequirements(
            required_semantics=[SemanticType.CATEGORICAL],
            optional_semantics={"amount": [SemanticType.NUMERIC_CONTINUOUS], "date": [SemanticType.TEMPORAL]},
            required_entities=[],
            preferred_domains=["crm", "sales"],
            applicable_tasks=["opportunity_scoring", "forecasting"],
            min_rows=1,
            min_numeric_cols=0,
        )

    def analyze(self, df: pd.DataFrame, config: dict | None = None) -> dict[str, Any]:
        """
        Score opportunities for win probability.

        Args:
            df: DataFrame with opportunity data
            config: Optional configuration with column hints

        Returns:
            Dict with:
            - scored_opportunities: List with scores 1-99
            - summary: Aggregate statistics
            - pipeline_health: Overall pipeline assessment
            - graphs: Visualization data
            - insights: AI-generated observations
        """
        config = config or {}
        avg_cycle = config.get("avg_cycle_days", DEFAULT_AVG_CYCLE_DAYS)

        results = {
            "engine": "opportunity_scoring",
            "summary": {},
            "scored_opportunities": [],
            "pipeline_health": {},
            "at_risk_deals": [],
            "graphs": [],
            "insights": [],
            "column_mappings": {},
        }

        try:
            # Profile dataset for smart column detection
            profiles = self.profiler.profile_dataset(df)

            # Detect columns
            col_mappings = self._detect_columns(df, profiles, config)
            results["column_mappings"] = col_mappings

            # Score each opportunity
            scored_opps = []
            at_risk = []

            for idx, row in df.iterrows():
                opp_score = self._score_opportunity(row, col_mappings, avg_cycle)
                scored_opps.append(opp_score)

                # Track at-risk deals
                if opp_score["health"] == "Red" or opp_score["score"] < 30:
                    at_risk.append(opp_score)

            results["scored_opportunities"] = scored_opps
            results["at_risk_deals"] = sorted(at_risk, key=lambda x: x["amount"], reverse=True)[:10]

            # Build summary statistics
            scores = [o["score"] for o in scored_opps]
            amounts = [o["amount"] for o in scored_opps if o["amount"] > 0]

            # Weighted pipeline value (amount * probability)
            weighted_pipeline = sum(o["amount"] * (o["score"] / 100) for o in scored_opps if o["stage_type"] == "open")

            total_pipeline = sum(o["amount"] for o in scored_opps if o["stage_type"] == "open")

            results["summary"] = {
                "total_opportunities": len(scored_opps),
                "avg_win_probability": float(np.mean(scores)),
                "median_win_probability": float(np.median(scores)),
                "weighted_pipeline_value": round(weighted_pipeline, 2),
                "total_pipeline_value": round(total_pipeline, 2),
                "avg_deal_size": round(float(np.mean(amounts)), 2) if amounts else 0,
                "high_probability_deals": sum(1 for s in scores if s >= 70),
                "at_risk_count": len(at_risk),
            }

            # Pipeline health assessment
            results["pipeline_health"] = self._assess_pipeline_health(scored_opps, results["summary"])

            # Generate visualizations
            results["graphs"] = self._generate_visualizations(scored_opps)

            # Generate insights
            results["insights"] = self._generate_insights(results)

        except Exception as e:
            logger.error(f"Opportunity scoring failed: {e}")
            results["error"] = str(e)

            return GemmaSummarizer.generate_fallback_summary(
                df, engine_name="opportunity_scoring", error_reason=str(e), config=config
            )

        return results

    def _detect_columns(self, df: pd.DataFrame, profiles: dict, config: dict) -> dict[str, str | None]:
        """Detect relevant columns using schema intelligence."""
        mappings = {}

        # Opportunity ID
        mappings["opp_id"] = self._find_column(
            df,
            profiles,
            config,
            hint_key="opportunity_id_column",
            keywords=["id", "opportunity_id", "oppid", "recordid"],
            semantic_type=None,
        )

        # Opportunity Name
        mappings["name"] = self._find_column(
            df,
            profiles,
            config,
            hint_key="name_column",
            keywords=["name", "opportunity_name", "deal", "oppname"],
            semantic_type=SemanticType.CATEGORICAL,
        )

        # Stage
        mappings["stage"] = self._find_column(
            df,
            profiles,
            config,
            hint_key="stage_column",
            keywords=["stage", "stagename", "status", "phase"],
            semantic_type=SemanticType.CATEGORICAL,
        )

        # Amount
        mappings["amount"] = self._find_column(
            df,
            profiles,
            config,
            hint_key="amount_column",
            keywords=["amount", "value", "deal_size", "revenue", "price"],
            semantic_type=SemanticType.NUMERIC_CONTINUOUS,
        )

        # Close Date
        mappings["close_date"] = self._find_column(
            df,
            profiles,
            config,
            hint_key="close_date_column",
            keywords=["close", "closedate", "expected_close", "close_date"],
            semantic_type=SemanticType.TEMPORAL,
        )

        # Created Date
        mappings["created_date"] = self._find_column(
            df,
            profiles,
            config,
            hint_key="created_date_column",
            keywords=["created", "createddate", "created_date", "createdat"],
            semantic_type=SemanticType.TEMPORAL,
        )

        # Account ID
        mappings["account_id"] = self._find_column(
            df,
            profiles,
            config,
            hint_key="account_id_column",
            keywords=["account", "accountid", "account_id", "company"],
            semantic_type=SemanticType.CATEGORICAL,
        )

        # Existing Probability
        mappings["probability"] = self._find_column(
            df,
            profiles,
            config,
            hint_key="probability_column",
            keywords=["probability", "prob", "win_rate", "confidence"],
            semantic_type=SemanticType.NUMERIC_CONTINUOUS,
        )

        return mappings

    def _find_column(
        self,
        df: pd.DataFrame,
        profiles: dict,
        config: dict,
        hint_key: str,
        keywords: list[str],
        semantic_type: SemanticType | None,
    ) -> str | None:
        """Find a column by hint, keywords, or semantic type."""
        if hint_key in config and config[hint_key] in df.columns:
            return config[hint_key]

        for col in df.columns:
            col_lower = col.lower().replace("_", "").replace(" ", "")
            for kw in keywords:
                if kw in col_lower:
                    return col

        if semantic_type and profiles:
            for col, profile in profiles.items():
                if profile.semantic_type == semantic_type:
                    return col

        return None

    def _score_opportunity(self, row: pd.Series, col_mappings: dict[str, str | None], avg_cycle: int) -> dict[str, Any]:
        """
        Score a single opportunity.

        Returns dict with:
        - opp_id: Identifier
        - name: Opportunity name
        - score: 1-99 win probability
        - health: Green/Yellow/Red
        - positive_factors: List of positive factors
        - negative_factors: List of negative factors
        - recommended_action: Suggested next step
        """
        positive_factors = []
        negative_factors = []

        opp_id = self._get_value(row, col_mappings, "opp_id", default=str(row.name))
        opp_name = self._get_value(row, col_mappings, "name", default=f"Opportunity {opp_id}")
        amount = self._get_value(row, col_mappings, "amount", default=0)

        try:
            amount = float(amount) if amount else 0
        except (ValueError, TypeError):
            amount = 0

        # =================================================================
        # 1. STAGE-BASED SCORING
        # =================================================================
        stage = self._get_value(row, col_mappings, "stage", default="")
        stage_str = str(stage).lower().strip() if stage else ""

        # Determine if closed
        is_closed_won = "closed" in stage_str and "won" in stage_str
        is_closed_lost = "closed" in stage_str and "lost" in stage_str
        is_closed = is_closed_won or is_closed_lost

        stage_type = "won" if is_closed_won else ("lost" if is_closed_lost else "open")

        # Get base probability from stage
        base_probability = 50  # default
        stage_matched = False

        for stage_pattern, prob in STAGE_WIN_RATES.items():
            if stage_pattern in stage_str:
                base_probability = prob
                stage_matched = True
                break

        if not stage_matched and stage_str:
            # Estimate based on common patterns
            if "prospect" in stage_str:
                base_probability = 15
            elif "qual" in stage_str:
                base_probability = 25
            elif "propos" in stage_str:
                base_probability = 55
            elif "negot" in stage_str:
                base_probability = 70

        if base_probability >= 60:
            positive_factors.append({"factor": f"Advanced stage: {stage}", "impact": 15})
        elif base_probability <= 25:
            negative_factors.append({"factor": f"Early stage: {stage}", "impact": -10})

        # =================================================================
        # 2. CLOSE DATE ANALYSIS
        # =================================================================
        close_date = self._get_value(row, col_mappings, "close_date")
        days_to_close = None
        is_overdue = False

        if close_date is not None and not is_closed:
            try:
                if isinstance(close_date, str):
                    close_dt = pd.to_datetime(close_date)
                else:
                    close_dt = pd.Timestamp(close_date)

                days_to_close = (close_dt - datetime.now()).days

                if days_to_close < 0:
                    is_overdue = True
                    negative_factors.append({"factor": f"Past due by {abs(days_to_close)} days", "impact": -15})
                    base_probability = max(10, base_probability - 20)
                elif days_to_close <= 7:
                    positive_factors.append({"factor": f"Closing soon ({days_to_close} days)", "impact": 10})
                elif days_to_close <= 30:
                    positive_factors.append({"factor": "Closing this month", "impact": 5})
            except Exception:
                pass

        # =================================================================
        # 3. DEAL VELOCITY ANALYSIS
        # =================================================================
        created_date = self._get_value(row, col_mappings, "created_date")
        days_in_pipeline = None

        if created_date is not None and not is_closed:
            try:
                if isinstance(created_date, str):
                    created_dt = pd.to_datetime(created_date)
                else:
                    created_dt = pd.Timestamp(created_date)

                days_in_pipeline = (datetime.now() - created_dt).days

                if days_in_pipeline > avg_cycle * 2:
                    negative_factors.append(
                        {"factor": f"Stalled deal ({days_in_pipeline} days in pipeline)", "impact": -15}
                    )
                    base_probability = max(5, base_probability - 15)
                elif days_in_pipeline > avg_cycle:
                    negative_factors.append(
                        {"factor": f"Slow velocity ({days_in_pipeline} days, avg: {avg_cycle})", "impact": -8}
                    )
                elif days_in_pipeline < avg_cycle * 0.5 and base_probability >= 50:
                    positive_factors.append({"factor": "Fast-moving deal", "impact": 10})
                    base_probability = min(95, base_probability + 5)
            except Exception:
                pass

        # =================================================================
        # 4. DEAL SIZE SCORING
        # =================================================================
        if amount > 0:
            if amount >= 100000:
                positive_factors.append({"factor": f"High-value deal (${amount:,.0f})", "impact": 5})
            elif amount >= 50000:
                positive_factors.append({"factor": f"Significant deal size (${amount:,.0f})", "impact": 3})
        else:
            negative_factors.append({"factor": "No deal amount specified", "impact": -5})

        # =================================================================
        # CALCULATE FINAL SCORE (1-99)
        # =================================================================
        # Apply factor adjustments
        adjustment = sum(f["impact"] for f in positive_factors) + sum(f["impact"] for f in negative_factors)
        final_score = base_probability + adjustment

        # Clamp to 1-99 range
        final_score = max(1, min(99, int(round(final_score))))

        # For closed deals, override
        if is_closed_won:
            final_score = 99
        elif is_closed_lost:
            final_score = 1

        # =================================================================
        # DETERMINE HEALTH INDICATOR
        # =================================================================
        if is_overdue or final_score < 30 or (days_in_pipeline and days_in_pipeline > avg_cycle * 1.5):
            health = "Red"
        elif final_score < 50 or (days_in_pipeline and days_in_pipeline > avg_cycle):
            health = "Yellow"
        else:
            health = "Green"

        # =================================================================
        # RECOMMENDED ACTION
        # =================================================================
        if is_closed:
            recommended_action = "No action - deal closed"
        elif is_overdue:
            recommended_action = "Urgent: Follow up on overdue close date"
        elif days_in_pipeline and days_in_pipeline > avg_cycle * 1.5:
            recommended_action = "Re-engage: Deal may be stalling"
        elif final_score >= 70 and days_to_close and days_to_close <= 14:
            recommended_action = "Push to close: High probability, closing soon"
        elif final_score >= 60:
            recommended_action = "Advance deal: Schedule decision maker meeting"
        elif final_score >= 40:
            recommended_action = "Qualification needed: Confirm budget and timeline"
        else:
            recommended_action = "Early stage: Focus on discovery and needs analysis"

        return {
            "opp_id": opp_id,
            "name": opp_name,
            "stage": stage,
            "stage_type": stage_type,
            "amount": amount,
            "score": final_score,
            "health": health,
            "days_in_pipeline": days_in_pipeline,
            "days_to_close": days_to_close,
            "positive_factors": positive_factors,
            "negative_factors": negative_factors,
            "recommended_action": recommended_action,
        }

    def _get_value(self, row: pd.Series, col_mappings: dict, key: str, default: Any = None) -> Any:
        """Safely get a value from a row using column mappings."""
        col = col_mappings.get(key)
        if col and col in row.index:
            val = row[col]
            if pd.isna(val):
                return default
            return val
        return default

    def _assess_pipeline_health(self, scored_opps: list[dict], summary: dict) -> dict[str, Any]:
        """Assess overall pipeline health."""
        open_opps = [o for o in scored_opps if o["stage_type"] == "open"]

        if not open_opps:
            return {"status": "Empty", "message": "No open opportunities in pipeline", "color": "gray"}

        avg_score = np.mean([o["score"] for o in open_opps])
        at_risk_count = sum(1 for o in open_opps if o["health"] == "Red")
        at_risk_pct = at_risk_count / len(open_opps) * 100

        if avg_score >= 60 and at_risk_pct < 20:
            return {
                "status": "Healthy",
                "message": f"Pipeline is strong with {avg_score:.0f}% avg probability",
                "color": "green",
                "avg_probability": round(avg_score, 1),
                "at_risk_percentage": round(at_risk_pct, 1),
            }
        elif avg_score >= 40 and at_risk_pct < 40:
            return {
                "status": "Moderate",
                "message": f"{at_risk_pct:.0f}% of deals need attention",
                "color": "yellow",
                "avg_probability": round(avg_score, 1),
                "at_risk_percentage": round(at_risk_pct, 1),
            }
        else:
            return {
                "status": "At Risk",
                "message": f"High risk: {at_risk_count} deals need immediate attention",
                "color": "red",
                "avg_probability": round(avg_score, 1),
                "at_risk_percentage": round(at_risk_pct, 1),
            }

    def _generate_visualizations(self, scored_opps: list[dict]) -> list[dict]:
        """Generate visualization data for the frontend."""
        graphs = []

        open_opps = [o for o in scored_opps if o["stage_type"] == "open"]

        if not open_opps:
            return graphs

        # 1. Win Probability Distribution
        scores = [o["score"] for o in open_opps]
        graphs.append(
            {
                "type": "histogram",
                "title": "Win Probability Distribution",
                "x_data": scores,
                "x_label": "Win Probability (%)",
                "bins": 10,
                "colors": ["#6366f1"],
            }
        )

        # 2. Pipeline by Health
        health_counts = {"Green": 0, "Yellow": 0, "Red": 0}
        for opp in open_opps:
            health_counts[opp["health"]] = health_counts.get(opp["health"], 0) + 1

        graphs.append(
            {
                "type": "pie_chart",
                "title": "Deal Health Distribution",
                "labels": list(health_counts.keys()),
                "values": list(health_counts.values()),
                "colors": ["#10b981", "#f59e0b", "#ef4444"],
            }
        )

        # 3. Pipeline Value by Stage
        stage_values = {}
        for opp in open_opps:
            stage = opp["stage"]
            if stage not in stage_values:
                stage_values[stage] = 0
            stage_values[stage] += opp["amount"]

        graphs.append(
            {
                "type": "bar_chart",
                "title": "Pipeline Value by Stage",
                "x_data": list(stage_values.keys()),
                "y_data": list(stage_values.values()),
                "x_label": "Stage",
                "y_label": "Pipeline Value ($)",
                "colors": ["#8b5cf6"],
            }
        )

        # 4. Top Opportunities by Score
        top_opps = sorted(open_opps, key=lambda x: x["score"], reverse=True)[:10]
        graphs.append(
            {
                "type": "bar_chart",
                "title": "Top 10 Opportunities by Win Probability",
                "x_data": [o["name"][:25] for o in top_opps],
                "y_data": [o["score"] for o in top_opps],
                "x_label": "Opportunity",
                "y_label": "Win Probability (%)",
                "colors": ["#10b981"],
            }
        )

        return graphs

    def _generate_insights(self, results: dict) -> list[str]:
        """Generate AI insights from scoring results."""
        insights = []
        summary = results["summary"]
        health = results["pipeline_health"]

        # Pipeline value insight
        weighted = summary.get("weighted_pipeline_value", 0)
        total = summary.get("total_pipeline_value", 0)
        if total > 0:
            insights.append(
                f"💰 Weighted pipeline value: ${weighted:,.0f} ({(weighted / total * 100):.0f}% of ${total:,.0f} total)"
            )

        # Pipeline health insight
        if health.get("status") == "Healthy":
            insights.append(f"✅ Pipeline is healthy with {health.get('avg_probability', 0):.0f}% avg win probability")
        elif health.get("status") == "At Risk":
            insights.append(f"🚨 Pipeline at risk: {health.get('message')}")
        else:
            insights.append(f"⚠️ Pipeline health: {health.get('message')}")

        # High probability insight
        high_prob = summary.get("high_probability_deals", 0)
        if high_prob > 0:
            insights.append(f"🎯 {high_prob} deals have >70% win probability - focus on closing!")

        # At-risk deals insight
        at_risk = results.get("at_risk_deals", [])
        if at_risk:
            total_at_risk_value = sum(d["amount"] for d in at_risk)
            insights.append(f"⚠️ {len(at_risk)} at-risk deals worth ${total_at_risk_value:,.0f} need attention")

        # Average deal size insight
        avg_size = summary.get("avg_deal_size", 0)
        if avg_size > 0:
            insights.append(f"📊 Average deal size: ${avg_size:,.0f}")

        return insights


__all__ = ["OpportunityScoringEngine"]
