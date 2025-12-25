"""
Lead Scoring Engine - Einstein-Style Predictive Lead Scoring

Assigns 1-99 scores to leads based on:
- Demographic fit (industry, company size, job title)
- Behavioral scoring (engagement, recency)
- Firmographic scoring (company revenue, employee count)

Provides explainable AI with positive/negative factors.
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
# SCORING WEIGHTS - Configurable ICP Criteria
# =============================================================================

DEFAULT_ICP_WEIGHTS = {
    # Industry alignment (which industries are ideal)
    "preferred_industries": ["Technology", "Financial Services", "Healthcare", "Manufacturing"],
    "industry_weight": 15,
    # Company size preferences
    "preferred_company_sizes": ["Enterprise", "Mid-Market"],
    "company_size_weight": 12,
    # Job title/role preferences
    "preferred_titles": ["CEO", "CTO", "CFO", "VP", "Director", "Manager", "Head of"],
    "title_weight": 15,
    # Lead source quality
    "high_quality_sources": ["Referral", "Partner", "Trade Show", "Webinar"],
    "medium_quality_sources": ["Website", "Content", "SEO"],
    "source_weight": 10,
    # Engagement signals
    "engagement_weight": 20,
    # Recency weight (how recently created/updated)
    "recency_weight": 15,
    # Data completeness
    "completeness_weight": 13,
}


class LeadScoringEngine:
    """
    Einstein-style Lead Scoring Engine.

    Produces scores 1-99 for each lead with explainable factors.
    Uses a weighted scoring model based on ICP (Ideal Customer Profile) criteria.

    Features:
    - Demographic fit scoring
    - Behavioral/engagement scoring
    - Firmographic scoring
    - Recency weighting
    - Explainable positive/negative factors
    - Segment classification (Hot, Warm, Cold)
    """

    def __init__(self):
        self.name = "Lead Scoring Engine (Einstein-Style)"
        self.profiler = ColumnProfiler()
        self.mapper = ColumnMapper()
        self.icp_weights = DEFAULT_ICP_WEIGHTS.copy()

    def get_engine_info(self) -> dict[str, str]:
        """Get engine display information."""
        return {
            "name": "lead_scoring",
            "display_name": "Lead Scoring",
            "icon": "🎯",
            "task_type": "scoring",
            "description": "AI-powered lead qualification scoring",
        }

    def get_config_schema(self) -> list[ConfigParameter]:
        """Get configuration schema for UI."""
        return [
            ConfigParameter(
                name="lead_id_column", type="select", default=None, range=[], description="Lead identifier column"
            ),
            ConfigParameter(
                name="industry_column", type="select", default=None, range=[], description="Industry/sector column"
            ),
            ConfigParameter(name="title_column", type="select", default=None, range=[], description="Job title column"),
            ConfigParameter(
                name="company_column", type="select", default=None, range=[], description="Company name column"
            ),
            ConfigParameter(
                name="source_column", type="select", default=None, range=[], description="Lead source column"
            ),
            ConfigParameter(
                name="created_date_column",
                type="select",
                default=None,
                range=[],
                description="Lead creation date column",
            ),
            ConfigParameter(
                name="status_column", type="select", default=None, range=[], description="Lead status column"
            ),
        ]

    def get_methodology_info(self) -> dict[str, Any]:
        """Get methodology documentation."""
        return {
            "name": "Predictive Lead Scoring",
            "url": "https://www.salesforce.com/products/einstein/features/lead-scoring/",
            "steps": [
                {
                    "step_number": 1,
                    "title": "Data Profiling",
                    "description": "Analyze lead attributes and detect column types",
                },
                {
                    "step_number": 2,
                    "title": "Demographic Scoring",
                    "description": "Score based on industry, title, company fit",
                },
                {
                    "step_number": 3,
                    "title": "Behavioral Scoring",
                    "description": "Score based on engagement and activity signals",
                },
                {
                    "step_number": 4,
                    "title": "Recency Weighting",
                    "description": "Apply time-decay to prioritize recent leads",
                },
                {
                    "step_number": 5,
                    "title": "Factor Analysis",
                    "description": "Generate explainable positive/negative factors",
                },
            ],
            "limitations": [
                "Requires sufficient lead attributes for accurate scoring",
                "Rule-based model - production may require ML training",
            ],
            "assumptions": ["Historical conversion patterns follow ICP criteria", "Lead data is reasonably complete"],
        }

    def get_requirements(self) -> EngineRequirements:
        """Get engine data requirements."""
        return EngineRequirements(
            required_semantics=[SemanticType.CATEGORICAL],
            optional_semantics={
                "id": [SemanticType.CATEGORICAL],
                "company": [SemanticType.CATEGORICAL],
                "numeric": [SemanticType.NUMERIC_CONTINUOUS],
            },
            required_entities=[],
            preferred_domains=["crm", "sales", "marketing"],
            applicable_tasks=["lead_scoring", "qualification"],
            min_rows=1,
            min_numeric_cols=0,
        )

    def analyze(self, df: pd.DataFrame, config: dict | None = None) -> dict[str, Any]:
        """
        Score leads using weighted multi-factor model.

        Args:
            df: DataFrame with lead data
            config: Optional configuration with column hints and ICP settings

        Returns:
            Dict with:
            - scored_leads: List of leads with scores 1-99
            - summary: Aggregate statistics
            - segments: Lead breakdown by score tier
            - graphs: Visualization data
            - insights: AI-generated observations
        """
        config = config or {}

        # Allow custom ICP weights
        if "icp_weights" in config:
            self.icp_weights.update(config["icp_weights"])

        results = {
            "engine": "lead_scoring",
            "summary": {},
            "scored_leads": [],
            "segments": {},
            "graphs": [],
            "insights": [],
            "column_mappings": {},
            "factors_analysis": {},
        }

        try:
            # Profile dataset for smart column detection
            profiles = self.profiler.profile_dataset(df)

            # Detect columns
            col_mappings = self._detect_columns(df, profiles, config)
            results["column_mappings"] = col_mappings

            # Score each lead
            scored_leads = []
            all_factors = {"positive": [], "negative": []}

            for idx, row in df.iterrows():
                lead_score = self._score_lead(row, col_mappings, profiles)
                scored_leads.append(lead_score)

                # Aggregate factors
                for factor in lead_score["positive_factors"]:
                    all_factors["positive"].append(factor["factor"])
                for factor in lead_score["negative_factors"]:
                    all_factors["negative"].append(factor["factor"])

            results["scored_leads"] = scored_leads

            # Build summary statistics
            scores = [l["score"] for l in scored_leads]
            results["summary"] = {
                "total_leads": len(scored_leads),
                "avg_score": float(np.mean(scores)),
                "median_score": float(np.median(scores)),
                "high_quality_leads": sum(1 for s in scores if s >= 70),
                "medium_quality_leads": sum(1 for s in scores if 40 <= s < 70),
                "low_quality_leads": sum(1 for s in scores if s < 40),
                "hot_leads_pct": round(sum(1 for s in scores if s >= 80) / len(scores) * 100, 1) if scores else 0,
            }

            # Segment leads
            results["segments"] = self._segment_leads(scored_leads)

            # Factor frequency analysis
            results["factors_analysis"] = {
                "top_positive_factors": self._count_factors(all_factors["positive"])[:5],
                "top_negative_factors": self._count_factors(all_factors["negative"])[:5],
            }

            # Generate visualizations
            results["graphs"] = self._generate_visualizations(scored_leads)

            # Generate insights
            results["insights"] = self._generate_insights(results)

        except Exception as e:
            logger.error(f"Lead scoring failed: {e}")
            results["error"] = str(e)

            # Fallback to Gemma summarizer
            return GemmaSummarizer.generate_fallback_summary(
                df, engine_name="lead_scoring", error_reason=str(e), config=config
            )

        return results

    def _detect_columns(self, df: pd.DataFrame, profiles: dict, config: dict) -> dict[str, str | None]:
        """Detect relevant columns using schema intelligence."""
        mappings = {}

        # Lead ID
        mappings["lead_id"] = self._find_column(
            df,
            profiles,
            config,
            hint_key="lead_id_column",
            keywords=["id", "lead_id", "leadid", "record"],
            semantic_type=None,
        )

        # Lead Name
        mappings["name"] = self._find_column(
            df,
            profiles,
            config,
            hint_key="name_column",
            keywords=["name", "lead_name", "fullname", "leadname"],
            semantic_type=SemanticType.CATEGORICAL,
        )

        # Industry
        mappings["industry"] = self._find_column(
            df,
            profiles,
            config,
            hint_key="industry_column",
            keywords=["industry", "sector", "vertical"],
            semantic_type=SemanticType.CATEGORICAL,
        )

        # Job Title
        mappings["title"] = self._find_column(
            df,
            profiles,
            config,
            hint_key="title_column",
            keywords=["title", "job_title", "jobtitle", "role", "position"],
            semantic_type=SemanticType.CATEGORICAL,
        )

        # Company
        mappings["company"] = self._find_column(
            df,
            profiles,
            config,
            hint_key="company_column",
            keywords=["company", "organization", "account", "firm"],
            semantic_type=SemanticType.CATEGORICAL,
        )

        # Lead Source
        mappings["source"] = self._find_column(
            df,
            profiles,
            config,
            hint_key="source_column",
            keywords=["source", "leadsource", "lead_source", "origin", "channel"],
            semantic_type=SemanticType.CATEGORICAL,
        )

        # Created Date
        mappings["created_date"] = self._find_column(
            df,
            profiles,
            config,
            hint_key="created_date_column",
            keywords=["created", "createddate", "date", "createdat"],
            semantic_type=SemanticType.TEMPORAL,
        )

        # Status
        mappings["status"] = self._find_column(
            df,
            profiles,
            config,
            hint_key="status_column",
            keywords=["status", "stage", "state", "leadstatus"],
            semantic_type=SemanticType.CATEGORICAL,
        )

        # Email (for completeness scoring)
        mappings["email"] = self._find_column(
            df,
            profiles,
            config,
            hint_key="email_column",
            keywords=["email", "mail", "e-mail"],
            semantic_type=SemanticType.CATEGORICAL,
        )

        # Phone
        mappings["phone"] = self._find_column(
            df,
            profiles,
            config,
            hint_key="phone_column",
            keywords=["phone", "telephone", "mobile", "cell"],
            semantic_type=SemanticType.CATEGORICAL,
        )

        # Company Size / Employees
        mappings["company_size"] = self._find_column(
            df,
            profiles,
            config,
            hint_key="company_size_column",
            keywords=["employees", "size", "numberofemployees", "company_size"],
            semantic_type=None,
        )

        # Annual Revenue
        mappings["revenue"] = self._find_column(
            df,
            profiles,
            config,
            hint_key="revenue_column",
            keywords=["revenue", "annualrevenue", "annual_revenue", "arr"],
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
        # Check config hint first
        if hint_key in config and config[hint_key] in df.columns:
            return config[hint_key]

        # Search by keywords
        for col in df.columns:
            col_lower = col.lower().replace("_", "").replace(" ", "")
            for kw in keywords:
                if kw in col_lower:
                    return col

        # Search by semantic type
        if semantic_type and profiles:
            for col, profile in profiles.items():
                if profile.semantic_type == semantic_type:
                    return col

        return None

    def _score_lead(self, row: pd.Series, col_mappings: dict[str, str | None], profiles: dict) -> dict[str, Any]:
        """
        Score a single lead using weighted factors.

        Returns dict with:
        - lead_id: Identifier
        - name: Lead name
        - score: 1-99 score
        - segment: Hot/Warm/Cold
        - positive_factors: List of positive scoring factors
        - negative_factors: List of negative scoring factors
        """
        score_components = []
        positive_factors = []
        negative_factors = []

        lead_id = self._get_value(row, col_mappings, "lead_id", default=str(row.name))
        lead_name = self._get_value(row, col_mappings, "name", default=f"Lead {lead_id}")

        # =================================================================
        # 1. INDUSTRY SCORING (0-15 points)
        # =================================================================
        industry = self._get_value(row, col_mappings, "industry")
        industry_score = 0

        if industry:
            industry_str = str(industry).strip()
            if any(pref.lower() in industry_str.lower() for pref in self.icp_weights["preferred_industries"]):
                industry_score = self.icp_weights["industry_weight"]
                positive_factors.append({"factor": f"Industry match: {industry_str}", "impact": industry_score})
            elif industry_str and industry_str.lower() not in ["unknown", "other", "n/a", ""]:
                industry_score = self.icp_weights["industry_weight"] * 0.4
            else:
                negative_factors.append({"factor": "Industry not specified", "impact": -5})
        else:
            negative_factors.append({"factor": "Missing industry data", "impact": -5})

        score_components.append(industry_score)

        # =================================================================
        # 2. JOB TITLE SCORING (0-15 points)
        # =================================================================
        title = self._get_value(row, col_mappings, "title")
        title_score = 0

        if title:
            title_str = str(title).strip()
            title_upper = title_str.upper()

            # C-level gets maximum
            if any(t in title_upper for t in ["CEO", "CTO", "CFO", "COO", "CMO", "CIO", "CHIEF"]):
                title_score = self.icp_weights["title_weight"]
                positive_factors.append({"factor": f"C-level executive: {title_str}", "impact": title_score})
            # VP level
            elif any(t in title_upper for t in ["VP", "VICE PRESIDENT", "HEAD OF"]):
                title_score = self.icp_weights["title_weight"] * 0.85
                positive_factors.append({"factor": f"VP-level decision maker: {title_str}", "impact": int(title_score)})
            # Director level
            elif "DIRECTOR" in title_upper:
                title_score = self.icp_weights["title_weight"] * 0.7
                positive_factors.append({"factor": f"Director-level: {title_str}", "impact": int(title_score)})
            # Manager level
            elif "MANAGER" in title_upper:
                title_score = self.icp_weights["title_weight"] * 0.5
            # Any title is better than none
            else:
                title_score = self.icp_weights["title_weight"] * 0.3
        else:
            negative_factors.append({"factor": "Job title not provided", "impact": -3})

        score_components.append(title_score)

        # =================================================================
        # 3. LEAD SOURCE SCORING (0-10 points)
        # =================================================================
        source = self._get_value(row, col_mappings, "source")
        source_score = 0

        if source:
            source_str = str(source).strip().lower()

            if any(s.lower() in source_str for s in self.icp_weights["high_quality_sources"]):
                source_score = self.icp_weights["source_weight"]
                positive_factors.append({"factor": f"High-quality lead source: {source}", "impact": source_score})
            elif any(s.lower() in source_str for s in self.icp_weights["medium_quality_sources"]):
                source_score = self.icp_weights["source_weight"] * 0.6
            else:
                source_score = self.icp_weights["source_weight"] * 0.3
        else:
            source_score = self.icp_weights["source_weight"] * 0.2

        score_components.append(source_score)

        # =================================================================
        # 4. COMPANY SIZE / FIRMOGRAPHIC SCORING (0-12 points)
        # =================================================================
        company_size = self._get_value(row, col_mappings, "company_size")
        revenue = self._get_value(row, col_mappings, "revenue")
        firmographic_score = 0

        if company_size is not None:
            try:
                emp_count = int(company_size)
                if emp_count >= 1000:
                    firmographic_score = self.icp_weights["company_size_weight"]
                    positive_factors.append(
                        {"factor": f"Enterprise company ({emp_count:,} employees)", "impact": firmographic_score}
                    )
                elif emp_count >= 200:
                    firmographic_score = self.icp_weights["company_size_weight"] * 0.8
                    positive_factors.append(
                        {"factor": f"Mid-market company ({emp_count} employees)", "impact": int(firmographic_score)}
                    )
                elif emp_count >= 50:
                    firmographic_score = self.icp_weights["company_size_weight"] * 0.5
                else:
                    firmographic_score = self.icp_weights["company_size_weight"] * 0.2
                    negative_factors.append({"factor": "Small company size", "impact": -3})
            except (ValueError, TypeError):
                pass

        if revenue is not None and firmographic_score == 0:
            try:
                rev = float(revenue)
                if rev >= 10_000_000:  # $10M+
                    firmographic_score = self.icp_weights["company_size_weight"] * 0.9
                    positive_factors.append(
                        {"factor": f"High revenue company (${rev:,.0f})", "impact": int(firmographic_score)}
                    )
                elif rev >= 1_000_000:
                    firmographic_score = self.icp_weights["company_size_weight"] * 0.5
            except (ValueError, TypeError):
                pass

        score_components.append(firmographic_score)

        # =================================================================
        # 5. DATA COMPLETENESS SCORING (0-13 points)
        # =================================================================
        completeness_score = 0
        fields_present = 0

        if self._get_value(row, col_mappings, "email"):
            fields_present += 1
        if self._get_value(row, col_mappings, "phone"):
            fields_present += 1
        if self._get_value(row, col_mappings, "company"):
            fields_present += 1
        if self._get_value(row, col_mappings, "title"):
            fields_present += 1
        if self._get_value(row, col_mappings, "industry"):
            fields_present += 1

        completeness_pct = fields_present / 5.0
        completeness_score = self.icp_weights["completeness_weight"] * completeness_pct

        if completeness_pct >= 0.8:
            positive_factors.append({"factor": "Complete lead profile", "impact": 5})
        elif completeness_pct < 0.4:
            negative_factors.append({"factor": "Incomplete lead data", "impact": -5})

        score_components.append(completeness_score)

        # =================================================================
        # 6. RECENCY SCORING (0-15 points)
        # =================================================================
        created_date = self._get_value(row, col_mappings, "created_date")
        recency_score = 0

        if created_date is not None:
            try:
                if isinstance(created_date, str):
                    created_dt = pd.to_datetime(created_date)
                else:
                    created_dt = created_date

                days_ago = (datetime.now() - created_dt).days

                if days_ago <= 7:
                    recency_score = self.icp_weights["recency_weight"]
                    positive_factors.append({"factor": "Fresh lead (< 7 days)", "impact": recency_score})
                elif days_ago <= 30:
                    recency_score = self.icp_weights["recency_weight"] * 0.7
                    positive_factors.append({"factor": "Recent lead (< 30 days)", "impact": int(recency_score)})
                elif days_ago <= 90:
                    recency_score = self.icp_weights["recency_weight"] * 0.4
                else:
                    recency_score = self.icp_weights["recency_weight"] * 0.1
                    negative_factors.append({"factor": f"Stale lead ({days_ago} days old)", "impact": -8})
            except Exception:
                recency_score = self.icp_weights["recency_weight"] * 0.3
        else:
            recency_score = self.icp_weights["recency_weight"] * 0.3

        score_components.append(recency_score)

        # =================================================================
        # 7. STATUS-BASED SCORING (bonus)
        # =================================================================
        status = self._get_value(row, col_mappings, "status")
        status_bonus = 0

        if status:
            status_str = str(status).strip().lower()
            if "qualified" in status_str or "working" in status_str:
                status_bonus = 10
                positive_factors.append({"factor": f"Already in active status: {status}", "impact": status_bonus})
            elif "converted" in status_str or "closed" in status_str:
                status_bonus = 5

        score_components.append(status_bonus)

        # =================================================================
        # CALCULATE FINAL SCORE (1-99)
        # =================================================================
        raw_score = sum(score_components)

        # Maximum possible: 15 + 15 + 10 + 12 + 13 + 15 + 10 = 90, plus some buffer
        max_score = 100

        # Normalize to 1-99 range
        final_score = max(1, min(99, int(round(raw_score))))

        # Determine segment
        if final_score >= 80:
            segment = "Hot"
        elif final_score >= 60:
            segment = "Warm"
        elif final_score >= 40:
            segment = "Lukewarm"
        else:
            segment = "Cold"

        return {
            "lead_id": lead_id,
            "name": lead_name,
            "company": self._get_value(row, col_mappings, "company", ""),
            "title": self._get_value(row, col_mappings, "title", ""),
            "score": final_score,
            "segment": segment,
            "positive_factors": positive_factors,
            "negative_factors": negative_factors,
            "score_breakdown": {
                "industry": round(score_components[0], 1),
                "title": round(score_components[1], 1),
                "source": round(score_components[2], 1),
                "firmographic": round(score_components[3], 1),
                "completeness": round(score_components[4], 1),
                "recency": round(score_components[5], 1),
                "status_bonus": round(score_components[6], 1),
            },
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

    def _segment_leads(self, scored_leads: list[dict]) -> dict[str, Any]:
        """Segment leads by score tier."""
        segments = {"Hot": [], "Warm": [], "Lukewarm": [], "Cold": []}

        for lead in scored_leads:
            segments[lead["segment"]].append(lead)

        return {
            "Hot": {
                "count": len(segments["Hot"]),
                "leads": segments["Hot"][:10],  # Top 10 for display
                "avg_score": np.mean([l["score"] for l in segments["Hot"]]) if segments["Hot"] else 0,
            },
            "Warm": {
                "count": len(segments["Warm"]),
                "leads": segments["Warm"][:10],
                "avg_score": np.mean([l["score"] for l in segments["Warm"]]) if segments["Warm"] else 0,
            },
            "Lukewarm": {
                "count": len(segments["Lukewarm"]),
                "leads": segments["Lukewarm"][:10],
                "avg_score": np.mean([l["score"] for l in segments["Lukewarm"]]) if segments["Lukewarm"] else 0,
            },
            "Cold": {
                "count": len(segments["Cold"]),
                "leads": segments["Cold"][:10],
                "avg_score": np.mean([l["score"] for l in segments["Cold"]]) if segments["Cold"] else 0,
            },
        }

    def _count_factors(self, factors: list[str]) -> list[dict]:
        """Count factor frequencies."""
        from collections import Counter

        counts = Counter(factors)
        return [{"factor": factor, "count": count} for factor, count in counts.most_common()]

    def _generate_visualizations(self, scored_leads: list[dict]) -> list[dict]:
        """Generate visualization data for the frontend."""
        graphs = []

        # 1. Score Distribution Histogram
        scores = [l["score"] for l in scored_leads]
        graphs.append(
            {
                "type": "histogram",
                "title": "Lead Score Distribution",
                "x_data": scores,
                "x_label": "Lead Score",
                "bins": 20,
                "colors": ["#6366f1"],
            }
        )

        # 2. Segment Breakdown Pie Chart
        segment_counts = {}
        for lead in scored_leads:
            seg = lead["segment"]
            segment_counts[seg] = segment_counts.get(seg, 0) + 1

        graphs.append(
            {
                "type": "pie_chart",
                "title": "Lead Quality Segments",
                "labels": list(segment_counts.keys()),
                "values": list(segment_counts.values()),
                "colors": ["#ef4444", "#f59e0b", "#eab308", "#6b7280"],  # Hot=red, Warm=orange, etc
            }
        )

        # 3. Top Leads Bar Chart
        top_leads = sorted(scored_leads, key=lambda x: x["score"], reverse=True)[:10]
        graphs.append(
            {
                "type": "bar_chart",
                "title": "Top 10 Leads by Score",
                "x_data": [l["name"][:20] for l in top_leads],
                "y_data": [l["score"] for l in top_leads],
                "x_label": "Lead",
                "y_label": "Score",
                "colors": ["#10b981"],
            }
        )

        return graphs

    def _generate_insights(self, results: dict) -> list[str]:
        """Generate AI insights from scoring results."""
        insights = []
        summary = results["summary"]

        # Overall quality insight
        hot_pct = summary.get("hot_leads_pct", 0)
        if hot_pct >= 20:
            insights.append(f"🔥 {hot_pct}% of leads are Hot quality - excellent pipeline!")
        elif hot_pct >= 10:
            insights.append(f"✅ {hot_pct}% of leads are Hot quality - healthy pipeline.")
        else:
            insights.append(f"⚠️ Only {hot_pct}% of leads are Hot quality - consider lead source optimization.")

        # Average score insight
        avg_score = summary.get("avg_score", 0)
        if avg_score >= 60:
            insights.append(f"📈 Strong average lead score of {avg_score:.0f} indicates good ICP alignment.")
        elif avg_score >= 40:
            insights.append(f"📊 Average lead score is {avg_score:.0f} - room for improvement in targeting.")
        else:
            insights.append(f"📉 Low average score of {avg_score:.0f} - review lead generation sources.")

        # Factor analysis insights
        if "factors_analysis" in results:
            top_pos = results["factors_analysis"].get("top_positive_factors", [])
            if top_pos:
                insights.append(f'💪 Top positive factor: "{top_pos[0]["factor"]}"')

            top_neg = results["factors_analysis"].get("top_negative_factors", [])
            if top_neg:
                insights.append(f'🎯 Area to improve: "{top_neg[0]["factor"]}"')

        # Segment-specific insights
        segments = results.get("segments", {})
        hot_count = segments.get("Hot", {}).get("count", 0)
        if hot_count > 0:
            insights.append(f"🎉 {hot_count} Hot leads ready for immediate outreach!")

        return insights


__all__ = ["LeadScoringEngine"]
