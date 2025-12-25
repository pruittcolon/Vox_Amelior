"""
Budget Variance Engine - Schema-Agnostic Version

Analyzes budget vs actual spending, identifies root causes of variance,
and forecasts end-of-period performance.

NOW SCHEMA-AGNOSTIC: Works with any dataset structure via semantic column mapping.
"""

import logging
import os
import sys
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import ColumnMapper, ColumnProfiler, DatasetClassifier, EngineRequirements, SemanticType
from core.gemma_summarizer import GemmaSummarizer
from core.premium_models import ConfigParameter

logger = logging.getLogger(__name__)


class BudgetVarianceEngine:
    """
    Schema-agnostic budget variance analysis engine.

    Automatically detects budget, actual, and category columns.
    """

    def __init__(self):
        self.name = "Budget Variance Engine (Schema-Agnostic)"
        self.profiler = ColumnProfiler()
        self.mapper = ColumnMapper()

    def get_engine_info(self) -> dict[str, str]:
        """Get engine display information."""
        return {
            "name": "budget_variance",
            "display_name": "Budget Variance Analysis",
            "icon": "📋",
            "task_type": "detection",
        }

    def get_config_schema(self) -> list[ConfigParameter]:
        """Get configuration schema for UI."""
        return [
            ConfigParameter(
                name="budget_column",
                type="select",
                default=None,
                range=[],
                description="Column containing budget values",
            ),
            ConfigParameter(
                name="actual_column",
                type="select",
                default=None,
                range=[],
                description="Column containing actual values",
            ),
            ConfigParameter(
                name="category_column",
                type="select",
                default=None,
                range=[],
                description="Column for budget categories",
            ),
        ]

    def get_methodology_info(self) -> dict[str, Any]:
        """Get methodology documentation."""
        return {
            "name": "Budget Variance Analysis",
            "url": "https://en.wikipedia.org/wiki/Variance_analysis",
            "steps": [
                {"step_number": 1, "title": "Column Detection", "description": "Identify budget and actual columns"},
                {
                    "step_number": 2,
                    "title": "Variance Calculation",
                    "description": "Calculate absolute and percentage variances",
                },
                {
                    "step_number": 3,
                    "title": "Root Cause Analysis",
                    "description": "Identify drivers of budget variance",
                },
                {"step_number": 4, "title": "Forecasting", "description": "Project end-of-period performance"},
            ],
            "limitations": ["Requires both budget and actual data", "Historical patterns inform forecasts"],
            "assumptions": ["Budget and actual data are comparable", "Categories are consistent"],
        }

    def get_requirements(self) -> EngineRequirements:
        """
        Define semantic requirements for budget variance analysis.

        Returns:
            EngineRequirements for dual numeric columns
        """
        return EngineRequirements(
            required_semantics=[SemanticType.NUMERIC_CONTINUOUS],
            optional_semantics={
                "budget": [SemanticType.NUMERIC_CONTINUOUS],
                "actual": [SemanticType.NUMERIC_CONTINUOUS],
                "category": [SemanticType.CATEGORICAL],
            },
            required_entities=[],
            preferred_domains=[],
            applicable_tasks=[],
            min_rows=5,
            min_numeric_cols=2,
        )

    def analyze(self, df: pd.DataFrame, config: dict | None = None) -> dict[str, Any]:
        """
        Run schema-agnostic budget variance analysis.

        Args:
            df: DataFrame with any structure containing budget/actual data
            config: Optional configuration with:
                - budget_column: Hint for budget column
                - actual_column: Hint for actual column
                - category_column: Hint for category column
                - skip_profiling: Skip schema intelligence (default: False)

        Returns:
            Dict with variance analysis and visualizations
        """
        config = config or {}

        results = {
            "summary": {},
            "variances": [],
            "graphs": [],
            "insights": [],
            "column_mappings": {},
            "profiling_used": not config.get("skip_profiling", False),
        }

        try:
            # SCHEMA INTELLIGENCE: Profile and detect columns
            if not config.get("skip_profiling", False):
                logger.info("Profiling dataset for budget variance analysis")
                profiles = self.profiler.profile_dataset(df)

                classifier = DatasetClassifier()
                classification = classifier.classify(profiles, len(df))
                results["summary"]["detected_domain"] = classification.domain.value

                # SMART DETECTION: Dual numeric columns
                budget_col = self._smart_detect(
                    df, profiles, keywords=["budget", "planned", "target", "forecast"], hint=config.get("budget_column")
                )

                actual_col = self._smart_detect(
                    df,
                    profiles,
                    keywords=["actual", "spend", "cost", "realized", "expense"],
                    hint=config.get("actual_column"),
                )

                category_col = self._smart_detect(
                    df,
                    profiles,
                    keywords=["category", "department", "project", "segment"],
                    hint=config.get("category_column"),
                    semantic_type=SemanticType.CATEGORICAL,
                )

                if budget_col:
                    results["column_mappings"]["budget"] = {"column": budget_col, "confidence": 0.9}
                if actual_col:
                    results["column_mappings"]["actual"] = {"column": actual_col, "confidence": 0.9}
                if category_col:
                    results["column_mappings"]["category"] = {"column": category_col, "confidence": 0.85}
            else:
                budget_col = config.get("budget_column") or self._detect_column(
                    df, ["budget", "planned", "target"], None
                )
                actual_col = config.get("actual_column") or self._detect_column(df, ["actual", "spend", "cost"], None)
                category_col = config.get("category_column") or self._detect_column(
                    df, ["category", "department"], None
                )
                results["profiling_used"] = False

            if not (budget_col and actual_col):
                # Use Gemma fallback when required columns are missing
                missing = []
                if not budget_col:
                    missing.append("budget/planned")
                if not actual_col:
                    missing.append("actual/spend")
                return GemmaSummarizer.generate_fallback_summary(
                    df,
                    engine_name="budget_variance",
                    error_reason=f"Could not detect {' and '.join(missing)} columns. Expected columns with 'budget', 'planned', 'actual', or 'spend' in name.",
                    config=config,
                )

            # ANALYSIS PIPELINE (schema-agnostic)
            total_budget = df[budget_col].sum()
            total_actual = df[actual_col].sum()
            total_variance = total_actual - total_budget
            variance_pct = (total_variance / total_budget * 100) if total_budget != 0 else 0

            results["summary"].update(
                {
                    "budget_column": budget_col,
                    "actual_column": actual_col,
                    "total_budget": float(total_budget),
                    "total_actual": float(total_actual),
                    "total_variance": float(total_variance),
                    "variance_percentage": float(variance_pct),
                    "status": "Over Budget" if total_variance > 0 else "Under Budget",
                }
            )

            # Variance by category
            if category_col:
                cat_variance = self._category_variance(df, budget_col, actual_col, category_col)
                results["variances"] = cat_variance["variances"]
                results["graphs"].append(cat_variance["graph"])
                results["graphs"].append(cat_variance["waterfall_graph"])

            # Trend analysis
            comparison = self._comparison_chart(df, budget_col, actual_col, category_col)
            results["graphs"].append(comparison["graph"])

            # Generate insights
            results["insights"] = self._generate_insights(results)

        except Exception as e:
            logger.error(f"Budget variance analysis failed: {e}", exc_info=True)
            results["error"] = str(e)

        return results

    def _smart_detect(
        self,
        df: pd.DataFrame,
        profiles: dict,
        keywords: list[str],
        hint: str | None = None,
        semantic_type: SemanticType | None = None,
    ) -> str | None:
        """Smart column detection using profiles + keywords."""
        if hint and hint in df.columns:
            return hint

        # Priority 1: Keyword match with correct semantic type
        for col in df.columns:
            if any(kw in col.lower() for kw in keywords):
                prof = profiles.get(col)
                if prof:
                    if semantic_type:
                        if prof.semantic_type == semantic_type:
                            return col
                    elif prof.semantic_type in [SemanticType.NUMERIC_CONTINUOUS, SemanticType.NUMERIC_DISCRETE]:
                        return col

        # Priority 2: Semantic type match
        if semantic_type:
            for col, prof in profiles.items():
                if prof.semantic_type == semantic_type:
                    return col

        return None

    def _detect_column(self, df: pd.DataFrame, keywords: list[str], hint: str | None = None) -> str | None:
        if hint and hint in df.columns:
            return hint
        for col in df.columns:
            if any(kw in col.lower() for kw in keywords):
                return col
        return None

    def _category_variance(self, df: pd.DataFrame, budget_col: str, actual_col: str, category_col: str) -> dict:
        """Calculate variance by category"""
        # Group by category
        grouped = df.groupby(category_col)[[budget_col, actual_col]].sum().reset_index()
        grouped["variance"] = grouped[actual_col] - grouped[budget_col]
        grouped["variance_pct"] = (grouped["variance"] / grouped[budget_col] * 100).replace([np.inf, -np.inf], 0)

        # Sort by absolute variance to find biggest drivers
        grouped["abs_variance"] = grouped["variance"].abs()
        grouped = grouped.sort_values("abs_variance", ascending=False)

        variances = []
        for _, row in grouped.head(10).iterrows():
            variances.append(
                {
                    "category": row[category_col],
                    "budget": float(row[budget_col]),
                    "actual": float(row[actual_col]),
                    "variance": float(row["variance"]),
                    "variance_pct": float(row["variance_pct"]),
                }
            )

        # Bar chart for top variances
        graph = {
            "type": "bar_chart",
            "title": "Top Budget Variances by Category",
            "x_data": grouped.head(10)[category_col].tolist(),
            "y_data": grouped.head(10)["variance"].tolist(),
            "x_label": "Category",
            "y_label": "Variance ($)",
            "colors": ["#ef4444" if v > 0 else "#10b981" for v in grouped.head(10)["variance"]],
        }

        # Waterfall chart data
        # Start with total budget, then add variances to reach total actual
        waterfall_x = ["Total Budget"] + grouped.head(5)[category_col].tolist() + ["Others", "Total Actual"]

        top_5_variance = grouped.head(5)["variance"].sum()
        total_variance = grouped["variance"].sum()
        others_variance = total_variance - top_5_variance

        waterfall_y = (
            [grouped[budget_col].sum()]
            + grouped.head(5)["variance"].tolist()
            + [others_variance, grouped[actual_col].sum()]
        )
        waterfall_types = ["absolute"] + ["relative"] * 6 + ["absolute"]

        waterfall_graph = {
            "type": "waterfall",
            "title": "Budget to Actual Walk",
            "x_data": waterfall_x,
            "y_data": waterfall_y,
            "measure": waterfall_types,
        }

        return {"variances": variances, "graph": graph, "waterfall_graph": waterfall_graph}

    def _comparison_chart(self, df: pd.DataFrame, budget_col: str, actual_col: str, category_col: str | None) -> dict:
        """Budget vs Actual comparison"""
        # If too many categories, aggregate to top 10
        if category_col and df[category_col].nunique() > 10:
            grouped = df.groupby(category_col)[[budget_col, actual_col]].sum()
            grouped["total"] = grouped[budget_col] + grouped[actual_col]
            grouped = grouped.sort_values("total", ascending=False).head(10)
            x_data = grouped.index.tolist()
            budget_data = grouped[budget_col].tolist()
            actual_data = grouped[actual_col].tolist()
        else:
            x_data = df[category_col].tolist() if category_col else list(range(len(df)))
            budget_data = df[budget_col].tolist()
            actual_data = df[actual_col].tolist()

        graph = {
            "type": "bar_chart_grouped",
            "title": "Budget vs Actual Comparison",
            "x_data": x_data,
            "datasets": [
                {"label": "Budget", "data": budget_data, "color": "#94a3b8"},
                {"label": "Actual", "data": actual_data, "color": "#3b82f6"},
            ],
        }

        return {"graph": graph}

    def _generate_insights(self, results: dict) -> list[str]:
        """Generate insights"""
        insights = []

        summary = results["summary"]
        if summary["total_variance"] > 0:
            insights.append(
                f"⚠️ Over Budget by ${summary['total_variance']:,.2f} ({summary['variance_percentage']:.1f}%)"
            )
        else:
            insights.append(
                f"✅ Under Budget by ${abs(summary['total_variance']):,.2f} "
                f"({abs(summary['variance_percentage']):.1f}%)"
            )

        if results["variances"]:
            top_var = results["variances"][0]
            insights.append(f"🔍 Largest driver: {top_var['category']} with ${top_var['variance']:,.2f} variance")

        return insights


__all__ = ["BudgetVarianceEngine"]
