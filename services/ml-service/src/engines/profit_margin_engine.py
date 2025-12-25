"""
Profit Margin Engine - Schema-Agnostic Version

Analyzes profit margins by product, customer, and region.
Identifies margin dilution and profitable growth opportunities.

NOW SCHEMA-AGNOSTIC: Automatically detects revenue, cost, and profit columns via semantic intelligence.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import logging
from typing import Any

import numpy as np
import pandas as pd
from core import ColumnMapper, ColumnProfiler, EngineRequirements, SemanticType
from core.premium_models import ConfigParameter

logger = logging.getLogger(__name__)


class ProfitMarginEngine:
    """Schema-agnostic profit margin and profitability analysis engine"""

    def __init__(self):
        self.name = "Profit Margin Engine (Schema-Agnostic)"
        self.profiler = ColumnProfiler()
        self.mapper = ColumnMapper()

    def get_engine_info(self) -> dict[str, str]:
        return {
            "name": "profit_margins",
            "display_name": "Profit Margin Analysis",
            "icon": "💵",
            "task_type": "detection",
        }

    def get_config_schema(self) -> list[ConfigParameter]:
        return [
            ConfigParameter(
                name="revenue_column", type="select", default=None, range=[], description="Column containing revenue"
            ),
            ConfigParameter(
                name="cost_column", type="select", default=None, range=[], description="Column containing costs"
            ),
        ]

    def get_methodology_info(self) -> dict[str, Any]:
        return {
            "name": "Profit Margin Analysis",
            "url": None,
            "steps": [
                {"step_number": 1, "title": "Column Detection", "description": "Identify revenue and cost columns"},
                {"step_number": 2, "title": "Margin Calculation", "description": "Calculate gross and net margins"},
                {"step_number": 3, "title": "Trend Analysis", "description": "Analyze margin trends over time"},
            ],
            "limitations": ["Requires accurate revenue and cost data"],
            "assumptions": ["Data represents complete financial picture"],
        }

    def get_requirements(self) -> EngineRequirements:
        """Define semantic requirements for profit margin analysis."""
        return EngineRequirements(
            required_semantics=[SemanticType.NUMERIC_CONTINUOUS],
            optional_semantics={},
            required_entities=[],
            preferred_domains=[],
            applicable_tasks=[],
            min_rows=10,
            min_numeric_cols=2,
        )

    def analyze(self, df: pd.DataFrame, config: dict | None = None) -> dict[str, Any]:
        """
        Run schema-agnostic profit margin analysis.

        Args:
            df: DataFrame with revenue and cost/profit data
            config: Configuration with optional keys:
                - revenue_column: Hint for revenue column
                - cost_column: Hint for cost column
                - profit_column: Hint for profit column
                - product_column: Hint for product grouping
                - customer_column: Hint for customer grouping
                - skip_profiling: Skip schema intelligence (default: False)

        Returns:
            Dict with margin analysis, visualizations, and column mappings
        """
        config = config or {}

        results = {
            "summary": {},
            "segments": {},
            "graphs": [],
            "insights": [],
            "column_mappings": {},
            "profiling_used": not config.get("skip_profiling", False),
        }

        try:
            # SCHEMA INTELLIGENCE: Profile dataset
            if not config.get("skip_profiling", False):
                profiles = self.profiler.profile_dataset(df)

                # Detect revenue, cost, profit columns
                rev_col = self._smart_detect(
                    df,
                    profiles,
                    keywords=["revenue", "sales", "income", "total"],
                    hint=config.get("revenue_column"),
                    semantic_type=SemanticType.NUMERIC_CONTINUOUS,
                )

                cost_col = self._smart_detect(
                    df,
                    profiles,
                    keywords=["cost", "cogs", "expense", "spend"],
                    hint=config.get("cost_column"),
                    semantic_type=SemanticType.NUMERIC_CONTINUOUS,
                )

                profit_col = self._smart_detect(
                    df,
                    profiles,
                    keywords=["profit", "margin", "net", "earnings"],
                    hint=config.get("profit_column"),
                    semantic_type=SemanticType.NUMERIC_CONTINUOUS,
                )

                prod_col = self._smart_detect(
                    df,
                    profiles,
                    keywords=["product", "item", "sku", "article"],
                    hint=config.get("product_column"),
                    semantic_type=SemanticType.CATEGORICAL,
                )

                cust_col = self._smart_detect(
                    df,
                    profiles,
                    keywords=["customer", "client", "account", "user"],
                    hint=config.get("customer_column"),
                    semantic_type=SemanticType.CATEGORICAL,
                )

                # Store mappings
                if rev_col:
                    results["column_mappings"]["revenue"] = {"column": rev_col, "confidence": 0.9}
                if cost_col:
                    results["column_mappings"]["cost"] = {"column": cost_col, "confidence": 0.9}
                if profit_col:
                    results["column_mappings"]["profit"] = {"column": profit_col, "confidence": 0.9}
            else:
                # Fallback: use hints or old detection
                rev_col = config.get("revenue_column") or self._detect_column(df, ["revenue", "sales"], None)
                cost_col = config.get("cost_column") or self._detect_column(df, ["cost", "cogs"], None)
                profit_col = config.get("profit_column") or self._detect_column(df, ["profit", "margin"], None)
                prod_col = config.get("product_column") or self._detect_column(df, ["product", "item"], None)
                cust_col = config.get("customer_column") or self._detect_column(df, ["customer", "client"], None)

            # Calculate profit if missing
            if not profit_col and rev_col and cost_col:
                df["calculated_profit"] = df[rev_col] - df[cost_col]
                profit_col = "calculated_profit"

            if rev_col and profit_col:
                # Calculate margin %
                df["margin_pct"] = (df[profit_col] / df[rev_col] * 100).replace([np.inf, -np.inf], 0)

                # Overall Summary
                total_rev = df[rev_col].sum()
                total_profit = df[profit_col].sum()
                avg_margin = (total_profit / total_rev * 100) if total_rev != 0 else 0

                results["summary"] = {
                    "total_revenue": float(total_rev),
                    "total_profit": float(total_profit),
                    "average_margin": float(avg_margin),
                    "profitable_transactions": int((df[profit_col] > 0).sum()),
                }

                # Product Profitability
                if prod_col:
                    prod_analysis = self._segment_analysis(df, prod_col, rev_col, profit_col, "Product")
                    results["segments"]["product"] = prod_analysis
                    results["graphs"].append(prod_analysis["graph"])
                    results["graphs"].append(prod_analysis["scatter_graph"])

                # Customer Profitability
                if cust_col:
                    cust_analysis = self._segment_analysis(df, cust_col, rev_col, profit_col, "Customer")
                    results["segments"]["customer"] = cust_analysis
                    results["graphs"].append(cust_analysis["graph"])

                # Margin Distribution
                dist = self._margin_distribution(df, "margin_pct")
                results["graphs"].append(dist["graph"])

                # Generate insights
                results["insights"] = self._generate_insights(results, rev_col, profit_col)

            else:
                results["error"] = "Could not detect revenue and profit/cost columns"

        except Exception as e:
            logger.error(f"Profit margin analysis failed: {e}")
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
        """Smart column detection with semantic type filtering."""
        if hint and hint in df.columns:
            return hint

        # Priority 1: Keyword match + semantic type
        for col in df.columns:
            if any(kw in col.lower() for kw in keywords):
                prof = profiles.get(col)
                if prof and (not semantic_type or prof.semantic_type == semantic_type):
                    return col

        # Priority 2: Semantic type only
        if semantic_type:
            for col, prof in profiles.items():
                if prof.semantic_type == semantic_type:
                    return col

        return None

    def _detect_column(self, df: pd.DataFrame, keywords: list[str], hint: str | None = None) -> str | None:
        """Legacy column detection (fallback only)"""
        if hint and hint in df.columns:
            return hint
        for col in df.columns:
            if any(kw in col.lower() for kw in keywords):
                return col
        return None

    def _segment_analysis(self, df: pd.DataFrame, group_col: str, rev_col: str, profit_col: str, label: str) -> dict:
        """Analyze profitability by segment"""
        grouped = df.groupby(group_col).agg({rev_col: "sum", profit_col: "sum"}).reset_index()

        grouped["margin_pct"] = (grouped[profit_col] / grouped[rev_col] * 100).fillna(0)
        grouped = grouped.sort_values(profit_col, ascending=False)

        # Bar chart: Top 10 by Profit
        graph = {
            "type": "bar_chart",
            "title": f"Top {label}s by Profit",
            "x_data": grouped.head(10)[group_col].tolist(),
            "y_data": grouped.head(10)[profit_col].tolist(),
            "x_label": label,
            "y_label": "Profit ($)",
            "colors": ["#10b981" if p > 0 else "#ef4444" for p in grouped.head(10)[profit_col]],
        }

        # Scatter plot: Revenue vs Margin % (Whale Curve analysis)
        scatter_graph = {
            "type": "scatter",
            "title": f"{label} Profitability Matrix",
            "x_data": grouped[rev_col].tolist(),
            "y_data": grouped["margin_pct"].tolist(),
            "labels": grouped[group_col].tolist(),
            "x_label": "Revenue ($)",
            "y_label": "Margin (%)",
        }

        # Identify unprofitable segments
        unprofitable = grouped[grouped[profit_col] < 0]

        return {
            "top_performers": grouped.head(5)[group_col].tolist(),
            "unprofitable_count": len(unprofitable),
            "unprofitable_loss": float(unprofitable[profit_col].sum()),
            "graph": graph,
            "scatter_graph": scatter_graph,
        }

    def _margin_distribution(self, df: pd.DataFrame, margin_col: str) -> dict:
        """Analyze distribution of margins"""
        # Filter out extreme outliers for better visualization
        q1 = df[margin_col].quantile(0.05)
        q3 = df[margin_col].quantile(0.95)
        filtered = df[(df[margin_col] >= q1) & (df[margin_col] <= q3)]

        graph = {
            "type": "histogram",
            "title": "Margin Distribution",
            "x_data": filtered[margin_col].tolist(),
            "x_label": "Margin (%)",
            "bins": 20,
        }

        return {"graph": graph}

    def _generate_insights(self, results: dict, rev_col: str | None = None, profit_col: str | None = None) -> list[str]:
        """Generate schema-aware insights"""
        insights = []

        # Schema-aware insight
        if rev_col and profit_col:
            insights.append(f"📊 Analyzed profit margins from '{rev_col}' and '{profit_col}' columns")

        summary = results["summary"]
        insights.append(
            f"💰 Average Margin: {summary['average_margin']:.1f}% (Total Profit: ${summary['total_profit']:,.2f})"
        )

        if "product" in results["segments"]:
            prod = results["segments"]["product"]
            if prod["unprofitable_count"] > 0:
                insights.append(
                    f"⚠️ {prod['unprofitable_count']} unprofitable products dragging down profit "
                    f"by ${abs(prod['unprofitable_loss']):,.2f}"
                )

        return insights


__all__ = ["ProfitMarginEngine"]
