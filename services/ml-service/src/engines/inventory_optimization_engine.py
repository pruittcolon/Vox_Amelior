"""
Inventory Optimization Engine - Schema-Agnostic Version

Optimizes inventory levels, calculates reorder points, identifies slow-moving stock.

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


class InventoryOptimizationEngine:
    """
    Schema-agnostic inventory optimization engine.

    Automatically detects stock level, demand, and product columns.
    """

    def __init__(self):
        self.name = "Inventory Optimization Engine (Schema-Agnostic)"
        self.profiler = ColumnProfiler()
        self.mapper = ColumnMapper()

    def get_engine_info(self) -> dict[str, str]:
        return {
            "name": "inventory_optimization",
            "display_name": "Inventory Optimization",
            "icon": "📦",
            "task_type": "prediction",
        }

    def get_config_schema(self) -> list[ConfigParameter]:
        return [
            ConfigParameter(
                name="stock_column", type="select", default=None, range=[], description="Column containing stock levels"
            ),
            ConfigParameter(
                name="demand_column", type="select", default=None, range=[], description="Column containing demand data"
            ),
        ]

    def get_methodology_info(self) -> dict[str, Any]:
        return {
            "name": "Inventory Optimization",
            "url": "https://en.wikipedia.org/wiki/Economic_order_quantity",
            "steps": [
                {"step_number": 1, "title": "Demand Analysis", "description": "Analyze demand patterns"},
                {"step_number": 2, "title": "EOQ Calculation", "description": "Calculate economic order quantity"},
                {"step_number": 3, "title": "Reorder Points", "description": "Determine optimal reorder points"},
            ],
            "limitations": ["Assumes stable demand patterns"],
            "assumptions": ["Lead times are predictable"],
        }

    def get_requirements(self) -> EngineRequirements:
        return EngineRequirements(
            required_semantics=[SemanticType.NUMERIC_CONTINUOUS],
            optional_semantics={
                "stock": [SemanticType.NUMERIC_CONTINUOUS],
                "demand": [SemanticType.NUMERIC_CONTINUOUS],
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
        Run schema-agnostic inventory optimization.

        Args:
            df: DataFrame with inventory data
            config: Optional configuration

        Returns:
            Dict with optimization results
        """
        config = config or {}
        holding_cost_pct = config.get("holding_cost_pct", 0.2)
        ordering_cost = config.get("ordering_cost", 50)
        lead_time = config.get("lead_time", 7)

        results = {
            "summary": {},
            "recommendations": [],
            "graphs": [],
            "insights": [],
            "column_mappings": {},
            "profiling_used": not config.get("skip_profiling", False),
        }

        try:
            # SCHEMA INTELLIGENCE
            if not config.get("skip_profiling", False):
                logger.info("Profiling dataset for inventory optimization")
                profiles = self.profiler.profile_dataset(df)

                classifier = DatasetClassifier()
                classification = classifier.classify(profiles, len(df))
                results["summary"]["detected_domain"] = classification.domain.value

                prod_col = self._smart_detect(
                    df,
                    profiles,
                    keywords=["product", "sku", "item", "material"],
                    hint=config.get("product_column"),
                    semantic_type=SemanticType.CATEGORICAL,
                )

                demand_col = self._smart_detect(
                    df,
                    profiles,
                    keywords=["demand", "sales", "usage", "quantity", "volume"],
                    hint=config.get("demand_column"),
                )

                cost_col = self._smart_detect(
                    df, profiles, keywords=["cost", "price", "unit_cost", "value"], hint=config.get("cost_column")
                )

                stock_col = self._smart_detect(
                    df, profiles, keywords=["stock", "inventory", "on_hand", "qty"], hint=config.get("stock_column")
                )

                if prod_col:
                    results["column_mappings"]["product"] = {"column": prod_col, "confidence": 0.9}
                if demand_col:
                    results["column_mappings"]["demand"] = {"column": demand_col, "confidence": 0.9}
                if cost_col:
                    results["column_mappings"]["cost"] = {"column": cost_col, "confidence": 0.85}
                if stock_col:
                    results["column_mappings"]["stock"] = {"column": stock_col, "confidence": 0.85}
            else:
                prod_col = config.get("product_column") or self._detect_column(df, ["product", "sku"], None)
                demand_col = config.get("demand_column") or self._detect_column(df, ["demand", "sales"], None)
                cost_col = config.get("cost_column") or self._detect_column(df, ["cost", "price"], None)
                stock_col = config.get("stock_column") or self._detect_column(df, ["stock", "inventory"], None)
                results["profiling_used"] = False

            if not (prod_col and demand_col and cost_col):
                # Use Gemma fallback when required columns are missing
                missing = []
                if not prod_col:
                    missing.append("product/SKU")
                if not demand_col:
                    missing.append("demand/sales")
                if not cost_col:
                    missing.append("cost/price")
                return GemmaSummarizer.generate_fallback_summary(
                    df,
                    engine_name="inventory_optimization",
                    error_reason=f"Could not detect {', '.join(missing)} columns. Inventory optimization requires product, demand, and cost columns.",
                    config=config,
                )

            results["summary"]["product_column"] = prod_col
            results["summary"]["demand_column"] = demand_col
            results["summary"]["cost_column"] = cost_col

            # ANALYSIS PIPELINE
            df["holding_cost"] = df[cost_col] * holding_cost_pct
            df["eoq"] = np.sqrt((2 * df[demand_col] * ordering_cost) / df["holding_cost"])

            daily_demand = df[demand_col] / 365
            safety_stock = daily_demand * lead_time * 0.5
            df["rop"] = (daily_demand * lead_time) + safety_stock

            # ABC Analysis
            df["annual_value"] = df[demand_col] * df[cost_col]
            df = df.sort_values("annual_value", ascending=False)
            df["cumulative_value"] = df["annual_value"].cumsum()
            df["cumulative_pct"] = df["cumulative_value"] / df["annual_value"].sum()

            df["abc_class"] = pd.cut(df["cumulative_pct"], bins=[0, 0.8, 0.95, 1.0], labels=["A", "B", "C"])

            # === ADD: abc_analysis for treemap visualization ===
            try:
                abc_analysis = {"classA": {}, "classB": {}, "classC": {}}
                for abc_class in ["A", "B", "C"]:
                    class_df = df[df["abc_class"] == abc_class]
                    if prod_col and len(class_df) > 0:
                        # Get top items by value for each class
                        top_items = class_df.nlargest(10, "annual_value")
                        abc_analysis[f"class{abc_class}"] = {
                            str(row[prod_col]): float(row["annual_value"]) for _, row in top_items.iterrows()
                        }
                results["abc_analysis"] = abc_analysis
            except Exception as e:
                logger.warning(f"Could not build abc_analysis: {e}")
                results["abc_analysis"] = {"classA": {}, "classB": {}, "classC": {}}

            # Summary
            results["summary"].update(
                {
                    "total_inventory_value": float(df["annual_value"].sum()),
                    "total_items": len(df),
                    "avg_turnover": float(df[demand_col].sum() / (df["eoq"].sum() / 2)),
                }
            )

            # Visualizations
            abc_chart = self._abc_chart(df)
            results["graphs"].append(abc_chart["graph"])

            # Stockout Risk
            if stock_col:
                risk_chart = self._stockout_risk(df, stock_col, "rop")
                results["graphs"].append(risk_chart["graph"])

                at_risk = df[df[stock_col] <= df["rop"]]
                for _, row in at_risk.head(5).iterrows():
                    results["recommendations"].append(
                        {
                            "product": row[prod_col],
                            "action": "Reorder Immediately",
                            "reason": f"Stock ({row[stock_col]}) below ROP ({row['rop']:.1f})",
                        }
                    )

            results["insights"] = self._generate_insights(results, df)

        except Exception as e:
            logger.error(f"Inventory optimization failed: {e}", exc_info=True)
        return results

    def _smart_detect(
        self,
        df: pd.DataFrame,
        profiles: dict,
        keywords: list[str],
        hint: str | None = None,
        semantic_type: SemanticType | None = None,
    ) -> str | None:
        """Smart column detection."""
        if hint and hint in df.columns:
            return hint

        # First pass: look for keyword matches with semantic type constraint
        for col in df.columns:
            if any(kw in col.lower() for kw in keywords):
                prof = profiles.get(col)
                if prof:
                    if semantic_type:
                        if prof.semantic_type == semantic_type:
                            return col
                    # For numeric detection, also accept IDENTIFIER if the column is numeric
                    elif prof.semantic_type in [SemanticType.NUMERIC_CONTINUOUS, SemanticType.NUMERIC_DISCRETE]:
                        return col
                    elif prof.semantic_type == SemanticType.IDENTIFIER:
                        # Check if it's actually a numeric column
                        if pd.api.types.is_numeric_dtype(df[col]):
                            return col

        # Second pass: just keyword match without semantic type
        for col in df.columns:
            if any(kw in col.lower() for kw in keywords):
                return col

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

    def _abc_chart(self, df: pd.DataFrame) -> dict:
        """Generate ABC analysis chart"""
        abc_counts = df["abc_class"].value_counts().sort_index()
        abc_values = df.groupby("abc_class", observed=False)["annual_value"].sum()

        graph = {
            "type": "bar_chart_grouped",
            "title": "ABC Inventory Analysis",
            "x_data": ["A Items", "B Items", "C Items"],
            "datasets": [
                {"label": "Item Count", "data": abc_counts.values.tolist(), "color": "#94a3b8"},
                {"label": "Value ($)", "data": abc_values.values.tolist(), "color": "#3b82f6", "yAxisID": "right"},
            ],
        }

        return {"graph": graph}

    def _stockout_risk(self, df: pd.DataFrame, stock_col: str, rop_col: str) -> dict:
        """Analyze stockout risk"""
        df["status"] = np.where(df[stock_col] <= df[rop_col], "Reorder", "OK")
        status_counts = df["status"].value_counts()

        graph = {
            "type": "pie_chart",
            "title": "Stock Status",
            "labels": status_counts.index.tolist(),
            "values": status_counts.values.tolist(),
            "colors": ["#ef4444", "#10b981"],
        }

        return {"graph": graph}

    def _generate_insights(self, results: dict, df: pd.DataFrame) -> list[str]:
        """Generate insights"""
        insights = []

        a_items = df[df["abc_class"] == "A"]
        insights.append(
            f"📦 ABC Analysis: {len(a_items)} 'A' items account for "
            f"{a_items['cumulative_pct'].max() * 100:.1f}% of inventory value"
        )

        if results["recommendations"]:
            insights.append(f"⚠️ Action Required: {len(results['recommendations'])} items are below reorder point")

        return insights


__all__ = ["InventoryOptimizationEngine"]
