"""
Resource Utilization Engine - Schema-Agnostic Version

Analyzes resource usage (CPU, Memory, Staff, Equipment) to identify bottlenecks and underutilization.

NOW SCHEMA-AGNOSTIC: Automatically detects resource, usage, and capacity columns via semantic intelligence.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import logging
from typing import Any

import numpy as np
import pandas as pd
from core import ColumnMapper, ColumnProfiler, EngineRequirements, SemanticType
from core.gemma_summarizer import GemmaSummarizer
from core.premium_models import ConfigParameter

logger = logging.getLogger(__name__)


class ResourceUtilizationEngine:
    """Schema-agnostic resource efficiency and bottleneck analysis engine"""

    def __init__(self):
        self.name = "Resource Utilization Engine (Schema-Agnostic)"
        self.profiler = ColumnProfiler()
        self.mapper = ColumnMapper()

    def get_engine_info(self) -> dict[str, str]:
        return {
            "name": "resource_utilization",
            "display_name": "Resource Utilization",
            "icon": "⚙️",
            "task_type": "detection",
        }

    def get_config_schema(self) -> list[ConfigParameter]:
        return [
            ConfigParameter(
                name="resource_column", type="select", default=None, range=[], description="Resource identifier column"
            ),
            ConfigParameter(
                name="usage_column", type="select", default=None, range=[], description="Usage/utilization column"
            ),
            ConfigParameter(
                name="capacity_column", type="select", default=None, range=[], description="Maximum capacity column"
            ),
        ]

    def get_methodology_info(self) -> dict[str, Any]:
        return {
            "name": "Resource Utilization Analysis",
            "url": None,
            "steps": [
                {"step_number": 1, "title": "Usage Analysis", "description": "Calculate utilization metrics"},
                {
                    "step_number": 2,
                    "title": "Bottleneck Detection",
                    "description": "Identify over/under-utilized resources",
                },
                {"step_number": 3, "title": "Optimization", "description": "Recommend rebalancing actions"},
            ],
            "limitations": ["Requires accurate usage data"],
            "assumptions": ["Capacity limits are known"],
        }

    def get_requirements(self) -> EngineRequirements:
        return EngineRequirements(required_semantics=[SemanticType.NUMERIC_CONTINUOUS], min_rows=10, min_numeric_cols=1)

    def analyze(self, df: pd.DataFrame, config: dict | None = None) -> dict[str, Any]:
        """
        Run utilization analysis

        Args:
            df: DataFrame with usage data
            config: Configuration with optional keys:
                - resource_column: Resource Name/ID
                - usage_column: Usage value (e.g., %)
                - capacity_column: Max capacity (optional)
                - time_column: Timestamp (optional)

        Returns:
            Dict with utilization analysis
        """
        config = config or {}

        results = {"summary": {}, "bottlenecks": [], "underutilized": [], "graphs": [], "insights": []}

        try:
            # Auto-detect columns
            res_col = self._detect_column(
                df, ["resource", "server", "machine", "employee"], config.get("resource_column")
            )
            usage_col = self._detect_column(df, ["usage", "load", "cpu", "memory", "hours"], config.get("usage_column"))
            cap_col = self._detect_column(df, ["capacity", "limit", "max"], config.get("capacity_column"))
            time_col = self._detect_column(df, ["time", "date", "timestamp"], config.get("time_column"))

            if res_col and usage_col:
                # Normalize usage if capacity provided
                if cap_col:
                    df["utilization_pct"] = (df[usage_col] / df[cap_col] * 100).replace([np.inf, -np.inf], 0)
                    metric_col = "utilization_pct"
                else:
                    # Assume usage_col is already percentage or absolute value to be analyzed as is
                    metric_col = usage_col

                # Average utilization by resource
                avg_util = df.groupby(res_col)[metric_col].mean().sort_values(ascending=False)

                # Identify Bottlenecks (> 80%)
                bottlenecks = avg_util[avg_util > 80]
                for res, val in bottlenecks.items():
                    results["bottlenecks"].append({"resource": res, "utilization": float(val)})

                # Identify Underutilized (< 20%)
                underutilized = avg_util[avg_util < 20]
                for res, val in underutilized.items():
                    results["underutilized"].append({"resource": res, "utilization": float(val)})

                # Summary
                results["summary"] = {
                    "avg_utilization": float(avg_util.mean()),
                    "peak_utilization": float(df[metric_col].max()),
                    "bottleneck_count": len(bottlenecks),
                    "underutilized_count": len(underutilized),
                }

                # Visualizations

                # 1. Heatmap (if time column exists)
                if time_col:
                    heatmap = self._utilization_heatmap(df, res_col, time_col, metric_col)
                    results["graphs"].append(heatmap["graph"])
                else:
                    # Bar chart of top utilized resources
                    bar = {
                        "type": "bar_chart",
                        "title": "Resource Utilization",
                        "x_data": avg_util.head(10).index.tolist(),
                        "y_data": avg_util.head(10).values.tolist(),
                        "x_label": "Resource",
                        "y_label": "Utilization (%)",
                        "colors": ["#ef4444" if v > 80 else "#3b82f6" for v in avg_util.head(10)],
                    }
                    results["graphs"].append(bar)

                # Generate insights
                results["insights"] = self._generate_insights(results)

            else:
                # FALLBACK: Compute utilization from any two numeric columns as a ratio
                # This allows the engine to work with ANY dataset
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
                
                if len(numeric_cols) >= 2 and len(cat_cols) >= 1:
                    # Try to find a meaningful ratio (e.g., students/teachers, usage/capacity)
                    # Pick first categorical as grouping, first two numeric as ratio
                    group_col = cat_cols[0]
                    num1, num2 = numeric_cols[0], numeric_cols[1]
                    
                    # Calculate ratio (smaller/larger for a percentage-like value)
                    df["_computed_util"] = (df[num1] / df[num2].replace(0, np.nan) * 100).fillna(0)
                    
                    # Cap at 100 if it looks like a percentage
                    if df["_computed_util"].max() > 200:
                        # Normalize to 0-100
                        df["_computed_util"] = (df["_computed_util"] / df["_computed_util"].max()) * 100
                    
                    avg_util = df.groupby(group_col)["_computed_util"].mean().sort_values(ascending=False)
                    
                    results["summary"] = {
                        "avg_utilization": float(avg_util.mean()),
                        "peak_utilization": float(df["_computed_util"].max()),
                        "computed_from": f"{num1}/{num2} grouped by {group_col}",
                    }
                    
                    # Bar chart
                    bar = {
                        "type": "bar_chart",
                        "title": f"Computed Utilization ({num1}/{num2})",
                        "x_data": avg_util.head(10).index.tolist(),
                        "y_data": avg_util.head(10).values.tolist(),
                        "x_label": group_col,
                        "y_label": "Computed Util (%)",
                    }
                    results["graphs"].append(bar)
                    results["insights"].append(f"📊 Computed utilization from {num1}/{num2} by {group_col}")
                else:
                    # True fallback - no usable columns
                    return GemmaSummarizer.generate_fallback_summary(
                        df,
                        engine_name="resource_utilization",
                        error_reason="Could not detect resource and usage columns. Expected columns like 'resource', 'usage', 'capacity'.",
                        config=config,
                    )

        except Exception as e:
            logger.error(f"Resource utilization analysis failed: {e}")
            # Use Gemma fallback on error
            return GemmaSummarizer.generate_fallback_summary(
                df, engine_name="resource_utilization", error_reason=str(e), config=config
            )


        return results

    def _detect_column(self, df: pd.DataFrame, keywords: list[str], hint: str | None = None) -> str | None:
        if hint and hint in df.columns:
            return hint
        for col in df.columns:
            if any(kw in col.lower() for kw in keywords):
                return col
        return None

    def _utilization_heatmap(self, df: pd.DataFrame, res_col: str, time_col: str, metric_col: str) -> dict:
        """Generate heatmap data"""
        # Pivot table: Index=Resource, Columns=Time (aggregated)
        # Simplify time to Hour or Day if needed
        try:
            df[time_col] = pd.to_datetime(df[time_col])
            # If too many time points, resample?
            # For now, just take top 10 resources and recent time
            top_res = df[res_col].unique()[:10]
            subset = df[df[res_col].isin(top_res)].copy()

            # Create a matrix for heatmap
            pivot = subset.pivot_table(index=res_col, columns=time_col, values=metric_col, aggfunc="mean")

            graph = {
                "type": "heatmap",
                "title": "Utilization Heatmap",
                "x_data": [str(c) for c in pivot.columns],
                "y_data": pivot.index.tolist(),
                "z_data": pivot.values.tolist(),
                "min": 0,
                "max": 100,
            }
            return {"graph": graph}
        except:
            # Fallback
            return {"graph": {}}

    def _generate_insights(self, results: dict) -> list[str]:
        """Generate insights"""
        insights = []

        if results["bottlenecks"]:
            top = results["bottlenecks"][0]
            insights.append(
                f"🔥 Critical Bottleneck: '{top['resource']}' is running at {top['utilization']:.1f}% capacity"
            )

        if results["underutilized"]:
            insights.append(
                f"💤 Efficiency Opportunity: {len(results['underutilized'])} resources are underutilized (<20%)"
            )

        return insights


__all__ = ["ResourceUtilizationEngine"]
