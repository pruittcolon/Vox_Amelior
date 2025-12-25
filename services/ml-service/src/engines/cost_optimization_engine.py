"""
Cost Optimization Engine - Schema-Agnostic Version

Identifies cost-saving opportunities, waste detection, and resource optimization.
Provides Pareto analysis, efficiency scoring, and supplier consolidation recommendations.

NOW SCHEMA-AGNOSTIC: Works with any dataset structure via semantic column mapping.
"""

import logging
import os

# Import Schema Intelligence Layer
import sys
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import ColumnMapper, ColumnProfiler, DatasetClassifier, EngineRequirements, SemanticType
from core.premium_models import ConfigParameter

logger = logging.getLogger(__name__)


class CostOptimizationEngine:
    """
    Schema-agnostic cost optimization analysis engine.

    Automatically detects cost and grouping columns using semantic intelligence,
    enabling analysis on ANY dataset with numerical cost data.
    """

    def __init__(self):
        self.name = "Cost Optimization Engine (Schema-Agnostic)"
        self.profiler = ColumnProfiler()
        self.mapper = ColumnMapper()

    def get_engine_info(self) -> dict[str, str]:
        """Get engine display information."""
        return {
            "name": "cost_optimization",
            "display_name": "Cost Optimization",
            "icon": "💰",
            "task_type": "detection",
        }

    def get_config_schema(self) -> list[ConfigParameter]:
        """Get configuration schema for UI."""
        return [
            ConfigParameter(
                name="cost_column", type="select", default=None, range=[], description="Column containing cost values"
            ),
            ConfigParameter(
                name="category_column", type="select", default=None, range=[], description="Column for grouping costs"
            ),
        ]

    def get_methodology_info(self) -> dict[str, Any]:
        """Get methodology documentation."""
        return {
            "name": "Cost Optimization Analysis",
            "url": None,
            "steps": [
                {
                    "step_number": 1,
                    "title": "Cost Detection",
                    "description": "Identify cost columns via semantic intelligence",
                },
                {
                    "step_number": 2,
                    "title": "Pareto Analysis",
                    "description": "Apply 80/20 rule to identify major cost drivers",
                },
                {"step_number": 3, "title": "Efficiency Scoring", "description": "Calculate cost efficiency metrics"},
                {
                    "step_number": 4,
                    "title": "Optimization Recommendations",
                    "description": "Generate actionable cost reduction suggestions",
                },
            ],
            "limitations": ["Requires accurate cost data", "Historical patterns may not predict future"],
            "assumptions": ["Cost data is comprehensive", "Categories are meaningful groupings"],
        }

    def get_requirements(self) -> EngineRequirements:
        """
        Define semantic requirements for this engine.

        Returns:
            EngineRequirements specifying what columns are needed
        """
        return EngineRequirements(
            required_semantics=[SemanticType.NUMERIC_CONTINUOUS],
            optional_semantics={"grouping": [SemanticType.CATEGORICAL], "temporal": [SemanticType.TEMPORAL]},
            required_entities=[],  # Flexible - can work with any numeric cost-like data
            preferred_domains=[],  # Universal
            applicable_tasks=[],
            min_rows=20,
            min_numeric_cols=1,
        )

    def analyze(self, df: pd.DataFrame, config: dict | None = None) -> dict[str, Any]:
        """
        Run schema-agnostic cost optimization analysis.

        Args:
            df: DataFrame with any structure containing numeric cost-like data
            config: Optional configuration dict with:
                - cost_column: Hint for cost column (optional)
                - category_column: Hint for grouping column (optional)
                - skip_profiling: If True, skip schema intelligence (default: False)

        Returns:
            Dict with analysis results, column mappings, and visualizations
        """
        config = config or {}

        results = {
            "summary": {},
            "opportunities": [],
            "graphs": [],
            "insights": [],
            "column_mappings": {},
            "profiling_used": not config.get("skip_profiling", False),
        }

        try:
            # SCHEMA INTELLIGENCE: Profile dataset
            if not config.get("skip_profiling", False):
                logger.info("Profiling dataset for schema-agnostic analysis")
                profiles = self.profiler.profile_dataset(df)

                # Classify dataset
                classifier = DatasetClassifier()
                classification = classifier.classify(profiles, len(df))
                results["summary"]["detected_domain"] = classification.domain.value
                results["summary"]["detected_tasks"] = [t.value for t in classification.task_types]

                # SMART COLUMN MAPPING
                requirements = self.get_requirements()
                mapping_result = self.mapper.map_columns(profiles, requirements)

                if not mapping_result.success:
                    results["error"] = f"Cannot analyze: {mapping_result.message}"
                    results["missing_requirements"] = mapping_result.missing_required
                    return results

                # UNIVERSAL COLUMN SELECTION: Use statistical properties, not just keywords
                cost_col = mapping_result.mappings.get("numeric_continuous")
                if cost_col:
                    cost_col = cost_col.column_name

                    # Score all numeric columns using data-driven approach
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    col_scores = {}

                    for col in numeric_cols:
                        score = self._score_column_for_optimization(df, col)
                        col_scores[col] = score

                    # Pick the best column
                    if col_scores:
                        best_col = max(col_scores.keys(), key=lambda k: col_scores[k])
                        cost_col = best_col

                    results["column_mappings"]["cost"] = {
                        "column": cost_col,
                        "confidence": mapping_result.mappings["numeric_continuous"].confidence,
                        "semantic_type": mapping_result.mappings["numeric_continuous"].semantic_type.value,
                        "selection_score": col_scores.get(cost_col, 0),
                    }

                category_col = mapping_result.mappings.get("grouping")
                if category_col:
                    category_col = category_col.column_name
                    results["column_mappings"]["grouping"] = {
                        "column": category_col,
                        "confidence": mapping_result.mappings["grouping"].confidence,
                        "semantic_type": mapping_result.mappings["grouping"].semantic_type.value,
                    }
                else:
                    category_col = None

                results["summary"]["mapping_confidence"] = mapping_result.confidence

            else:
                # Fallback: use hints or old detection
                cost_col = config.get("cost_column") or self._detect_cost_column(df, None)
                category_col = config.get("category_column") or self._detect_category_column(df, None)
                results["profiling_used"] = False

            # FLEXIBLE METRIC SELECTION: Prefer cost columns, but accept any numeric metric
            if cost_col:
                # Check if it's a true cost column
                cost_keywords = [
                    "cost",
                    "price",
                    "amount",
                    "expense",
                    "spend",
                    "payment",
                    "charge",
                    "fee",
                    "revenue",
                    "sales",
                    "profit",
                ]
                col_name_lower = cost_col.lower()
                is_cost_col = any(keyword in col_name_lower for keyword in cost_keywords)

                if is_cost_col:
                    metric_type = "financial"
                    metric_description = "cost/financial"
                else:
                    # It's not a cost column, but we can still analyze it as a general metric
                    # Determine if it's something to minimize or maximize based on column name
                    minimize_keywords = ["error", "defect", "fault", "failure", "loss", "waste", "risk", "affair"]
                    maximize_keywords = ["quality", "satisfaction", "rating", "score", "performance", "efficiency"]

                    if any(kw in col_name_lower for kw in minimize_keywords):
                        metric_type = "minimize"
                        metric_description = "outcome to minimize"
                    elif any(kw in col_name_lower for kw in maximize_keywords):
                        metric_type = "maximize"
                        metric_description = "outcome to maximize"
                    else:
                        metric_type = "neutral"
                        metric_description = "numeric metric"

                results["summary"]["metric_type"] = metric_type
                results["summary"]["metric_description"] = metric_description
                results["summary"]["analysis_column"] = cost_col

            if not cost_col:
                results["error"] = "No numeric column suitable for cost analysis found"
                return results

            # ANALYSIS PIPELINE (same as before, but now schema-agnostic)
            # Total cost analysis
            total_cost = df[cost_col].sum()
            results["summary"]["total_cost"] = float(total_cost)
            results["summary"]["row_count"] = len(df)
            results["summary"]["avg_cost"] = float(df[cost_col].mean())

            # Pareto analysis (80/20 rule)
            if category_col:
                pareto_results = self._pareto_analysis(df, cost_col, category_col)
                results["pareto"] = pareto_results
                results["graphs"].append(pareto_results["graph"])

                # === ADD: cost_categories for treemap visualization ===
                try:
                    category_costs = df.groupby(category_col)[cost_col].sum()
                    results["cost_categories"] = {str(cat): float(val) for cat, val in category_costs.items()}
                except Exception as e:
                    logger.warning(f"Could not build cost_categories: {e}")
                    results["cost_categories"] = {}

                # Identify waste (low-value high-cost items)
                waste = self._identify_waste(df, cost_col, category_col)
                results["waste"] = waste
                results["opportunities"].extend(waste["opportunities"])

            # Cost distribution analysis
            cost_dist = self._cost_distribution(df, cost_col)
            results["distribution"] = cost_dist
            results["graphs"].append(cost_dist["graph"])

            # Outlier detection (unusually high costs)
            outliers = self._detect_cost_outliers(df, cost_col, category_col)
            results["outliers"] = outliers
            results["opportunities"].extend(outliers["opportunities"])
            results["graphs"].append(outliers["graph"])

            # Generate insights
            results["insights"] = self._generate_insights(results, cost_col, category_col)

        except Exception as e:
            logger.error(f"Cost optimization analysis failed: {e}", exc_info=True)
            results["error"] = str(e)

        return results

    def _score_column_for_optimization(self, df: pd.DataFrame, col_name: str) -> float:
        """
        Score a column's suitability for optimization using statistical properties.

        This is UNIVERSAL - works for ANY dataset, not just financial data.
        Uses: coefficient of variation, skewness, zero-inflation, outliers, etc.
        """
        col = df[col_name]
        score = 0.0

        try:
            # 1. Coefficient of Variation (max 30 points)
            if col.std() > 0 and col.mean() != 0:
                cv = abs(col.std() / col.mean())
                score += min(cv * 10, 30)

            # 2. Zero-inflation (20 points if 20-90% zeros)
            zero_pct = (col == 0).sum() / len(col)
            if 0.2 <= zero_pct <= 0.9:
                score += 20
            elif zero_pct > 0.9:
                score += 5  # Too sparse

            # 3. Skewness (15 points if highly skewed)
            skew = abs(col.skew())
            if skew > 2:
                score += 15
            elif skew > 1:
                score += 10

            # 4. Outliers (15 points if >5% outliers)
            Q1, Q3 = col.quantile([0.25, 0.75])
            IQR = Q3 - Q1
            if IQR > 0:
                outlier_mask = (col < Q1 - 1.5 * IQR) | (col > Q3 + 1.5 * IQR)
                if outlier_mask.sum() / len(col) > 0.05:
                    score += 15

            # 5. Integer counts (10 points)
            if col.dtype in ["int64", "int32"] and col.min() >= 0:
                score += 10

            # 6. Semantic keywords (10 point bonus)
            opt_keywords = [
                "cost",
                "price",
                "error",
                "defect",
                "failure",
                "loss",
                "incident",
                "complaint",
                "violation",
                "affair",
                "revenue",
                "sales",
                "profit",
                "expense",
                "spend",
            ]
            if any(kw in col_name.lower() for kw in opt_keywords):
                score += 10

            # PENALTIES
            # ID columns (very high uniqueness)
            if col.nunique() / len(col) > 0.95:
                score -= 50

            # Low-variance demographics
            demographic_kw = ["age", "gender", "id", "name", "date", "time", "zip"]
            if any(kw in col_name.lower() for kw in demographic_kw):
                cv = abs(col.std() / col.mean()) if col.mean() != 0 else 0
                if cv < 0.5:
                    score -= 20

        except Exception as e:
            # If scoring fails, return 0
            logger.debug(f"Column scoring failed for {col_name}: {e}")
            return 0.0

        return score

    def _detect_cost_column(self, df: pd.DataFrame, hint: str | None = None) -> str | None:
        """Auto-detect cost/price column"""
        if hint and hint in df.columns:
            return hint

        cost_keywords = ["cost", "price", "amount", "expense", "spend", "payment"]
        for col in df.columns:
            if any(keyword in col.lower() for keyword in cost_keywords):
                if pd.api.types.is_numeric_dtype(df[col]):
                    return col
        return None

    def _detect_category_column(self, df: pd.DataFrame, hint: str | None = None) -> str | None:
        """Auto-detect category column"""
        if hint and hint in df.columns:
            return hint

        cat_keywords = ["category", "type", "department", "supplier", "vendor", "item"]
        for col in df.columns:
            if any(keyword in col.lower() for keyword in cat_keywords):
                # Check if column has reasonable number of unique values
                if df[col].nunique() < len(df) * 0.5:
                    return col
        return None

    def _pareto_analysis(self, df: pd.DataFrame, cost_col: str, category_col: str) -> dict:
        """Perform 80/20 Pareto analysis"""
        # Aggregate costs by category
        category_costs = df.groupby(category_col)[cost_col].sum().sort_values(ascending=False)
        total_cost = category_costs.sum()

        # Calculate cumulative percentage
        cumulative_pct = (category_costs.cumsum() / total_cost * 100).values

        # Find 80% threshold
        threshold_idx = np.where(cumulative_pct >= 80)[0]
        vital_few = threshold_idx[0] + 1 if len(threshold_idx) > 0 else len(category_costs)

        # Generate Pareto chart data
        graph = {
            "type": "pareto",
            "title": "Cost Pareto Analysis (80/20 Rule)",
            "x_data": category_costs.index.tolist(),
            "y_data": category_costs.values.tolist(),
            "cumulative": cumulative_pct.tolist(),
            "vital_few": vital_few,
            "description": f"{vital_few} categories account for 80% of total costs",
        }

        return {
            "vital_few_count": vital_few,
            "vital_few_categories": category_costs.head(vital_few).index.tolist(),
            "vital_few_cost": float(category_costs.head(vital_few).sum()),
            "vital_few_percentage": float(category_costs.head(vital_few).sum() / total_cost * 100),
            "graph": graph,
        }

    def _identify_waste(self, df: pd.DataFrame, cost_col: str, category_col: str) -> dict:
        """Identify wasteful spending patterns"""
        opportunities = []

        # Find categories with high cost but low frequency
        category_stats = df.groupby(category_col).agg({cost_col: ["sum", "count", "mean", "std"]}).reset_index()
        category_stats.columns = ["category", "total_cost", "frequency", "avg_cost", "std_cost"]

        # High cost, low frequency = potential waste
        total_cost = category_stats["total_cost"].sum()
        category_stats["cost_pct"] = category_stats["total_cost"] / total_cost * 100

        waste_candidates = category_stats[
            (category_stats["cost_pct"] > 5)  # Significant cost
            & (category_stats["frequency"] < category_stats["frequency"].median())  # Low frequency
        ]

        for _, row in waste_candidates.iterrows():
            opportunities.append(
                {
                    "type": "waste_reduction",
                    "category": row["category"],
                    "current_cost": float(row["total_cost"]),
                    "frequency": int(row["frequency"]),
                    "recommendation": f"Review {row['category']} - {row['cost_pct']:.1f}% of costs with low usage",
                    "potential_savings": float(row["total_cost"] * 0.2),  # Estimate 20% savings
                }
            )

        return {"waste_categories": len(waste_candidates), "opportunities": opportunities}

    def _cost_distribution(self, df: pd.DataFrame, cost_col: str) -> dict:
        """Analyze cost distribution"""
        quartiles = df[cost_col].quantile([0.25, 0.5, 0.75]).values

        graph = {
            "type": "histogram",
            "title": "Cost Distribution",
            "x_data": df[cost_col].values.tolist(),
            "bins": 30,
            "q1": float(quartiles[0]),
            "median": float(quartiles[1]),
            "q3": float(quartiles[2]),
        }

        return {
            "mean": float(df[cost_col].mean()),
            "median": float(df[cost_col].median()),
            "std": float(df[cost_col].std()),
            "min": float(df[cost_col].min()),
            "max": float(df[cost_col].max()),
            "quartiles": quartiles.tolist(),
            "graph": graph,
        }

    def _detect_cost_outliers(self, df: pd.DataFrame, cost_col: str, category_col: str | None) -> dict:
        """Detect unusually high costs"""
        # Use IQR method
        Q1 = df[cost_col].quantile(0.25)
        Q3 = df[cost_col].quantile(0.75)
        IQR = Q3 - Q1

        outlier_threshold = Q3 + 1.5 * IQR
        outliers = df[df[cost_col] > outlier_threshold]

        opportunities = []
        for _, row in outliers.head(10).iterrows():
            opportunities.append(
                {
                    "type": "outlier_review",
                    "cost": float(row[cost_col]),
                    "category": row[category_col] if category_col else "Unknown",
                    "recommendation": f"Review unusually high cost: ${row[cost_col]:,.2f}",
                    "potential_savings": float((row[cost_col] - df[cost_col].median()) * 0.5),
                }
            )

        graph = {
            "type": "box_plot",
            "title": "Cost Outlier Detection",
            "data": df[cost_col].values.tolist(),
            "outliers": outliers[cost_col].values.tolist(),
            "threshold": float(outlier_threshold),
        }

        return {
            "outlier_count": len(outliers),
            "outlier_total": float(outliers[cost_col].sum()),
            "opportunities": opportunities,
            "graph": graph,
        }

    def _utilization_analysis(self, df: pd.DataFrame, cost_col: str) -> dict:
        """Analyze resource utilization efficiency"""
        # Find quantity-related columns
        qty_col = None
        for col in df.columns:
            if any(kw in col.lower() for kw in ["quantity", "volume", "units", "count"]):
                qty_col = col
                break

        if qty_col and pd.api.types.is_numeric_dtype(df[qty_col]):
            df["unit_cost"] = df[cost_col] / df[qty_col].replace(0, np.nan)

            graph = {
                "type": "scatter",
                "title": "Cost vs Quantity (Efficiency Analysis)",
                "x_data": df[qty_col].values.tolist(),
                "y_data": df[cost_col].values.tolist(),
                "x_label": qty_col,
                "y_label": cost_col,
            }

            return {"avg_unit_cost": float(df["unit_cost"].mean()), "graph": graph}

        return {}

    def _generate_insights(self, results: dict, cost_col: str = None, category_col: str = None) -> list[str]:
        """Generate actionable insights from analysis"""
        insights = []

        # Determine metric type and adapt language
        metric_type = results.get("summary", {}).get("metric_type", "neutral")
        metric_desc = results.get("summary", {}).get("metric_description", "numeric metric")

        # Schema-aware insight (mention which columns were used)
        if cost_col:
            if metric_type == "financial":
                insights.append(f"📊 Analyzed cost data from column '{cost_col}'")
            elif metric_type == "minimize":
                insights.append(f"📊 Analyzed '{cost_col}' ({metric_desc}) to identify reduction opportunities")
            elif metric_type == "maximize":
                insights.append(f"📊 Analyzed '{cost_col}' ({metric_desc}) to identify improvement opportunities")
            else:
                insights.append(f"📊 Analyzed numeric metric '{cost_col}' for optimization patterns")

        if category_col:
            insights.append(f"🏷️ Grouped by '{category_col}' for detailed breakdown")

        if "pareto" in results:
            vital_few = results["pareto"]["vital_few_count"]
            vital_pct = results["pareto"]["vital_few_percentage"]

            if metric_type == "financial":
                insights.append(
                    f"💡 Focus on top {vital_few} categories - they represent {vital_pct:.1f}% of total costs (Pareto Principle)"
                )
            elif metric_type == "minimize":
                insights.append(
                    f"💡 Focus on top {vital_few} categories - they account for {vital_pct:.1f}% of total occurrences (Pareto Principle)"
                )
            elif metric_type == "maximize":
                insights.append(
                    f"💡 Top {vital_few} categories account for {vital_pct:.1f}% of total value (Pareto Principle)"
                )

        if "waste" in results:
            waste = results["waste"]
            total_waste = waste.get("total_waste_opportunities", len(waste.get("opportunities", [])))
            if total_waste > 0:
                if metric_type == "financial":
                    insights.append(f"⚠️ Found {total_waste} waste reduction opportunities")
                elif metric_type == "minimize":
                    insights.append(f"⚠️ Found {total_waste} high-impact reduction opportunities")
                else:
                    insights.append(f"⚠️ Found {total_waste} optimization opportunities")

        if "outliers" in results:
            outliers = results["outliers"]
            if outliers["outlier_count"] > 0:
                if metric_type == "financial":
                    insights.append(
                        f"🔍 Found {outliers['outlier_count']} cost outliers totaling "
                        f"${outliers['outlier_total']:,.2f} - recommend manual review"
                    )
                elif metric_type == "minimize":
                    insights.append(
                        f"🔍 Found {outliers['outlier_count']} unusual high values totaling "
                        f"{outliers['outlier_total']:.0f} - these warrant investigation"
                    )
                else:
                    insights.append(f"🔍 Found {outliers['outlier_count']} outlier values - recommend review")

        # Calculate total potential savings/improvement
        total_savings = sum(opp["potential_savings"] for opp in results.get("opportunities", []))
        if total_savings > 0:
            total_metric = results["summary"].get("total_cost", 1)
            pct = total_savings / total_metric * 100 if total_metric > 0 else 0

            if metric_type == "financial":
                insights.append(f"💰 Potential annual savings: ${total_savings:,.2f} ({pct:.1f}% of total costs)")
            elif metric_type == "minimize":
                insights.append(f"📉 Potential reduction: {total_savings:.0f} ({pct:.1f}% improvement possible)")
            elif metric_type == "maximize":
                insights.append(f"📈 Potential increase: {total_savings:.0f} ({pct:.1f}% improvement possible)")

        return insights

        if "pareto" in results:
            vital_few = results["pareto"]["vital_few_count"]
            vital_pct = results["pareto"]["vital_few_percentage"]
            insights.append(
                f"💡 Focus on top {vital_few} categories - they represent {vital_pct:.1f}% of total costs (Pareto Principle)"
            )

        if "waste" in results:
            waste = results["waste"]
            total_waste = waste.get("total_waste_opportunities", len(waste.get("opportunities", [])))
            if total_waste > 0:
                insights.append(f"⚠️ Found {total_waste} waste reduction opportunities")

        if "outliers" in results:
            outliers = results["outliers"]
            if outliers["outlier_count"] > 0:
                insights.append(
                    f"🔍 Found {outliers['outlier_count']} cost outliers totaling "
                    f"${outliers['outlier_total']:,.2f} - recommend manual review"
                )

        # Calculate total potential savings
        total_savings = sum(opp["potential_savings"] for opp in results.get("opportunities", []))
        if total_savings > 0:
            insights.append(
                f"💰 Potential annual savings: ${total_savings:,.2f} "
                f"({total_savings / results['summary'].get('total_cost', 1) * 100:.1f}% of total costs)"
            )

        return insights


# Export
__all__ = ["CostOptimizationEngine"]
