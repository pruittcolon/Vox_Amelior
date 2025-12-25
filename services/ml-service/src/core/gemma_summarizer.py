"""
Gemma Summarizer - Lightweight fallback for engines with limited output

When an engine produces insufficient results (e.g., missing required columns),
this module provides a simple Gemma-based summary of the data to give users
actionable insights.
"""

import logging
import os
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)


class GemmaSummarizer:
    """
    Lightweight Gemma integration for generating data summaries
    when specialized engines cannot analyze the data.
    """

    GEMMA_SERVICE_URL = os.getenv("GEMMA_SERVICE_URL", "http://gemma-service:8001")
    TIMEOUT = 30.0

    @classmethod
    def generate_fallback_summary(
        cls, df: pd.DataFrame, engine_name: str, error_reason: str, config: dict | None = None
    ) -> dict[str, Any]:
        """
        Generate a simple statistical summary when the specialized engine fails.
        Falls back to basic pandas analysis when Gemma is unavailable.

        Args:
            df: The input DataFrame
            engine_name: Name of the engine that failed
            error_reason: Why the engine couldn't process the data
            config: Optional configuration

        Returns:
            Dict with summary, insights, and recommendations
        """
        # First, generate basic stats that we can always provide
        basic_summary = cls._generate_basic_stats(df)

        # Try to get Gemma summary if available
        gemma_summary = cls._try_gemma_summary(df, engine_name, error_reason)

        return {
            "summary": {"engine": engine_name, "status": "fallback_summary", "reason": error_reason, **basic_summary},
            "insights": gemma_summary.get("insights", cls._generate_basic_insights(df, engine_name)),
            "recommendations": gemma_summary.get(
                "recommendations",
                [
                    {
                        "action": "Review column names",
                        "reason": f"The {engine_name} engine expects specific column patterns. Check documentation for required columns.",
                    }
                ],
            ),
            "graphs": cls._generate_fallback_graphs(df),
            "fallback_used": True,
            "gemma_enhanced": gemma_summary.get("success", False),
        }

    @classmethod
    def _generate_basic_stats(cls, df: pd.DataFrame) -> dict[str, Any]:
        """Generate basic descriptive statistics."""
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        stats = {
            "row_count": len(df),
            "column_count": len(df.columns),
            "numeric_columns": len(numeric_cols),
            "categorical_columns": len(categorical_cols),
            "missing_values": int(df.isnull().sum().sum()),
        }

        if numeric_cols:
            stats["numeric_summary"] = {
                col: {
                    "mean": float(df[col].mean()) if pd.notna(df[col].mean()) else None,
                    "std": float(df[col].std()) if pd.notna(df[col].std()) else None,
                    "min": float(df[col].min()) if pd.notna(df[col].min()) else None,
                    "max": float(df[col].max()) if pd.notna(df[col].max()) else None,
                }
                for col in numeric_cols[:5]  # Limit to first 5
            }

        if categorical_cols:
            stats["categorical_summary"] = {
                col: {"unique": int(df[col].nunique()), "top_values": df[col].value_counts().head(3).to_dict()}
                for col in categorical_cols[:3]  # Limit to first 3
            }

        return stats

    @classmethod
    def _generate_basic_insights(cls, df: pd.DataFrame, engine_name: str) -> list[str]:
        """Generate basic insights without Gemma."""
        insights = []

        # Data overview
        insights.append(f"📊 Dataset contains {len(df)} rows and {len(df.columns)} columns")

        # Numeric column insights
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if numeric_cols:
            for col in numeric_cols[:3]:
                mean_val = df[col].mean()
                if pd.notna(mean_val):
                    insights.append(f"📈 Average {col}: {mean_val:,.2f}")

        # Missing data
        missing = df.isnull().sum().sum()
        if missing > 0:
            insights.append(f"⚠️ {missing} missing values detected across the dataset")

        # Column suggestions based on engine type
        column_hints = cls._get_column_hints(engine_name, df.columns.tolist())
        if column_hints:
            insights.append(f"💡 {column_hints}")

        return insights

    @classmethod
    def _get_column_hints(cls, engine_name: str, columns: list[str]) -> str | None:
        """Suggest which columns might work for this engine."""
        engine_requirements = {
            "resource_utilization": {
                "required": ["resource", "usage", "capacity"],
                "hint": "Try renaming columns to include 'resource', 'usage', 'capacity', or 'utilization'",
            },
            "rag_evaluation": {
                "required": ["query", "retrieved_docs", "generated_answer"],
                "hint": "RAG evaluation needs 'query', 'retrieved_docs', 'relevant_docs', and 'generated_answer' columns",
            },
            "inventory_optimization": {
                "required": ["product", "demand", "cost", "stock"],
                "hint": "Try including columns with 'product', 'demand', 'stock', or 'inventory' in the name",
            },
            "cash_flow": {
                "required": ["date", "amount", "inflow", "outflow"],
                "hint": "Include a date column and columns with 'inflow', 'outflow', or 'amount'",
            },
        }

        if engine_name in engine_requirements:
            return engine_requirements[engine_name]["hint"]
        return None

    @classmethod
    def _generate_fallback_graphs(cls, df: pd.DataFrame) -> list[dict[str, Any]]:
        """Generate simple fallback visualizations."""
        graphs = []

        # Column type distribution
        numeric_count = len(df.select_dtypes(include=["number"]).columns)
        categorical_count = len(df.select_dtypes(include=["object", "category"]).columns)
        datetime_count = len(df.select_dtypes(include=["datetime64"]).columns)

        graphs.append(
            {
                "type": "pie_chart",
                "title": "Column Type Distribution",
                "labels": ["Numeric", "Categorical", "DateTime"],
                "values": [numeric_count, categorical_count, datetime_count],
                "colors": ["#3b82f6", "#10b981", "#f59e0b"],
            }
        )

        # Top numeric column distribution (if available)
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if numeric_cols:
            col = numeric_cols[0]
            try:
                hist_data = df[col].dropna()
                if len(hist_data) > 0:
                    graphs.append(
                        {
                            "type": "histogram",
                            "title": f"Distribution of {col}",
                            "x_data": hist_data.tolist()[:100],  # Limit data points
                            "bins": 20,
                        }
                    )
            except Exception:
                pass

        return graphs

    @classmethod
    def _try_gemma_summary(cls, df: pd.DataFrame, engine_name: str, error_reason: str) -> dict[str, Any]:
        """
        Attempt to get a Gemma-enhanced summary.
        Returns empty dict with success=False if unavailable.
        """
        try:
            # Prepare a compact data description for Gemma
            data_description = cls._prepare_data_description(df)

            prompt = f"""Briefly analyze this dataset that was meant for {engine_name} analysis but couldn't be processed because: {error_reason}

Data Overview:
{data_description}

Provide:
1. 2-3 key observations about this data
2. 1-2 actionable recommendations

Keep response under 150 words. Be direct and specific."""

            # Try to call Gemma service
            response = requests.post(
                f"{cls.GEMMA_SERVICE_URL}/generate",
                json={"prompt": prompt, "max_tokens": 200, "temperature": 0.3},
                timeout=cls.TIMEOUT,
            )

            if response.status_code == 200:
                result = response.json()
                text = result.get("response", result.get("text", ""))

                # Parse Gemma response into insights
                insights = [line.strip() for line in text.split("\n") if line.strip() and len(line.strip()) > 10]

                return {
                    "success": True,
                    "insights": insights[:4],  # Max 4 insights
                    "recommendations": [
                        {"action": "Follow Gemma suggestions", "reason": text[:200] if len(text) > 200 else text}
                    ],
                }
        except requests.exceptions.ConnectionError:
            logger.debug("Gemma service unavailable - using basic summary")
        except requests.exceptions.Timeout:
            logger.debug("Gemma service timeout - using basic summary")
        except Exception as e:
            logger.debug(f"Gemma summary failed: {e}")

        return {"success": False}

    @classmethod
    def _prepare_data_description(cls, df: pd.DataFrame) -> str:
        """Prepare a compact description of the data for Gemma."""
        lines = []
        lines.append(f"- {len(df)} rows, {len(df.columns)} columns")
        lines.append(f"- Columns: {', '.join(df.columns[:10])}")

        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if numeric_cols:
            col = numeric_cols[0]
            lines.append(
                f"- Sample numeric ({col}): mean={df[col].mean():.2f}, range=[{df[col].min():.2f}, {df[col].max():.2f}]"
            )

        return "\n".join(lines)


def needs_gemma_fallback(result: dict[str, Any], threshold: float = 0.3) -> bool:
    """
    Check if an engine result needs Gemma fallback.

    Args:
        result: Engine output dict
        threshold: Minimum population ratio (0-1)

    Returns:
        True if fallback is recommended
    """
    if not result or "error" in result:
        return True

    # Count populated fields
    key_fields = ["insights", "summary", "recommendations", "metrics", "graphs", "analysis"]
    populated = 0
    total = 0

    for key in key_fields:
        if key in result:
            total += 1
            val = result[key]
            if val and val != [] and val != {}:
                populated += 1

    if total == 0:
        return True

    return (populated / total) < threshold


__all__ = ["GemmaSummarizer", "needs_gemma_fallback"]
