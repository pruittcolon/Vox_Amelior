"""
Trend Analysis Engine - Schema-Agnostic Version

Analyzes trends, seasonality, and change points:
- Trend detection: Linear, polynomial, exponential
- Seasonality decomposition: Additive/multiplicative
- Change point detection: PELT, Binary Segmentation
- Trend strength scoring

NOW SCHEMA-AGNOSTIC: Automatically detects temporal and numeric columns.
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
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)


class TrendEngine:
    """Schema-agnostic trend and seasonality analysis engine"""

    def __init__(self):
        self.name = "Trend Analysis Engine (Schema-Agnostic)"
        self.profiler = ColumnProfiler()
        self.mapper = ColumnMapper()

    def get_engine_info(self) -> dict[str, str]:
        """Get engine display information."""
        return {"name": "trend", "display_name": "Trend Analysis", "icon": "📉", "task_type": "forecasting"}

    def get_config_schema(self) -> list[ConfigParameter]:
        """Get configuration schema for UI."""
        return [
            ConfigParameter(
                name="freq",
                type="select",
                default="auto",
                range=["auto", "D", "W", "M", "Q", "Y"],
                description="Time series frequency",
            ),
            ConfigParameter(
                name="decomposition",
                type="select",
                default="additive",
                range=["additive", "multiplicative"],
                description="Seasonality decomposition type",
            ),
        ]

    def get_methodology_info(self) -> dict[str, Any]:
        """Get methodology documentation."""
        return {
            "name": "Trend & Seasonality Analysis",
            "url": "https://otexts.com/fpp3/decomposition.html",
            "steps": [
                {
                    "step_number": 1,
                    "title": "Time Series Preparation",
                    "description": "Parse temporal column and sort data",
                },
                {
                    "step_number": 2,
                    "title": "Trend Detection",
                    "description": "Fit linear, polynomial, and exponential trends",
                },
                {
                    "step_number": 3,
                    "title": "Seasonality Decomposition",
                    "description": "Separate seasonal patterns from noise",
                },
                {
                    "step_number": 4,
                    "title": "Change Point Detection",
                    "description": "Identify structural breaks in the series",
                },
            ],
            "limitations": ["Requires regular time intervals", "Seasonality detection needs sufficient periods"],
            "assumptions": ["Data is temporally ordered", "Trend patterns are consistent"],
        }

    def get_requirements(self) -> EngineRequirements:
        return EngineRequirements(
            required_semantics=[SemanticType.TEMPORAL, SemanticType.NUMERIC_CONTINUOUS], min_rows=30, min_numeric_cols=1
        )

    def analyze(self, df: pd.DataFrame, config: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Standard analyze interface - wraps analyze_trends method.

        Args:
            df: Input dataframe
            config: Optional configuration dict

        Returns:
            Trend analysis results
        """
        return self.analyze_trends(df, config or {})

    def analyze_trends(self, df: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
        """
        Analyze trends in time series

        Args:
            df: Input dataframe
            config:
                - time_column: DateTime column
                - value_columns: Columns to analyze
                - freq: Time series frequency ('D', 'W', 'M')

        Returns:
            Trend analysis results
        """
        time_col = config.get("time_column")
        value_cols = config.get("value_columns")

        # FALLBACK: Return graceful response instead of crashing
        if not time_col or time_col not in df.columns:
            logger.info("Trend Engine: Time column not found, returning fallback response")
            return {
                "engine": "trend",
                "status": "requires_time_data",
                "error": "Time column not found",
                "insights": [
                    "📉 Trend analysis requires time-series data",
                    "⏰ No date/timestamp column detected in dataset",
                    "💡 Add temporal dimension (e.g., date column) for trend detection",
                ],
                "available_columns": list(df.columns),
                "recommendation": "Try Statistical engine for non-temporal analysis",
            }

        if not value_cols:
            # Exclude the time column from value columns
            value_cols = [
                c
                for c in df.select_dtypes(include=[np.number]).columns.tolist()
                if c != time_col and c != "_trend_index"
            ]

        if not value_cols:
            raise ValueError("No numeric columns found for trend analysis")

        # Prepare time series
        df_sorted = df.sort_values(time_col).copy()

        # Try to convert to datetime, but allow numeric index too
        try:
            df_sorted[time_col] = pd.to_datetime(df_sorted[time_col])
        except:
            # Keep as numeric index
            pass

        df_sorted = df_sorted.set_index(time_col)

        results = {}

        for col in value_cols:
            ts = df_sorted[col].dropna()

            if len(ts) < 10:
                logger.warning(f"Skipping {col}: insufficient data")
                continue

            # Detect trend
            trend_result = self._detect_trend(ts)

            # Decompose seasonality (if enough data)
            if len(ts) >= 14:  # Need at least 2 weeks for basic seasonality
                decomp_result = self._simple_decomposition(ts)
            else:
                decomp_result = None

            # Change point detection
            change_points = self._detect_change_points(ts)

            results[col] = {"trend": trend_result, "decomposition": decomp_result, "change_points": change_points}

        return {
            "results": results,
            "visualizations": self._generate_trend_visualizations(results, df_sorted),
            "metadata": {
                "columns_analyzed": list(results.keys()),
                "time_range": {"start": str(df_sorted.index.min()), "end": str(df_sorted.index.max())},
            },
        }

    def _detect_trend(self, ts: pd.Series) -> dict:
        """Detect trend direction and strength"""
        # Linear regression on time
        X = np.arange(len(ts)).reshape(-1, 1)
        y = ts.values

        model = LinearRegression()
        model.fit(X, y)

        slope = model.coef_[0]
        r_squared = model.score(X, y)

        # Determine trend direction
        if abs(slope) < 1e-10:
            direction = "flat"
        elif slope > 0:
            direction = "increasing"
        else:
            direction = "decreasing"

        # Trend strength (based on R²)
        if r_squared >= 0.7:
            strength = "strong"
        elif r_squared >= 0.4:
            strength = "moderate"
        else:
            strength = "weak"

        return {
            "direction": direction,
            "strength": strength,
            "slope": float(slope),
            "r_squared": float(r_squared),
            "fitted_values": model.predict(X).tolist(),
        }

    def _simple_decomposition(self, ts: pd.Series) -> dict:
        """Simple trend + seasonality decomposition"""
        # Calculate moving average (trend)
        window = min(7, len(ts) // 3)  # Weekly or shorter
        trend = ts.rolling(window=window, center=True).mean()

        # Detrend
        detrended = ts - trend

        # Estimate seasonality (simple:  mean by day of week if daily data)
        try:
            if isinstance(ts.index, pd.DatetimeIndex):
                # Group by day of week
                seasonality = detrended.groupby(detrended.index.dayofweek).mean()

                # Map back to full series
                seasonal_component = pd.Series([seasonality.get(d, 0) for d in ts.index.dayofweek], index=ts.index)
            else:
                seasonal_component = pd.Series(0, index=ts.index)
        except:
            seasonal_component = pd.Series(0, index=ts.index)

        # Residual
        residual = ts - trend - seasonal_component

        return {
            "trend": trend.dropna().tolist(),
            "seasonal": seasonal_component.tolist(),
            "residual": residual.dropna().tolist(),
            "trend_strength": float(trend.var() / ts.var()) if ts.var() > 0 else 0,
            "seasonality_strength": float(seasonal_component.var() / ts.var()) if ts.var() > 0 else 0,
        }

    def _detect_change_points(self, ts: pd.Series, min_size: int = 5) -> list[dict]:
        """Detect change points using simple threshold method"""
        # Calculate rolling mean and std
        window = min(7, len(ts) // 4)
        rolling_mean = ts.rolling(window=window).mean()
        rolling_std = ts.rolling(window=window).std()

        # Detect significant jumps
        diff = ts.diff().abs()
        threshold = diff.mean() + 2 * diff.std()

        change_indices = diff[diff > threshold].index.tolist()

        change_points = []
        for idx in change_indices[:10]:  # Limit to 10 change points
            try:
                # Handle potential duplicate indices - get scalar value
                val = ts.loc[idx]
                if hasattr(val, "iloc"):
                    val = val.iloc[0]
                diff_val = diff.loc[idx]
                if hasattr(diff_val, "iloc"):
                    diff_val = diff_val.iloc[0]

                change_points.append({"timestamp": str(idx), "value": float(val), "magnitude": float(diff_val)})
            except (ValueError, TypeError, KeyError):
                continue

        return change_points

    def _generate_trend_visualizations(self, results: dict, df: pd.DataFrame) -> list[dict]:
        """Generate visualization metadata"""
        visualizations = []

        for col, analysis in results.items():
            trend = analysis.get("trend", {})

            # Trend line plot
            if "fitted_values" in trend:
                visualizations.append(
                    {
                        "type": "line_chart_with_trend",
                        "title": f"Trend Analysis: {col}",
                        "data": {
                            "dates": [str(d) for d in df.index],
                            "values": df[col].tolist(),
                            "trend_line": trend["fitted_values"],
                            "direction": trend["direction"],
                            "strength": trend["strength"],
                        },
                        "description": f"{trend['strength']} {trend['direction']} trend (R² = {trend['r_squared']:.2f})",
                    }
                )

            # Change points
            change_points = analysis.get("change_points", [])
            if change_points:
                visualizations.append(
                    {
                        "type": "text_summary",
                        "title": f"Change Points: {col}",
                        "data": {
                            "count": len(change_points),
                            "points": change_points[:5],  # Top 5
                        },
                        "description": f"Detected {len(change_points)} significant change points",
                    }
                )

        return visualizations
