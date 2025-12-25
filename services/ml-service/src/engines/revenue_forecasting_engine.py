"""
Revenue Forecasting Engine - Schema-Agnostic Version

Predicts future revenue using time-series forecasting (Prophet/ARIMA style).
Handles seasonality, trends, and growth scenarios.

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
from core.premium_models import ConfigParameter

logger = logging.getLogger(__name__)


class RevenueForecastingEngine:
    """
    Schema-agnostic revenue forecasting engine.

    Automatically detects revenue and temporal columns for time-series forecasting.
    """

    def __init__(self):
        self.name = "Revenue Forecasting Engine (Schema-Agnostic)"
        self.profiler = ColumnProfiler()
        self.mapper = ColumnMapper()

    def get_engine_info(self) -> dict[str, str]:
        return {
            "name": "revenue_forecasting",
            "display_name": "Revenue Forecasting",
            "icon": "📈",
            "task_type": "forecasting",
        }

    def get_config_schema(self) -> list[ConfigParameter]:
        return [
            ConfigParameter(
                name="revenue_column", type="select", default=None, range=[], description="Column containing revenue"
            ),
            ConfigParameter(
                name="date_column", type="select", default=None, range=[], description="Column containing dates"
            ),
            ConfigParameter(
                name="forecast_periods", type="int", default=12, range=[1, 52], description="Periods to forecast ahead"
            ),
        ]

    def get_methodology_info(self) -> dict[str, Any]:
        return {
            "name": "Revenue Forecasting",
            "url": "https://otexts.com/fpp3/",
            "steps": [
                {"step_number": 1, "title": "Time Series Prep", "description": "Prepare temporal data for forecasting"},
                {"step_number": 2, "title": "Trend Detection", "description": "Identify growth patterns"},
                {"step_number": 3, "title": "Forecasting", "description": "Generate revenue predictions"},
            ],
            "limitations": ["Past performance may not predict future"],
            "assumptions": ["Business conditions remain stable"],
        }

    def get_requirements(self) -> EngineRequirements:
        """
        Define semantic requirements for revenue forecasting.

        Returns:
            EngineRequirements for time-series forecasting
        """
        return EngineRequirements(
            required_semantics=[SemanticType.TEMPORAL, SemanticType.NUMERIC_CONTINUOUS],
            optional_semantics={},
            required_entities=[],
            preferred_domains=[],
            applicable_tasks=[],
            min_rows=12,  # At least 1 year
            min_numeric_cols=1,
        )

    def analyze(self, df: pd.DataFrame, config: dict | None = None) -> dict[str, Any]:
        """
        Run schema-agnostic revenue forecasting.

        Args:
            df: DataFrame with any structure containing time-series revenue data
            config: Optional configuration with:
                - date_column: Hint for date column
                - revenue_column: Hint for revenue column
                - horizon: Number of periods to forecast (default: 12)
                - skip_profiling: Skip schema intelligence (default: False)

        Returns:
            Dict with forecast and visualizations
        """
        config = config or {}
        horizon = config.get("horizon", 12)

        results = {
            "summary": {},
            "forecast": [],
            "graphs": [],
            "insights": [],
            "column_mappings": {},
            "profiling_used": not config.get("skip_profiling", False),
        }

        try:
            # SCHEMA INTELLIGENCE: Profile and detect
            if not config.get("skip_profiling", False):
                logger.info("Profiling dataset for revenue forecasting")
                profiles = self.profiler.profile_dataset(df)

                classifier = DatasetClassifier()
                classification = classifier.classify(profiles, len(df))
                results["summary"]["detected_domain"] = classification.domain.value

                date_col = self._smart_detect(
                    df,
                    profiles,
                    keywords=["date", "time", "period", "timestamp"],
                    hint=config.get("date_column"),
                    semantic_type=SemanticType.TEMPORAL,
                )

                rev_col = self._smart_detect(
                    df, profiles, keywords=["revenue", "sales", "amount", "income"], hint=config.get("revenue_column")
                )

                if date_col:
                    results["column_mappings"]["temporal"] = {"column": date_col, "confidence": 0.95}
                if rev_col:
                    results["column_mappings"]["revenue"] = {"column": rev_col, "confidence": 0.9}
            else:
                date_col = config.get("date_column") or self._detect_column(df, ["date", "time", "period"], None)
                rev_col = config.get("revenue_column") or self._detect_column(df, ["revenue", "sales", "amount"], None)
                results["profiling_used"] = False

            if not (date_col and rev_col):
                # Try harder to find columns
                if not date_col:
                    # Look for any date-like column
                    for col in df.columns:
                        if "date" in col.lower() or "time" in col.lower():
                            date_col = col
                            break
                if not rev_col:
                    # Look for any numeric column that could be revenue
                    for col in df.columns:
                        if col.lower() in ["weekly_sales", "sales", "revenue", "amount", "total", "value"]:
                            rev_col = col
                            break
                    # If still not found, use first large numeric column
                    if not rev_col:
                        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                        if numeric_cols:
                            # Pick the one with largest values (likely revenue)
                            max_vals = {c: df[c].max() for c in numeric_cols}
                            rev_col = max(max_vals, key=max_vals.get)

            if not (date_col and rev_col):
                results["error"] = "Could not detect date and revenue columns"
                results["missing_requirements"] = []
                if not date_col:
                    results["missing_requirements"].append("date/time column")
                if not rev_col:
                    results["missing_requirements"].append("revenue/sales column")
                return results

            results["summary"]["date_column"] = date_col
            results["summary"]["revenue_column"] = rev_col

            # ANALYSIS PIPELINE - use flexible date parsing
            try:
                df[date_col] = pd.to_datetime(df[date_col], format="mixed", dayfirst=True)
            except:
                # Fallback to infer_datetime_format
                df[date_col] = pd.to_datetime(df[date_col], infer_datetime_format=True)
            df = df.sort_values(date_col)
            df.set_index(date_col, inplace=True)
            monthly_rev = df[rev_col].resample("ME").sum()

            forecast_data = self._holt_winters_forecast(monthly_rev, horizon)
            results["forecast"] = forecast_data
            results["graphs"].append(forecast_data["graph"])

            growth = self._analyze_growth(monthly_rev)
            results["growth"] = growth

            results["insights"] = self._generate_insights(results, date_col, rev_col)

        except Exception as e:
            logger.error(f"Revenue forecasting failed: {e}", exc_info=True)
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
        """Smart column detection."""
        if hint and hint in df.columns:
            return hint

        for col in df.columns:
            if any(kw in col.lower() for kw in keywords):
                prof = profiles.get(col)
                if prof:
                    if semantic_type:
                        if prof.semantic_type == semantic_type:
                            return col
                    elif prof.semantic_type in [SemanticType.NUMERIC_CONTINUOUS, SemanticType.NUMERIC_DISCRETE]:
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

    def _holt_winters_forecast(self, series: pd.Series, horizon: int) -> dict:
        """
        Simple implementation of Holt-Winters forecasting
        """
        # Basic linear regression + seasonality for robustness without heavy statsmodels dependency
        y = series.values
        x = np.arange(len(y))

        # Linear trend
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        trend = p(x)

        # Seasonality (detrended)
        detrended = y - trend
        seasonality = np.zeros(12)
        counts = np.zeros(12)

        for i, val in enumerate(detrended):
            idx = i % 12
            seasonality[idx] += val
            counts[idx] += 1

        seasonality = seasonality / np.maximum(counts, 1)

        # Forecast
        last_x = x[-1]
        future_x = np.arange(last_x + 1, last_x + 1 + horizon)
        future_trend = p(future_x)

        future_seasonality = []
        for i in range(horizon):
            idx = (len(y) + i) % 12
            future_seasonality.append(seasonality[idx])

        forecast = future_trend + np.array(future_seasonality)

        # Confidence intervals (simple std dev based)
        std_resid = np.std(y - (trend + [seasonality[i % 12] for i in range(len(y))]))
        upper = forecast + 1.96 * std_resid
        lower = forecast - 1.96 * std_resid

        # Graph data
        dates = series.index
        future_dates = [dates[-1] + pd.DateOffset(months=i + 1) for i in range(horizon)]

        graph = {
            "type": "forecast",
            "title": "Revenue Forecast (Next 12 Months)",
            "x_data": [d.strftime("%Y-%m") for d in dates] + [d.strftime("%Y-%m") for d in future_dates],
            "y_data": list(y) + list(forecast),
            "actual_len": len(y),
            "lower_bound": [None] * len(y) + list(lower),
            "upper_bound": [None] * len(y) + list(upper),
        }

        return {
            "next_period_forecast": float(forecast[0]),
            "total_forecast_revenue": float(sum(forecast)),
            "graph": graph,
        }

    def _analyze_growth(self, series: pd.Series) -> dict:
        """Analyze historical growth rates"""
        yoy_growth = series.pct_change(12).mean() * 100
        cagr = ((series.iloc[-1] / series.iloc[0]) ** (1 / (len(series) / 12)) - 1) * 100 if len(series) > 12 else 0

        return {"avg_yoy_growth": float(yoy_growth), "cagr": float(cagr)}

    def _generate_insights(self, results: dict, date_col: str = None, rev_col: str = None) -> list[str]:
        """Generate insights mentioning detected columns."""
        insights = []

        if date_col and rev_col:
            insights.append(f"📊 Forecasting '{rev_col}' over time using '{date_col}'")

        if "forecast" in results:
            f = results["forecast"]
            insights.append(f"📈 Forecasted Revenue: ${f['total_forecast_revenue']:,.2f} over next 12 months")

        if "growth" in results:
            g = results["growth"]
            insights.append(f"🚀 Historical Growth: {g['cagr']:.1f}% CAGR")

        return insights


__all__ = ["RevenueForecastingEngine"]
