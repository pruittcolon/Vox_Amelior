"""
Anomaly Detection Engine - Schema-Agnostic Version

Multi-method anomaly detection:
- Statistical: Z-score, Modified Z-score, IQR
- Isolation Forest: Unsupervised outlier detection
- Local Outlier Factor (LOF): Density-based
- DBSCAN: Cluster-based noise detection

Includes ensemble voting and human-in-the-loop validation.
NOW SCHEMA-AGNOSTIC: Automatically detects numeric columns via semantic intelligence.
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
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class AnomalyEngine:
    """Schema-agnostic multi-method anomaly detection with ensemble voting"""

    def __init__(self):
        self.name = "Anomaly Detection Engine (Schema-Agnostic)"
        self.profiler = ColumnProfiler()
        self.mapper = ColumnMapper()
        self.scalers = {}
        self.models = {}

    def get_engine_info(self) -> dict[str, str]:
        """Get engine display information."""
        return {"name": "anomaly", "display_name": "Anomaly Detection", "icon": "🔍", "task_type": "detection"}

    def get_config_schema(self) -> list[ConfigParameter]:
        """Get configuration schema for UI."""
        return [
            ConfigParameter(
                name="methods",
                type="select",
                default=["ensemble"],
                range=["zscore", "iqr", "isolation_forest", "lof", "dbscan", "ensemble"],
                description="Anomaly detection methods to use",
            ),
            ConfigParameter(
                name="contamination",
                type="float",
                default=0.05,
                range=[0.01, 0.5],
                description="Expected proportion of anomalies",
            ),
            ConfigParameter(
                name="threshold",
                type="float",
                default=3.0,
                range=[2.0, 5.0],
                description="Z-score threshold for statistical methods",
            ),
        ]

    def get_methodology_info(self) -> dict[str, Any]:
        """Get methodology documentation."""
        return {
            "name": "Ensemble Anomaly Detection",
            "url": "https://scikit-learn.org/stable/modules/outlier_detection.html",
            "steps": [
                {
                    "step_number": 1,
                    "title": "Column Detection",
                    "description": "Identify numeric columns via schema intelligence",
                },
                {
                    "step_number": 2,
                    "title": "Multi-Method Detection",
                    "description": "Run Z-score, IQR, Isolation Forest, LOF, DBSCAN",
                },
                {
                    "step_number": 3,
                    "title": "Ensemble Voting",
                    "description": "Aggregate results from multiple methods",
                },
                {"step_number": 4, "title": "Scoring", "description": "Calculate anomaly scores and confidence"},
            ],
            "limitations": ["High dimensionality may reduce accuracy", "Requires sufficient data for training"],
            "assumptions": ["Anomalies are rare events", "Normal data follows expected patterns"],
        }

    def get_requirements(self) -> EngineRequirements:
        return EngineRequirements(
            required_semantics=[SemanticType.NUMERIC_CONTINUOUS],
            optional_semantics={},
            required_entities=[],
            min_rows=20,
        )

    def analyze(self, df: pd.DataFrame, config: dict[str, Any] = None) -> dict[str, Any]:
        """
        Standard analyze() interface that wraps detect().
        Required for compatibility with testing framework.
        """
        return self.detect(df, config)

    def detect(self, df: pd.DataFrame, config: dict[str, Any] = None) -> dict[str, Any]:
        """
        Schema-agnostic anomaly detection using multiple methods.

        Args:
            df: Input dataframe
            config:
                - target_columns: Hint for columns to analyze
                - methods: List of methods ['zscore', 'isolation_forest', 'lof', 'dbscan', 'ensemble']
                - contamination:Expected anomaly rate (0.01-0.5)
                - threshold: Z-score threshold (default: 3.0)
                - skip_profiling: Skip schema intelligence (default: False)

        Returns:
            Anomaly detection results with scores, visualizations, and column mappings
        """
        # Handle None config
        config = config or {}

        # SCHEMA INTELLIGENCE
        if not config.get("skip_profiling", False):
            profiles = self.profiler.profile_dataset(df)
            numeric_cols = [
                col
                for col, prof in profiles.items()
                if prof.semantic_type in [SemanticType.NUMERIC_CONTINUOUS, SemanticType.NUMERIC_DISCRETE]
            ]
            target_cols = config.get("target_columns") or numeric_cols
        else:
            target_cols = config.get("target_columns")
            if not target_cols:
                target_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Fallback: try to convert string columns to numeric
        if not target_cols:
            for col in df.columns:
                if df[col].dtype == "object":
                    try:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                    except:
                        pass
            target_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if not target_cols:
            raise ValueError("No numeric columns found for anomaly detection")

        methods = config.get("methods", ["ensemble"])
        contamination = config.get("contamination", 0.05)
        z_threshold = config.get("threshold", 3.0)

        # Prepare data
        X = df[target_cols].dropna()

        if len(X) < 10:
            raise ValueError("Need at least 10 data points for anomaly detection")

        # Run each method
        results = {}
        anomaly_flags = {}

        if "zscore" in methods or "ensemble" in methods:
            results["zscore"] = self._zscore_detection(X, z_threshold)
            anomaly_flags["zscore"] = results["zscore"]["is_anomaly"]

        if "iqr" in methods or "ensemble" in methods:
            results["iqr"] = self._iqr_detection(X)
            anomaly_flags["iqr"] = results["iqr"]["is_anomaly"]

        if "isolation_forest" in methods or "ensemble" in methods:
            results["isolation_forest"] = self._isolation_forest_detection(X, contamination)
            anomaly_flags["isolation_forest"] = results["isolation_forest"]["is_anomaly"]

        if "lof" in methods or "ensemble" in methods:
            results["lof"] = self._lof_detection(X, contamination)
            anomaly_flags["lof"] = results["lof"]["is_anomaly"]

        if "dbscan" in methods or "ensemble" in methods:
            results["dbscan"] = self._dbscan_detection(X)
            anomaly_flags["dbscan"] = results["dbscan"]["is_anomaly"]

        # Ensemble voting
        if len(anomaly_flags) > 1:
            ensemble_result = self._ensemble_voting(anomaly_flags)
            results["ensemble"] = ensemble_result

        # Select primary result
        if "ensemble" in results:
            primary_result = results["ensemble"]
            primary_method = "ensemble"
        else:
            # Use first method
            primary_method = list(results.keys())[0]
            primary_result = results[primary_method]

        # Add row indices to results
        anomalies = X.index[primary_result["is_anomaly"]].tolist()

        return {
            "anomalies": anomalies,
            "anomaly_count": len(anomalies),
            "anomaly_rate": len(anomalies) / len(X),
            "primary_method": primary_method,
            "method_results": results,
            "column_mappings": {"target_columns": target_cols},
            "profiling_used": not config.get("skip_profiling", False),
            "visualizations": self._generate_anomaly_visualizations(X, primary_result, target_cols),
            "metadata": {"total_points": len(X), "columns_analyzed": target_cols, "methods_used": list(results.keys())},
        }

    def _zscore_detection(self, X: pd.DataFrame, threshold: float = 3.0) -> dict:
        """Z-score based anomaly detection"""
        # Calculate z-scores
        z_scores = np.abs(stats.zscore(X, nan_policy="omit"))

        # Flag anomalies (any column exceeds threshold)
        is_anomaly = (z_scores > threshold).any(axis=1)

        # Anomaly scores (max z-score across columns)
        scores = z_scores.max(axis=1)

        return {
            "is_anomaly": is_anomaly if isinstance(is_anomaly, np.ndarray) else is_anomaly.values,
            "scores": scores if isinstance(scores, np.ndarray) else scores.values,
            "method": "zscore",
            "threshold": threshold,
        }

    def _iqr_detection(self, X: pd.DataFrame) -> dict:
        """IQR-based anomaly detection"""
        is_anomaly = np.zeros(len(X), dtype=bool)
        scores = np.zeros(len(X))

        for col in X.columns:
            # Skip boolean columns - quantile fails on bool dtype
            if X[col].dtype == "bool":
                continue

            try:
                # Convert to float to avoid boolean issues
                col_data = X[col].astype(float)
                q1 = col_data.quantile(0.25)
                q3 = col_data.quantile(0.75)
                iqr = q3 - q1

                if iqr == 0:
                    continue

                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr

                col_anomalies = (col_data < lower) | (col_data > upper)
                is_anomaly |= col_anomalies.values

                # Score: distance from bounds (normalized)
                col_scores = np.maximum((lower - col_data) / iqr, (col_data - upper) / iqr).fillna(0).clip(lower=0)

                scores = np.maximum(scores, col_scores.values)
            except (TypeError, ValueError):
                continue

        return {"is_anomaly": is_anomaly, "scores": scores, "method": "iqr"}

    def _isolation_forest_detection(self, X: pd.DataFrame, contamination: float = 0.05) -> dict:
        """Isolation Forest anomaly detection"""
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train Isolation Forest
        model = IsolationForest(contamination=contamination, random_state=42)
        predictions = model.fit_predict(X_scaled)

        # Get anomaly scores (lower = more anomalous)
        scores = -model.score_samples(X_scaled)  # Negate to make higher = more anomalous

        # -1 = anomaly, 1 = normal
        is_anomaly = predictions == -1

        return {
            "is_anomaly": is_anomaly,
            "scores": scores,
            "method": "isolation_forest",
            "contamination": contamination,
        }

    def _lof_detection(self, X: pd.DataFrame, contamination: float = 0.05) -> dict:
        """Local Outlier Factor detection"""
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Compute LOF
        n_neighbors = min(20, len(X) - 1)
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
        predictions = lof.fit_predict(X_scaled)

        # Get negative outlier factor (higher = more anomalous)
        scores = -lof.negative_outlier_factor_

        is_anomaly = predictions == -1

        return {"is_anomaly": is_anomaly, "scores": scores, "method": "lof", "contamination": contamination}

    def _dbscan_detection(self, X: pd.DataFrame) -> dict:
        """DBSCAN-based noise detection"""
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Run DBSCAN
        eps = 0.5
        min_samples = max(2, int(len(X) * 0.01))  # 1% of data or minimum 2

        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X_scaled)

        # -1 = noise/anomaly
        is_anomaly = labels == -1

        # Score: distance to nearest core point (approximation)
        # For simplicity, binary: anomaly=1, normal=0
        scores = is_anomaly.astype(float)

        return {"is_anomaly": is_anomaly, "scores": scores, "method": "dbscan", "eps": eps, "min_samples": min_samples}

    def _ensemble_voting(self, anomaly_flags: dict[str, np.ndarray]) -> dict:
        """Ensemble voting across multiple methods"""
        # Stack all flags
        flags_matrix = np.column_stack(list(anomaly_flags.values()))

        # Majority voting
        vote_counts = flags_matrix.sum(axis=1)
        threshold = len(anomaly_flags) / 2  # More than half methods agree

        is_anomaly = vote_counts > threshold

        # Ensemble score: fraction of methods that flagged as anomaly
        scores = vote_counts / len(anomaly_flags)

        return {
            "is_anomaly": is_anomaly,
            "scores": scores,
            "method": "ensemble",
            "num_methods": len(anomaly_flags),
            "vote_counts": vote_counts.tolist(),
        }

    def _generate_anomaly_visualizations(self, X: pd.DataFrame, result: dict, columns: list[str]) -> list[dict]:
        """Generate visualization metadata"""
        visualizations = []

        # Scatter plot with anomalies highlighted
        if len(columns) >= 2:
            visualizations.append(
                {
                    "type": "scatter_plot",
                    "title": f"Anomalies: {columns[0]} vs {columns[1]}",
                    "data": {
                        "x": X[columns[0]].tolist(),
                        "y": X[columns[1]].tolist(),
                        "is_anomaly": result["is_anomaly"].tolist(),
                        "x_label": columns[0],
                        "y_label": columns[1],
                    },
                    "description": "Anomalies highlighted in red",
                }
            )

        # Anomaly score distribution
        visualizations.append(
            {
                "type": "histogram",
                "title": "Anomaly Score Distribution",
                "data": {"scores": result["scores"].tolist()},
                "description": f"Distribution of anomaly scores ({result['method']} method)",
            }
        )

        # Time series with anomalies (if index is datetime)
        if isinstance(X.index, pd.DatetimeIndex) and len(columns) > 0:
            visualizations.append(
                {
                    "type": "line_chart_with_markers",
                    "title": f"Time Series with Anomalies: {columns[0]}",
                    "data": {
                        "dates": [str(d) for d in X.index],
                        "values": X[columns[0]].tolist(),
                        "is_anomaly": result["is_anomaly"].tolist(),
                    },
                    "description": "Anomalies marked with red dots",
                }
            )

        return visualizations
