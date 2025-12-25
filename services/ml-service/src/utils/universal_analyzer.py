"""
Universal Data Analysis Engine

Generates comprehensive, human-readable analysis with:
1. Clear explanations of what was found
2. Statistical summaries in plain English
3. Predictive insights with confidence intervals
4. Chart data with forecast lines and variance bands
"""

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from .column_classifier import ColumnClassifier, ColumnProfile, ColumnRole, StatisticalType, classify_dataset
except ImportError:
    from utils.column_classifier import ColumnClassifier, ColumnProfile, ColumnRole, StatisticalType


@dataclass
class AnalysisResult:
    """Complete analysis result with explanations and visualizations."""

    # Dataset overview
    dataset_summary: dict[str, Any] = field(default_factory=dict)

    # Column classification
    column_profiles: dict[str, ColumnProfile] = field(default_factory=dict)

    # Key findings in plain English
    key_findings: list[str] = field(default_factory=list)

    # Detailed explanations
    explanations: dict[str, str] = field(default_factory=dict)

    # Chart data for visualization
    charts: dict[str, Any] = field(default_factory=dict)

    # Prediction/forecast data
    predictions: dict[str, Any] = field(default_factory=dict)

    # Recommended next steps
    next_steps: list[str] = field(default_factory=list)

    # Overall data quality score
    quality_score: float = 0.0
    quality_label: str = ""


class UniversalAnalyzer:
    """
    Universal data analyzer that works on any dataset.
    Uses statistical methods instead of keyword matching.
    """

    def __init__(self):
        self.classifier = ColumnClassifier()

    def analyze(self, df: pd.DataFrame) -> AnalysisResult:
        """
        Perform comprehensive analysis on any dataset.

        Args:
            df: Input DataFrame

        Returns:
            AnalysisResult with complete analysis
        """
        result = AnalysisResult()

        # Step 1: Classify all columns
        result.column_profiles = self.classifier.classify_dataframe(df)

        # Step 2: Generate dataset summary
        result.dataset_summary = self._generate_dataset_summary(df, result.column_profiles)

        # Step 3: Calculate data quality
        result.quality_score, result.quality_label = self._calculate_quality(df, result.column_profiles)

        # Step 4: Generate key findings
        result.key_findings = self._generate_key_findings(df, result.column_profiles, result.dataset_summary)

        # Step 5: Generate detailed explanations
        result.explanations = self._generate_explanations(df, result.column_profiles, result.dataset_summary)

        # Step 6: Generate chart data
        result.charts = self._generate_chart_data(df, result.column_profiles)

        # Step 7: Generate predictions with confidence intervals
        result.predictions = self._generate_predictions(df, result.column_profiles)

        # Step 8: Generate next steps
        result.next_steps = self._generate_next_steps(df, result.column_profiles, result.dataset_summary)

        return result

    def _generate_dataset_summary(self, df: pd.DataFrame, profiles: dict[str, ColumnProfile]) -> dict[str, Any]:
        """Generate comprehensive dataset summary."""
        n_rows, n_cols = df.shape

        # Count by role
        roles = {}
        for profile in profiles.values():
            role = profile.role.value
            roles[role] = roles.get(role, 0) + 1

        # Count by statistical type
        types = {}
        for profile in profiles.values():
            if profile.is_usable:
                stype = profile.statistical_type.value
                types[stype] = types.get(stype, 0) + 1

        # Get usable columns
        usable = [name for name, p in profiles.items() if p.is_usable and p.role == ColumnRole.FEATURE]

        # Get target
        target = None
        target_type = None
        for name, p in profiles.items():
            if p.role == ColumnRole.TARGET:
                target = name
                target_type = p.statistical_type.value
                break

        # Size classification
        if n_rows < 50:
            size_class = "very small"
            size_emoji = "📄"
        elif n_rows < 500:
            size_class = "small"
            size_emoji = "📋"
        elif n_rows < 5000:
            size_class = "medium"
            size_emoji = "📊"
        elif n_rows < 50000:
            size_class = "large"
            size_emoji = "📈"
        else:
            size_class = "very large"
            size_emoji = "🗄️"

        return {
            "n_rows": n_rows,
            "n_cols": n_cols,
            "n_usable_features": len(usable),
            "usable_features": usable,
            "target_column": target,
            "target_type": target_type,
            "column_roles": roles,
            "statistical_types": types,
            "size_class": size_class,
            "size_emoji": size_emoji,
            "has_sufficient_data": n_rows >= 30,
            "has_sufficient_features": len(usable) >= 2,
        }

    def _calculate_quality(self, df: pd.DataFrame, profiles: dict[str, ColumnProfile]) -> tuple[float, str]:
        """Calculate overall data quality score."""
        scores = []

        # Factor 1: Completeness (missing data)
        total_cells = df.size
        missing_cells = df.isnull().sum().sum()
        completeness = (1 - missing_cells / total_cells) * 100 if total_cells > 0 else 0
        scores.append(("completeness", completeness, 0.4))

        # Factor 2: Duplicates
        n_duplicates = df.duplicated().sum()
        uniqueness = (1 - n_duplicates / len(df)) * 100 if len(df) > 0 else 0
        scores.append(("uniqueness", uniqueness, 0.2))

        # Factor 3: Usable columns ratio
        usable_ratio = sum(1 for p in profiles.values() if p.is_usable) / len(profiles) * 100
        scores.append(("usability", usable_ratio, 0.2))

        # Factor 4: Target availability
        has_target = any(p.role == ColumnRole.TARGET for p in profiles.values())
        target_score = 100 if has_target else 50
        scores.append(("target", target_score, 0.2))

        # Weighted average
        weighted_sum = sum(score * weight for _, score, weight in scores)
        total_weight = sum(weight for _, _, weight in scores)
        quality_score = weighted_sum / total_weight

        # Label
        if quality_score >= 90:
            label = "Excellent"
        elif quality_score >= 75:
            label = "Good"
        elif quality_score >= 60:
            label = "Fair"
        else:
            label = "Needs Improvement"

        return round(quality_score, 1), label

    def _generate_key_findings(
        self, df: pd.DataFrame, profiles: dict[str, ColumnProfile], summary: dict[str, Any]
    ) -> list[str]:
        """Generate key findings in plain English."""
        findings = []

        # Finding 1: Dataset size and shape
        n_rows = summary["n_rows"]
        n_features = summary["n_usable_features"]
        size_class = summary["size_class"]

        if n_rows >= 1000:
            row_str = f"{n_rows:,}"
        else:
            row_str = str(n_rows)

        findings.append(
            f"{summary['size_emoji']} **Dataset Overview**: {row_str} records with "
            f"{n_features} usable features for analysis."
        )

        # Finding 2: What we can predict
        target = summary["target_column"]
        target_type = summary["target_type"]

        if target:
            target_profile = profiles[target]

            if target_type == "boolean":
                findings.append(
                    f"🎯 **Prediction Target**: '{target}' - This is a Yes/No outcome. "
                    f"We can predict which category a record belongs to."
                )
            elif target_type == "categorical":
                n_classes = target_profile.n_unique
                findings.append(
                    f"🎯 **Prediction Target**: '{target}' - This has {n_classes} possible values. "
                    f"We can classify records into these categories."
                )
            elif target_type == "continuous":
                if target_profile.mean and target_profile.std:
                    findings.append(
                        f"🎯 **Prediction Target**: '{target}' - Numeric values "
                        f"(avg: {target_profile.mean:.2f}, range: {target_profile.min_val:.2f} to {target_profile.max_val:.2f}). "
                        f"We can forecast future values."
                    )
        else:
            findings.append(
                "⚠️ **No Clear Prediction Target**: The data doesn't have an obvious outcome to predict. "
                "Consider specifying what you want to analyze."
            )

        # Finding 3: Data completeness issues
        missing_cols = [name for name, p in profiles.items() if p.missing_pct > 10]
        if missing_cols:
            if len(missing_cols) <= 3:
                cols_str = ", ".join(f"'{c}'" for c in missing_cols)
            else:
                cols_str = f"{len(missing_cols)} columns"
            findings.append(
                f"⚠️ **Missing Data**: {cols_str} have significant missing values that may affect analysis accuracy."
            )

        # Finding 4: What columns were excluded and why
        excluded = {
            "identifiers": [],
            "geographic": [],
            "text": [],
            "constant": [],
        }
        for name, p in profiles.items():
            if p.role == ColumnRole.IDENTIFIER:
                excluded["identifiers"].append(name)
            elif p.role == ColumnRole.GEOGRAPHIC:
                excluded["geographic"].append(name)
            elif p.role == ColumnRole.METADATA:
                excluded["text"].append(name)
            elif p.role == ColumnRole.CONSTANT:
                excluded["constant"].append(name)

        exclusion_notes = []
        if excluded["identifiers"]:
            exclusion_notes.append(f"{len(excluded['identifiers'])} ID column(s)")
        if excluded["geographic"]:
            exclusion_notes.append(f"{len(excluded['geographic'])} geographic column(s)")
        if excluded["text"]:
            exclusion_notes.append(f"{len(excluded['text'])} text/description column(s)")
        if excluded["constant"]:
            exclusion_notes.append(f"{len(excluded['constant'])} constant column(s)")

        if exclusion_notes:
            findings.append(
                f"🔍 **Columns Filtered Out**: {', '.join(exclusion_notes)} were excluded from analysis "
                f"because they don't contain patterns useful for prediction."
            )

        # Finding 5: Key relationships (correlations for numeric data)
        numeric_cols = [
            name for name, p in profiles.items() if p.is_usable and p.statistical_type == StatisticalType.CONTINUOUS
        ]

        if len(numeric_cols) >= 2:
            numeric_df = df[numeric_cols].dropna()
            if len(numeric_df) > 10:
                corr_matrix = numeric_df.corr()

                # Find strongest correlation
                high_corr = []
                for i, c1 in enumerate(corr_matrix.columns):
                    for c2 in corr_matrix.columns[i + 1 :]:
                        val = corr_matrix.loc[c1, c2]
                        if abs(val) > 0.5 and abs(val) < 0.98:
                            high_corr.append((c1, c2, val))

                if high_corr:
                    high_corr.sort(key=lambda x: abs(x[2]), reverse=True)
                    c1, c2, val = high_corr[0]
                    direction = "increases" if val > 0 else "decreases"
                    strength = "strongly" if abs(val) > 0.7 else "moderately"

                    findings.append(
                        f"🔗 **Key Relationship**: '{c1}' and '{c2}' are {strength} related "
                        f"({abs(val):.0%}). When one {direction}, the other tends to follow."
                    )

        return findings

    def _generate_explanations(
        self, df: pd.DataFrame, profiles: dict[str, ColumnProfile], summary: dict[str, Any]
    ) -> dict[str, str]:
        """Generate detailed explanations for different aspects."""
        explanations = {}

        # Explain the dataset
        explanations["dataset"] = self._explain_dataset(df, summary)

        # Explain the target
        if summary["target_column"]:
            explanations["target"] = self._explain_target(df, profiles, summary)

        # Explain data quality
        explanations["quality"] = self._explain_quality(df, profiles)

        # Explain analysis approach
        explanations["approach"] = self._explain_approach(df, profiles, summary)

        return explanations

    def _explain_dataset(self, df: pd.DataFrame, summary: dict[str, Any]) -> str:
        """Generate dataset explanation."""
        n_rows = summary["n_rows"]
        size_class = summary["size_class"]
        n_features = summary["n_usable_features"]

        # Data adequacy
        if n_rows < 30:
            adequacy = (
                "This dataset is quite small, which limits the reliability of statistical analysis. "
                "Results should be considered preliminary."
            )
        elif n_rows < 100:
            adequacy = (
                "This is a small dataset. While analysis is possible, "
                "patterns found may not generalize well to larger populations."
            )
        elif n_rows < 1000:
            adequacy = (
                "This dataset has a moderate size, suitable for most analysis techniques. "
                "Results should be reasonably reliable."
            )
        else:
            adequacy = (
                "This is a substantial dataset with enough data points for robust analysis. "
                "Patterns found are likely to be statistically significant."
            )

        return (
            f"Your {size_class} dataset contains {n_rows:,} records and {summary['n_cols']} columns, "
            f"of which {n_features} are suitable for predictive analysis. {adequacy}"
        )

    def _explain_target(self, df: pd.DataFrame, profiles: dict[str, ColumnProfile], summary: dict[str, Any]) -> str:
        """Explain the target column and prediction type."""
        target = summary["target_column"]
        target_type = summary["target_type"]
        profile = profiles[target]

        if target_type == "boolean":
            # Get class distribution
            value_counts = df[target].value_counts()
            if len(value_counts) == 2:
                class1, class2 = value_counts.index[:2]
                pct1 = value_counts.iloc[0] / len(df) * 100
                pct2 = value_counts.iloc[1] / len(df) * 100

                return (
                    f"The target '{target}' is a binary outcome with two possible values: "
                    f"'{class1}' ({pct1:.1f}%) and '{class2}' ({pct2:.1f}%). "
                    f"This is a classification problem where we predict which category each record belongs to."
                )

        elif target_type == "categorical":
            n_classes = profile.n_unique
            return (
                f"The target '{target}' is a categorical variable with {n_classes} possible values. "
                f"This is a multi-class classification problem."
            )

        elif target_type == "continuous":
            return (
                f"The target '{target}' is a continuous numeric variable "
                f"(average: {profile.mean:.2f}, std: {profile.std:.2f}). "
                f"This is a regression problem where we predict numeric values."
            )

        return f"The target '{target}' will be used for prediction."

    def _explain_quality(self, df: pd.DataFrame, profiles: dict[str, ColumnProfile]) -> str:
        """Explain data quality issues."""
        issues = []

        # Check for missing data
        total_missing = df.isnull().sum().sum()
        total_cells = df.size
        missing_pct = total_missing / total_cells * 100 if total_cells > 0 else 0

        if missing_pct > 5:
            issues.append(f"{missing_pct:.1f}% of data is missing")

        # Check for duplicates
        n_dupes = df.duplicated().sum()
        if n_dupes > 0:
            issues.append(f"{n_dupes} duplicate rows found")

        # Check for constant columns
        constant_cols = [name for name, p in profiles.items() if p.role == ColumnRole.CONSTANT]
        if constant_cols:
            issues.append(f"{len(constant_cols)} column(s) have only one value")

        if issues:
            return f"Data quality considerations: {'; '.join(issues)}. These factors may affect analysis accuracy."
        else:
            return "Data quality is good with minimal missing values and no major issues detected."

    def _explain_approach(self, df: pd.DataFrame, profiles: dict[str, ColumnProfile], summary: dict[str, Any]) -> str:
        """Explain the analysis approach used."""
        n_features = summary["n_usable_features"]
        target_type = summary["target_type"]

        if n_features < 2:
            return (
                "Limited feature analysis is possible due to few usable columns. "
                "Consider adding more data fields for deeper insights."
            )

        approach_parts = []

        if target_type == "boolean":
            approach_parts.append("binary classification to predict Yes/No outcomes")
        elif target_type == "categorical":
            approach_parts.append("multi-class classification to categorize records")
        elif target_type == "continuous":
            approach_parts.append("regression analysis to forecast numeric values")

        # Add feature analysis
        numeric_features = sum(
            1 for p in profiles.values() if p.is_usable and p.statistical_type == StatisticalType.CONTINUOUS
        )
        categorical_features = sum(
            1
            for p in profiles.values()
            if p.is_usable and p.statistical_type in [StatisticalType.CATEGORICAL, StatisticalType.BOOLEAN]
        )

        if numeric_features > 0:
            approach_parts.append(f"correlation analysis on {numeric_features} numeric features")
        if categorical_features > 0:
            approach_parts.append(f"category distribution analysis on {categorical_features} categorical features")

        return f"Analysis approach: {', '.join(approach_parts)}."

    def _generate_chart_data(self, df: pd.DataFrame, profiles: dict[str, ColumnProfile]) -> dict[str, Any]:
        """Generate chart data for visualization."""
        charts = {}

        # 1. Data quality gauge
        quality_score, quality_label = self._calculate_quality(df, profiles)
        charts["quality_gauge"] = {
            "type": "gauge",
            "value": quality_score,
            "max": 100,
            "label": quality_label,
            "color": "#10b981" if quality_score >= 80 else "#f59e0b" if quality_score >= 60 else "#ef4444",
        }

        # 2. Column types pie chart
        type_counts = {"Numeric": 0, "Categorical": 0, "Boolean": 0, "Other": 0}
        for p in profiles.values():
            if not p.is_usable:
                continue
            if p.statistical_type == StatisticalType.CONTINUOUS:
                type_counts["Numeric"] += 1
            elif p.statistical_type == StatisticalType.CATEGORICAL:
                type_counts["Categorical"] += 1
            elif p.statistical_type == StatisticalType.BOOLEAN:
                type_counts["Boolean"] += 1
            else:
                type_counts["Other"] += 1

        charts["column_types"] = {
            "type": "pie",
            "labels": list(type_counts.keys()),
            "values": list(type_counts.values()),
            "colors": ["#8b5cf6", "#3b82f6", "#10b981", "#6b7280"],
        }

        # 3. Feature importance (if we have a target)
        target_col = None
        for name, p in profiles.items():
            if p.role == ColumnRole.TARGET:
                target_col = name
                break

        if target_col:
            charts["feature_importance"] = self._calculate_feature_importance(df, profiles, target_col)

        # 4. Target distribution
        if target_col:
            charts["target_distribution"] = self._get_target_distribution(df, target_col, profiles)

        # 5. Numeric distributions
        charts["distributions"] = self._get_numeric_distributions(df, profiles)

        return charts

    def _calculate_feature_importance(
        self, df: pd.DataFrame, profiles: dict[str, ColumnProfile], target_col: str
    ) -> dict[str, Any]:
        """Calculate feature importance scores for any feature type."""
        # Get ALL usable features (numeric and categorical)
        feature_cols = [name for name, p in profiles.items() if p.is_usable and p.role == ColumnRole.FEATURE]

        if len(feature_cols) < 1:
            return {"type": "bar", "labels": [], "values": [], "title": "Feature Importance"}

        try:
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            from sklearn.preprocessing import LabelEncoder

            # Prepare feature matrix - encode all columns appropriately
            X_encoded = pd.DataFrame()
            encoded_cols = []

            for col in feature_cols:
                profile = profiles[col]
                col_data = df[col].copy()

                if profile.statistical_type in [StatisticalType.CONTINUOUS, StatisticalType.DISCRETE]:
                    # Numeric - just fill NA
                    X_encoded[col] = col_data.fillna(col_data.median())
                    encoded_cols.append(col)

                elif profile.statistical_type == StatisticalType.BOOLEAN:
                    # Boolean - convert Yes/No to 1/0
                    if col_data.dtype == "object":
                        le = LabelEncoder()
                        encoded = le.fit_transform(col_data.fillna("Unknown").astype(str))
                        X_encoded[col] = encoded
                    else:
                        X_encoded[col] = col_data.fillna(0).astype(int)
                    encoded_cols.append(col)

                elif profile.statistical_type == StatisticalType.CATEGORICAL:
                    # Categorical - label encode (limit cardinality)
                    if profile.n_unique <= 20:  # Only encode low-cardinality
                        le = LabelEncoder()
                        encoded = le.fit_transform(col_data.fillna("_Missing_").astype(str))
                        X_encoded[col] = encoded
                        encoded_cols.append(col)

            if len(encoded_cols) < 1:
                return {"type": "bar", "labels": [], "values": [], "title": "Feature Importance"}

            # Prepare target
            y = df[target_col].copy()

            # Encode target if needed
            if y.dtype == "object" or not pd.api.types.is_numeric_dtype(y):
                le = LabelEncoder()
                y = le.fit_transform(y.fillna("_Missing_").astype(str))
            else:
                y = y.values

            # Remove rows with missing data
            mask = ~pd.isnull(y) & X_encoded.notna().all(axis=1)
            X = X_encoded[mask]
            y = y[mask]

            if len(X) < 10:
                return {"type": "bar", "labels": [], "values": [], "title": "Feature Importance"}

            # Train simple model
            target_profile = profiles[target_col]
            if target_profile.statistical_type in [StatisticalType.BOOLEAN, StatisticalType.CATEGORICAL]:
                model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
            else:
                model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)

            model.fit(X, y)

            # Get importance scores - use encoded_cols to match features
            importance = dict(zip(encoded_cols, model.feature_importances_ * 100))
            importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10])

            return {
                "type": "bar",
                "labels": list(importance.keys()),
                "values": [round(v, 1) for v in importance.values()],
                "title": f"Feature Importance for Predicting {target_col}",
                "colors": [f"rgba(139, 92, 246, {1 - i * 0.08})" for i in range(len(importance))],
            }
        except Exception as e:
            return {"type": "bar", "labels": [], "values": [], "title": "Feature Importance", "error": str(e)}

    def _get_target_distribution(
        self, df: pd.DataFrame, target_col: str, profiles: dict[str, ColumnProfile]
    ) -> dict[str, Any]:
        """Get target variable distribution."""
        profile = profiles[target_col]

        if profile.statistical_type in [StatisticalType.BOOLEAN, StatisticalType.CATEGORICAL]:
            value_counts = df[target_col].value_counts()
            return {
                "type": "bar",
                "labels": [str(v) for v in value_counts.index.tolist()[:10]],
                "values": value_counts.values.tolist()[:10],
                "title": f"Distribution of {target_col}",
            }
        else:
            # Histogram for continuous
            data = df[target_col].dropna()
            hist, bin_edges = np.histogram(data, bins=20)
            return {
                "type": "histogram",
                "values": hist.tolist(),
                "bin_labels": [f"{bin_edges[i]:.1f}" for i in range(len(bin_edges) - 1)],
                "title": f"Distribution of {target_col}",
                "mean": float(data.mean()),
                "std": float(data.std()),
            }

    def _get_numeric_distributions(self, df: pd.DataFrame, profiles: dict[str, ColumnProfile]) -> list[dict]:
        """Get distributions for numeric columns."""
        distributions = []

        numeric_cols = [
            name for name, p in profiles.items() if p.is_usable and p.statistical_type == StatisticalType.CONTINUOUS
        ][:5]

        for col in numeric_cols:
            data = df[col].dropna()
            if len(data) > 0:
                hist, bin_edges = np.histogram(data, bins=15)
                distributions.append(
                    {
                        "column": col,
                        "type": "histogram",
                        "values": hist.tolist(),
                        "bin_labels": [f"{bin_edges[i]:.1f}" for i in range(len(bin_edges) - 1)],
                        "mean": float(data.mean()),
                        "median": float(data.median()),
                        "std": float(data.std()),
                    }
                )

        return distributions

    def _generate_predictions(self, df: pd.DataFrame, profiles: dict[str, ColumnProfile]) -> dict[str, Any]:
        """
        Generate prediction data with confidence intervals.
        Creates forecast lines with dotted predictions and shaded variance bands.
        """
        predictions = {}

        # Find target
        target_col = None
        for name, p in profiles.items():
            if p.role == ColumnRole.TARGET:
                target_col = name
                break

        if not target_col:
            return predictions

        target_profile = profiles[target_col]

        # For classification targets, predict class probabilities
        if target_profile.statistical_type in [StatisticalType.BOOLEAN, StatisticalType.CATEGORICAL]:
            predictions["classification"] = self._generate_classification_prediction(df, profiles, target_col)

        # For continuous targets, create trend forecast
        if target_profile.statistical_type == StatisticalType.CONTINUOUS:
            predictions["regression"] = self._generate_regression_forecast(df, profiles, target_col)

        # Generate confidence chart data
        predictions["confidence_chart"] = self._generate_confidence_chart(df, profiles, target_col)

        return predictions

    def _generate_classification_prediction(
        self, df: pd.DataFrame, profiles: dict[str, ColumnProfile], target_col: str
    ) -> dict[str, Any]:
        """Generate classification predictions with confidence."""
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import LabelEncoder

            # Get features - include categorical with low cardinality
            feature_cols = [
                name
                for name, p in profiles.items()
                if p.is_usable
                and p.role == ColumnRole.FEATURE
                and p.statistical_type
                in [
                    StatisticalType.CONTINUOUS,
                    StatisticalType.DISCRETE,
                    StatisticalType.BOOLEAN,
                    StatisticalType.CATEGORICAL,
                ]
            ]

            if len(feature_cols) < 1:
                return {"success": False, "error": "Insufficient features"}

            # Prepare features with proper encoding
            X = df[feature_cols].copy()
            encoded_cols = []

            for col in feature_cols:
                profile = profiles[col]
                if profile.statistical_type in [StatisticalType.BOOLEAN, StatisticalType.CATEGORICAL]:
                    # Only encode if low cardinality
                    if X[col].nunique() <= 20:
                        col_le = LabelEncoder()
                        X[col] = col_le.fit_transform(X[col].astype(str).fillna("_missing_"))
                        encoded_cols.append(col)
                    else:
                        # Skip high cardinality categorical
                        X = X.drop(columns=[col])
                else:
                    # Numeric - fill with median
                    X[col] = X[col].fillna(X[col].median())
                    encoded_cols.append(col)

            if len(encoded_cols) < 1:
                return {"success": False, "error": "No usable features after encoding"}

            X = X[encoded_cols]
            y = df[target_col]

            # Encode target
            le = LabelEncoder()
            y_encoded = le.fit_transform(y.astype(str))

            # Remove missing
            mask = ~pd.isnull(y)
            X = X[mask]
            y_encoded = y_encoded[mask]

            if len(X) < 30:
                return {"success": False, "error": "Insufficient data"}

            # Train model with probability
            model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)

            # Cross-validation for confidence
            cv_folds = min(5, len(X) // 10)
            if cv_folds >= 2:
                from sklearn.model_selection import cross_val_score

                scores = cross_val_score(model, X, y_encoded, cv=cv_folds, scoring="accuracy")

                model.fit(X, y_encoded)

                return {
                    "success": True,
                    "accuracy": float(scores.mean()),
                    "accuracy_std": float(scores.std()),
                    "confidence_interval": [
                        float(scores.mean() - 1.96 * scores.std()),
                        float(scores.mean() + 1.96 * scores.std()),
                    ],
                    "classes": le.classes_.tolist(),
                    "class_probabilities": model.predict_proba(X).mean(axis=0).tolist(),
                }

            return {"success": False, "error": "Insufficient data for cross-validation"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _generate_regression_forecast(
        self, df: pd.DataFrame, profiles: dict[str, ColumnProfile], target_col: str
    ) -> dict[str, Any]:
        """
        Generate regression forecast with confidence intervals.
        Creates data for trend line with shaded variance band.
        """
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import cross_val_score
            from sklearn.preprocessing import LabelEncoder

            # Get features - include categorical with low cardinality
            feature_cols = [
                name
                for name, p in profiles.items()
                if p.is_usable
                and p.role == ColumnRole.FEATURE
                and p.statistical_type
                in [
                    StatisticalType.CONTINUOUS,
                    StatisticalType.DISCRETE,
                    StatisticalType.BOOLEAN,
                    StatisticalType.CATEGORICAL,
                ]
            ]

            if len(feature_cols) < 1:
                return {"success": False, "error": "Insufficient features"}

            # Prepare features with proper encoding
            X = df[feature_cols].copy()
            encoded_cols = []

            for col in feature_cols:
                profile = profiles[col]
                if profile.statistical_type in [StatisticalType.BOOLEAN, StatisticalType.CATEGORICAL]:
                    # Only encode if low cardinality
                    if X[col].nunique() <= 20:
                        le = LabelEncoder()
                        X[col] = le.fit_transform(X[col].astype(str).fillna("_missing_"))
                        encoded_cols.append(col)
                    else:
                        # Skip high cardinality categorical
                        X = X.drop(columns=[col])
                else:
                    # Numeric - fill with median
                    X[col] = X[col].fillna(X[col].median())
                    encoded_cols.append(col)

            if len(encoded_cols) < 1:
                return {"success": False, "error": "No usable features after encoding"}

            X = X[encoded_cols]
            y = df[target_col].values

            # Remove missing
            mask = ~pd.isnull(y)
            X = X[mask]
            y = y[mask]

            if len(X) < 30:
                return {"success": False, "error": "Insufficient data"}

            # Train model
            model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)

            # Cross-validation
            cv_folds = min(5, len(X) // 10)
            if cv_folds >= 2:
                scores = cross_val_score(model, X, y, cv=cv_folds, scoring="neg_mean_absolute_error")
                mae = -scores.mean()
                mae_std = scores.std()

                model.fit(X, y)
                predictions = model.predict(X)

                # Generate forecast line with confidence band
                # Sort by first numeric feature for visualization
                if len(encoded_cols) > 0:
                    sort_col = encoded_cols[0]
                    sort_idx = np.argsort(X[sort_col].values)

                    # Sample points for chart (max 50)
                    n_points = min(50, len(sort_idx))
                    sample_idx = sort_idx[:: len(sort_idx) // n_points][:n_points]

                    x_values = X[sort_col].values[sample_idx]
                    y_actual = y[sample_idx]
                    y_predicted = predictions[sample_idx]

                    # Confidence band (±1.96 * std)
                    residuals = y - predictions
                    residual_std = np.std(residuals)

                    return {
                        "success": True,
                        "mae": float(mae),
                        "mae_std": float(mae_std),
                        "r2": float(1 - np.var(y - predictions) / np.var(y)),
                        "x_feature": sort_col,
                        "chart_data": {
                            "x_values": x_values.tolist(),
                            "y_actual": y_actual.tolist(),
                            "y_predicted": y_predicted.tolist(),
                            "upper_bound": (y_predicted + 1.96 * residual_std).tolist(),
                            "lower_bound": (y_predicted - 1.96 * residual_std).tolist(),
                        },
                        "residual_std": float(residual_std),
                    }

            return {"success": False, "error": "Insufficient data for cross-validation"}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _generate_confidence_chart(
        self, df: pd.DataFrame, profiles: dict[str, ColumnProfile], target_col: str
    ) -> dict[str, Any]:
        """
        Generate chart data with prediction line and confidence bands.
        Format for Chart.js with dotted forecast line and shaded area.
        """
        target_profile = profiles[target_col]

        chart = {"type": "line_with_confidence", "title": f"Prediction Analysis: {target_col}", "datasets": []}

        if target_profile.statistical_type in [StatisticalType.BOOLEAN, StatisticalType.CATEGORICAL]:
            # For classification, show class distribution and prediction confidence
            classification_result = self._generate_classification_prediction(df, profiles, target_col)

            if classification_result.get("success"):
                chart["type"] = "classification_confidence"
                chart["classes"] = classification_result.get("classes", [])
                chart["probabilities"] = classification_result.get("class_probabilities", [])
                chart["accuracy"] = classification_result.get("accuracy", 0)
                chart["confidence_interval"] = classification_result.get("confidence_interval", [0, 0])

                # Create bar chart for class distribution - match by string conversion
                class_counts = df[target_col].astype(str).value_counts()
                classes = classification_result.get("classes", [])
                chart["datasets"].append(
                    {
                        "label": "Class Distribution",
                        "data": [int(class_counts.get(str(c), 0)) for c in classes],
                        "type": "bar",
                        "backgroundColor": ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6"][: len(classes)],
                    }
                )

                # Create prediction accuracy visualization
                chart["summary"] = {
                    "accuracy": classification_result.get("accuracy", 0),
                    "accuracy_std": classification_result.get("accuracy_std", 0),
                    "confidence_interval": classification_result.get("confidence_interval", [0, 0]),
                }

        elif target_profile.statistical_type == StatisticalType.CONTINUOUS:
            # For regression, create trend forecast
            regression_result = self._generate_regression_forecast(df, profiles, target_col)

            if regression_result.get("success") and "chart_data" in regression_result:
                data = regression_result["chart_data"]

                chart["x_label"] = regression_result.get("x_feature", "Index")
                chart["y_label"] = target_col
                chart["x_values"] = data["x_values"]

                # Actual values (solid line)
                chart["datasets"].append(
                    {
                        "label": "Actual Values",
                        "data": data["y_actual"],
                        "type": "scatter",
                        "borderColor": "#3b82f6",
                        "backgroundColor": "rgba(59, 130, 246, 0.5)",
                        "pointRadius": 3,
                    }
                )

                # Predicted trend (dotted line)
                chart["datasets"].append(
                    {
                        "label": "Predicted Trend",
                        "data": data["y_predicted"],
                        "type": "line",
                        "borderColor": "#8b5cf6",
                        "borderDash": [5, 5],  # Dotted line
                        "fill": False,
                        "pointRadius": 0,
                    }
                )

                # Confidence band (shaded area)
                chart["datasets"].append(
                    {
                        "label": "95% Confidence Band",
                        "data": data["upper_bound"],
                        "type": "line",
                        "borderColor": "rgba(139, 92, 246, 0.3)",
                        "backgroundColor": "rgba(139, 92, 246, 0.1)",
                        "fill": "+1",  # Fill to next dataset
                        "pointRadius": 0,
                    }
                )
                chart["datasets"].append(
                    {
                        "label": "Lower Bound",
                        "data": data["lower_bound"],
                        "type": "line",
                        "borderColor": "rgba(139, 92, 246, 0.3)",
                        "fill": False,
                        "pointRadius": 0,
                        "hidden": True,  # Used only for fill reference
                    }
                )

                # Add summary stats
                chart["summary"] = {
                    "r2": regression_result.get("r2", 0),
                    "mae": regression_result.get("mae", 0),
                    "confidence_range": regression_result.get("residual_std", 0) * 1.96,
                }

        return chart

    def _generate_next_steps(
        self, df: pd.DataFrame, profiles: dict[str, ColumnProfile], summary: dict[str, Any]
    ) -> list[str]:
        """Generate recommended next steps."""
        steps = []

        # Step based on data quality
        missing_pct = df.isnull().sum().sum() / df.size * 100 if df.size > 0 else 0
        if missing_pct > 10:
            steps.append("🧹 **Clean missing data** - Fill or remove entries with missing values to improve accuracy.")

        # Step based on available features
        if summary["n_usable_features"] < 3:
            steps.append("📊 **Add more features** - Including additional data columns would enable deeper analysis.")

        # Step based on target
        if summary["target_column"]:
            target_type = summary["target_type"]
            if target_type in ["boolean", "categorical"]:
                steps.append("🎯 **Run classification** - Build a model to predict categories for new records.")
            else:
                steps.append("📈 **Run forecasting** - Build a model to predict future numeric values.")
        else:
            steps.append("🎯 **Define a target** - Specify what outcome you want to predict or analyze.")

        # Always suggest exploration
        if summary["n_usable_features"] >= 2:
            steps.append("🔍 **Explore relationships** - Investigate how different factors affect your outcomes.")

        return steps[:4]  # Limit to 4 steps


def analyze_dataset(df: pd.DataFrame) -> AnalysisResult:
    """
    Convenience function for quick analysis.

    Args:
        df: Input DataFrame

    Returns:
        Complete AnalysisResult
    """
    analyzer = UniversalAnalyzer()
    return analyzer.analyze(df)
