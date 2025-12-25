"""
Universal Column Classifier for Automatic Data Understanding

Based on best practices from:
- pandas-profiling / ydata-profiling
- AutoGluon, H2O AutoML
- Industry standard heuristics

This module provides statistical methods to classify columns without
relying on keyword matching - works for ANY dataset.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd


class ColumnRole(Enum):
    """The role a column plays in the dataset."""

    IDENTIFIER = "identifier"  # ID, primary key, row number
    DATETIME = "datetime"  # Date/time column
    GEOGRAPHIC = "geographic"  # Lat/long/coordinates
    TARGET = "target"  # Potential prediction target
    FEATURE = "feature"  # Regular feature column
    METADATA = "metadata"  # Descriptive text, notes
    CONSTANT = "constant"  # Single value (useless)


class StatisticalType(Enum):
    """Statistical type for analysis purposes."""

    BOOLEAN = "boolean"  # True/False, Yes/No, 0/1
    CATEGORICAL = "categorical"  # Discrete categories (nominal/ordinal)
    CONTINUOUS = "continuous"  # Continuous numeric values
    DISCRETE = "discrete"  # Integer counts
    DATETIME = "datetime"  # Temporal data
    TEXT = "text"  # Free-form text
    MIXED = "mixed"  # Mixed types
    UNKNOWN = "unknown"


@dataclass
class ColumnProfile:
    """Complete profile of a column's characteristics."""

    name: str
    dtype: str
    role: ColumnRole
    statistical_type: StatisticalType

    # Basic stats
    n_rows: int = 0
    n_unique: int = 0
    n_missing: int = 0
    missing_pct: float = 0.0

    # Derived metrics
    uniqueness_ratio: float = 0.0  # n_unique / n_rows
    completeness: float = 0.0  # 1 - missing_pct

    # For numeric columns
    mean: float | None = None
    std: float | None = None
    min_val: float | None = None
    max_val: float | None = None

    # For categorical columns
    top_values: list[tuple[Any, int]] = field(default_factory=list)

    # Classification confidence
    confidence: float = 0.0
    classification_reasons: list[str] = field(default_factory=list)

    # Feature importance score (for target selection)
    target_score: float = 0.0

    # Should this column be used in analysis?
    is_usable: bool = True
    skip_reason: str | None = None


class ColumnClassifier:
    """
    Universal column type classifier using statistical methods.
    No keyword matching - works purely on data characteristics.
    """

    # Configurable thresholds (based on research)
    THRESHOLDS = {
        # ID Detection
        "id_uniqueness_min": 0.95,  # If >95% unique, likely ID
        "id_sequential_threshold": 0.95,  # If >95% sequential diffs of 1
        # Categorical Detection
        "categorical_max_unique": 50,  # Max unique values for categorical
        "categorical_max_ratio": 0.5,  # Max unique/total ratio
        "numeric_categorical_max": 10,  # Numeric with <=10 unique → categorical
        # Datetime Detection
        "datetime_parse_threshold": 0.8,  # If >80% parse successfully
        # Text Detection
        "text_uniqueness_min": 0.9,  # High uniqueness
        "text_avg_words_min": 2.0,  # Multiple words per entry
        "text_avg_length_min": 20,  # Reasonably long entries
        # Geographic Detection
        "geo_latitude_range": (-90, 90),
        "geo_longitude_range": (-180, 180),
        "geo_decimal_precision_min": 4,  # Coords usually have 4+ decimals
        # Target Detection
        "target_max_missing": 0.3,  # Skip if >30% missing
        "target_binary_boost": 3.0,  # Boost for binary columns
        "target_multiclass_boost": 2.0,  # Boost for multiclass
        "target_regression_boost": 1.5,  # Boost for regression candidates
        "target_last_column_boost": 0.5,  # Small boost for last column
    }

    def __init__(self, custom_thresholds: dict | None = None):
        """Initialize with optional custom thresholds."""
        self.thresholds = self.THRESHOLDS.copy()
        if custom_thresholds:
            self.thresholds.update(custom_thresholds)

    def classify_column(self, series: pd.Series, df: pd.DataFrame | None = None) -> ColumnProfile:
        """
        Classify a single column using statistical methods.

        Args:
            series: The column to classify
            df: Optional full dataframe for context

        Returns:
            ColumnProfile with complete classification
        """
        profile = ColumnProfile(
            name=str(series.name),
            dtype=str(series.dtype),
            role=ColumnRole.FEATURE,
            statistical_type=StatisticalType.UNKNOWN,
            n_rows=len(series),
            n_unique=series.nunique(),
            n_missing=series.isna().sum(),
        )

        # Calculate derived metrics
        profile.missing_pct = (profile.n_missing / profile.n_rows * 100) if profile.n_rows > 0 else 0
        profile.completeness = 100 - profile.missing_pct
        profile.uniqueness_ratio = profile.n_unique / profile.n_rows if profile.n_rows > 0 else 0

        # Get non-null values for analysis
        valid_data = series.dropna()

        # Check for constant columns first
        if profile.n_unique <= 1:
            profile.role = ColumnRole.CONSTANT
            profile.statistical_type = StatisticalType.UNKNOWN
            profile.is_usable = False
            profile.skip_reason = "Constant value - no variance"
            profile.classification_reasons.append("Only 1 unique value")
            return profile

        # Run classification checks in priority order

        # 1. Check for ID column
        if self._is_identifier(series, valid_data, profile, df):
            return profile

        # 2. Check for datetime
        if self._is_datetime(series, valid_data, profile):
            return profile

        # 3. Check for geographic
        if self._is_geographic(series, valid_data, profile):
            return profile

        # 4. Check for boolean
        if self._is_boolean(series, valid_data, profile):
            return profile

        # 5. Check for numeric (continuous vs discrete vs categorical)
        if pd.api.types.is_numeric_dtype(series):
            self._classify_numeric(series, valid_data, profile)
            return profile

        # 6. Check for text vs categorical
        self._classify_text(series, valid_data, profile)

        return profile

    def _is_identifier(
        self, series: pd.Series, valid_data: pd.Series, profile: ColumnProfile, df: pd.DataFrame | None
    ) -> bool:
        """Detect identifier/ID columns using multiple heuristics."""
        score = 0
        reasons = []

        # Heuristic 1: Very high uniqueness
        if profile.uniqueness_ratio >= self.thresholds["id_uniqueness_min"]:
            score += 2
            reasons.append(f"High uniqueness ({profile.uniqueness_ratio:.1%})")

        # Heuristic 2: Sequential integers
        if pd.api.types.is_integer_dtype(series):
            sorted_vals = valid_data.sort_values().values
            if len(sorted_vals) > 1:
                diffs = np.diff(sorted_vals)
                sequential_ratio = (diffs == 1).mean()
                if sequential_ratio >= self.thresholds["id_sequential_threshold"]:
                    score += 3
                    reasons.append(f"Sequential integers ({sequential_ratio:.1%})")

        # Heuristic 3: Monotonically increasing/decreasing
        if valid_data.is_monotonic_increasing or valid_data.is_monotonic_decreasing:
            score += 1.5
            reasons.append("Monotonic sequence")

        # Heuristic 4: First column position (common convention)
        if df is not None:
            try:
                col_idx = list(df.columns).index(series.name)
                if col_idx == 0:
                    score += 0.5
                    reasons.append("First column position")
            except ValueError:
                pass

        # Heuristic 5: String pattern matching for ID formats
        if pd.api.types.is_string_dtype(series) or series.dtype == "object":
            sample = valid_data.astype(str).head(100)

            # UUID pattern
            uuid_pattern = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
            if sample.str.match(uuid_pattern, case=False).mean() > 0.8:
                score += 3
                reasons.append("UUID format")

            # Alphanumeric code pattern (e.g., ABC123, A1B2C3)
            code_pattern = r"^[A-Z0-9]{6,}$"
            if sample.str.match(code_pattern, case=False).mean() > 0.8:
                score += 2
                reasons.append("Alphanumeric code format")

        # Decision threshold
        if score >= 2.5:
            profile.role = ColumnRole.IDENTIFIER
            profile.statistical_type = StatisticalType.CATEGORICAL
            profile.is_usable = False
            profile.skip_reason = "Identifier column"
            profile.classification_reasons = reasons
            profile.confidence = min(score / 5, 1.0)
            return True

        return False

    def _is_datetime(self, series: pd.Series, valid_data: pd.Series, profile: ColumnProfile) -> bool:
        """Detect datetime columns."""
        # Already datetime dtype
        if pd.api.types.is_datetime64_any_dtype(series):
            profile.role = ColumnRole.DATETIME
            profile.statistical_type = StatisticalType.DATETIME
            profile.classification_reasons.append("Native datetime dtype")
            profile.confidence = 1.0
            return True

        # Try to parse string columns as dates
        if pd.api.types.is_string_dtype(series) or series.dtype == "object":
            sample = valid_data.head(100)
            try:
                parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
                success_rate = parsed.notna().mean()

                if success_rate >= self.thresholds["datetime_parse_threshold"]:
                    profile.role = ColumnRole.DATETIME
                    profile.statistical_type = StatisticalType.DATETIME
                    profile.classification_reasons.append(f"Datetime parseable ({success_rate:.1%})")
                    profile.confidence = success_rate
                    return True
            except Exception:
                pass

        # Check for Unix timestamps (numeric)
        if pd.api.types.is_numeric_dtype(series):
            vals = valid_data.values
            # Unix timestamp range: 1970-01-01 to ~2100
            if len(vals) > 0:
                # Seconds timestamp
                if np.all((vals > 0) & (vals < 4102444800)):
                    # Check if values look like timestamps (not just small integers)
                    if np.min(vals) > 1000000000:  # After ~2001
                        profile.role = ColumnRole.DATETIME
                        profile.statistical_type = StatisticalType.DATETIME
                        profile.classification_reasons.append("Unix timestamp range")
                        profile.confidence = 0.8
                        return True

        return False

    def _is_geographic(self, series: pd.Series, valid_data: pd.Series, profile: ColumnProfile) -> bool:
        """Detect geographic coordinate columns."""
        if not pd.api.types.is_numeric_dtype(series):
            return False

        vals = valid_data.values
        if len(vals) == 0:
            return False

        min_val, max_val = np.min(vals), np.max(vals)

        # Check for latitude range (-90 to 90)
        lat_range = self.thresholds["geo_latitude_range"]
        is_lat_range = lat_range[0] <= min_val and max_val <= lat_range[1]

        # Check for longitude range (-180 to 180)
        lon_range = self.thresholds["geo_longitude_range"]
        is_lon_range = lon_range[0] <= min_val and max_val <= lon_range[1]

        if not (is_lat_range or is_lon_range):
            return False

        # Check decimal precision (geo coords typically have 4+ decimal places)
        sample = valid_data.head(100)
        decimal_places = sample.apply(lambda x: len(str(float(x)).split(".")[-1]) if "." in str(float(x)) else 0)
        median_precision = decimal_places.median()

        if median_precision >= self.thresholds["geo_decimal_precision_min"]:
            profile.role = ColumnRole.GEOGRAPHIC
            profile.statistical_type = StatisticalType.CONTINUOUS
            profile.is_usable = False
            profile.skip_reason = "Geographic coordinate"

            if is_lat_range and median_precision >= 4:
                profile.classification_reasons.append(f"Latitude range with {median_precision:.0f} decimal precision")
            else:
                profile.classification_reasons.append(f"Longitude range with {median_precision:.0f} decimal precision")

            profile.confidence = min(median_precision / 6, 1.0)
            return True

        return False

    def _is_boolean(self, series: pd.Series, valid_data: pd.Series, profile: ColumnProfile) -> bool:
        """Detect boolean/binary columns."""
        if profile.n_unique != 2:
            return False

        profile.statistical_type = StatisticalType.BOOLEAN
        profile.classification_reasons.append("Exactly 2 unique values")
        profile.confidence = 1.0

        # Get the two values
        unique_vals = valid_data.unique()
        profile.top_values = [(v, (valid_data == v).sum()) for v in unique_vals]

        # This is a good target candidate
        profile.target_score = self.thresholds["target_binary_boost"]

        return True

    def _classify_numeric(self, series: pd.Series, valid_data: pd.Series, profile: ColumnProfile) -> None:
        """Classify numeric columns as continuous, discrete, or categorical."""
        # Calculate numeric stats
        profile.mean = float(valid_data.mean())
        profile.std = float(valid_data.std())
        profile.min_val = float(valid_data.min())
        profile.max_val = float(valid_data.max())

        # Check if it should be treated as categorical (few unique values)
        if profile.n_unique <= self.thresholds["numeric_categorical_max"]:
            profile.statistical_type = StatisticalType.CATEGORICAL
            profile.classification_reasons.append(f"Low cardinality numeric ({profile.n_unique} unique)")
            profile.target_score = self.thresholds["target_multiclass_boost"]
            return

        # Check if discrete (integers) or continuous (floats)
        if pd.api.types.is_integer_dtype(series):
            # Additional check: are the values reasonably bounded?
            value_range = profile.max_val - profile.min_val
            if value_range <= 100 and profile.n_unique <= 50:
                profile.statistical_type = StatisticalType.DISCRETE
                profile.classification_reasons.append("Bounded integer values")
            else:
                profile.statistical_type = StatisticalType.CONTINUOUS
                profile.classification_reasons.append("Integer with high cardinality")
        else:
            profile.statistical_type = StatisticalType.CONTINUOUS
            profile.classification_reasons.append("Floating point values")

        # Good regression target if continuous with reasonable variance
        if profile.statistical_type == StatisticalType.CONTINUOUS:
            cv = profile.std / abs(profile.mean) if profile.mean != 0 else 0
            if cv > 0.1:  # Coefficient of variation > 10%
                profile.target_score = self.thresholds["target_regression_boost"]

    def _classify_text(self, series: pd.Series, valid_data: pd.Series, profile: ColumnProfile) -> None:
        """Classify text columns as categorical or free-form text."""
        sample = valid_data.astype(str).head(500)

        # Calculate text characteristics
        word_counts = sample.str.split().str.len()
        avg_words = word_counts.mean()

        char_lengths = sample.str.len()
        avg_length = char_lengths.mean()

        # Decide: categorical vs free-form text
        is_high_unique = profile.uniqueness_ratio >= self.thresholds["text_uniqueness_min"]
        is_multi_word = avg_words >= self.thresholds["text_avg_words_min"]
        is_long_text = avg_length >= self.thresholds["text_avg_length_min"]

        if is_high_unique and (is_multi_word or is_long_text):
            profile.statistical_type = StatisticalType.TEXT
            profile.role = ColumnRole.METADATA
            profile.is_usable = False
            profile.skip_reason = "Free-form text"
            profile.classification_reasons.append(
                f"High uniqueness ({profile.uniqueness_ratio:.1%}), "
                f"avg {avg_words:.1f} words, avg {avg_length:.0f} chars"
            )
        elif profile.n_unique <= self.thresholds["categorical_max_unique"]:
            profile.statistical_type = StatisticalType.CATEGORICAL
            profile.classification_reasons.append(f"Categorical ({profile.n_unique} unique values)")

            # Get top values
            value_counts = valid_data.value_counts().head(5)
            profile.top_values = list(value_counts.items())

            # Good target candidate if reasonable cardinality
            if 2 <= profile.n_unique <= 20:
                profile.target_score = self.thresholds["target_multiclass_boost"]
        else:
            # High cardinality text - probably not useful
            profile.statistical_type = StatisticalType.TEXT
            profile.role = ColumnRole.METADATA
            profile.is_usable = False
            profile.skip_reason = "High cardinality text"
            profile.classification_reasons.append(f"High cardinality categorical ({profile.n_unique} unique)")

    def classify_dataframe(self, df: pd.DataFrame) -> dict[str, ColumnProfile]:
        """
        Classify all columns in a DataFrame.

        Returns:
            Dictionary mapping column names to their profiles
        """
        profiles = {}

        for col in df.columns:
            profiles[col] = self.classify_column(df[col], df)

        # Post-processing: identify best target column
        self._identify_best_target(profiles, df)

        return profiles

    def _identify_best_target(self, profiles: dict[str, ColumnProfile], df: pd.DataFrame) -> None:
        """Identify the best column to use as prediction target."""
        candidates = []

        for name, profile in profiles.items():
            if not profile.is_usable:
                continue
            if profile.missing_pct > self.thresholds["target_max_missing"] * 100:
                continue

            score = profile.target_score

            # Boost for last column (common convention)
            if name == df.columns[-1]:
                score += self.thresholds["target_last_column_boost"]

            # Penalize columns with very low variance
            if profile.statistical_type == StatisticalType.CONTINUOUS:
                if profile.std is not None and profile.mean is not None and profile.mean != 0:
                    cv = profile.std / abs(profile.mean)
                    if cv < 0.05:  # Very low variance
                        score *= 0.5

            if score > 0:
                candidates.append((name, score, profile))

        # Sort by score and mark the best as target
        candidates.sort(key=lambda x: x[1], reverse=True)

        if candidates:
            best_name, best_score, best_profile = candidates[0]
            best_profile.role = ColumnRole.TARGET
            best_profile.classification_reasons.append(f"Best target candidate (score: {best_score:.2f})")

    def get_usable_features(self, profiles: dict[str, ColumnProfile]) -> list[str]:
        """Get list of columns usable as features."""
        return [name for name, profile in profiles.items() if profile.is_usable and profile.role == ColumnRole.FEATURE]

    def get_target_column(self, profiles: dict[str, ColumnProfile]) -> str | None:
        """Get the identified target column."""
        for name, profile in profiles.items():
            if profile.role == ColumnRole.TARGET:
                return name
        return None

    def get_analysis_summary(self, profiles: dict[str, ColumnProfile]) -> dict[str, Any]:
        """Generate a summary of the column classification."""
        summary = {
            "total_columns": len(profiles),
            "usable_features": 0,
            "identifiers": 0,
            "datetime_columns": 0,
            "geographic_columns": 0,
            "constant_columns": 0,
            "text_columns": 0,
            "target_column": None,
            "column_types": {
                "boolean": 0,
                "categorical": 0,
                "continuous": 0,
                "discrete": 0,
                "datetime": 0,
                "text": 0,
            },
        }

        for name, profile in profiles.items():
            # Count roles
            if profile.role == ColumnRole.IDENTIFIER:
                summary["identifiers"] += 1
            elif profile.role == ColumnRole.DATETIME:
                summary["datetime_columns"] += 1
            elif profile.role == ColumnRole.GEOGRAPHIC:
                summary["geographic_columns"] += 1
            elif profile.role == ColumnRole.CONSTANT:
                summary["constant_columns"] += 1
            elif profile.role == ColumnRole.METADATA:
                summary["text_columns"] += 1
            elif profile.role == ColumnRole.TARGET:
                summary["target_column"] = name

            if profile.is_usable:
                summary["usable_features"] += 1

            # Count statistical types
            type_key = profile.statistical_type.value
            if type_key in summary["column_types"]:
                summary["column_types"][type_key] += 1

        return summary


def classify_dataset(
    df: pd.DataFrame, custom_thresholds: dict | None = None
) -> tuple[dict[str, ColumnProfile], dict[str, Any]]:
    """
    Convenience function to classify a dataset.

    Returns:
        Tuple of (column_profiles, summary)
    """
    classifier = ColumnClassifier(custom_thresholds)
    profiles = classifier.classify_dataframe(df)
    summary = classifier.get_analysis_summary(profiles)

    return profiles, summary
