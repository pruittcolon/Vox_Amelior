"""
Schema Intelligence Layer - Core Module

This module provides automatic schema understanding and semantic type detection
for tabular datasets, enabling schema-agnostic analytics.

Author: Nemo Server ML Team
Date: 2025-11-27
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum

import pandas as pd

logger = logging.getLogger(__name__)


class SemanticType(Enum):
    """Semantic types for columns beyond basic dtypes."""

    IDENTIFIER = "identifier"  # Primary keys, IDs
    TEMPORAL = "temporal"  # Dates, timestamps
    NUMERIC_CONTINUOUS = "numeric_continuous"  # Revenue, cost, measurements
    NUMERIC_DISCRETE = "numeric_discrete"  # Counts, quantities
    CATEGORICAL = "categorical"  # Categories, statuses
    TEXT = "text"  # Free-form text
    BOOLEAN = "boolean"  # Binary flags
    TARGET = "target"  # Potential prediction target
    UNKNOWN = "unknown"  # Could not classify


class BusinessEntity(Enum):
    """Common business entities detected via NER and pattern matching."""

    DATE = "date"
    TIMESTAMP = "timestamp"
    COST = "cost"
    REVENUE = "revenue"
    PRICE = "price"
    QUANTITY = "quantity"
    AMOUNT = "amount"
    CUSTOMER_ID = "customer_id"
    PRODUCT_ID = "product_id"
    TRANSACTION_ID = "transaction_id"
    CATEGORY = "category"
    STATUS = "status"
    NAME = "name"
    EMAIL = "email"
    PHONE = "phone"
    ADDRESS = "address"
    PERCENTAGE = "percentage"
    RATING = "rating"
    UNKNOWN = "unknown"


@dataclass
class ColumnProfile:
    """Complete profile of a single column."""

    name: str
    dtype: str
    semantic_type: SemanticType
    detected_entity: BusinessEntity
    cardinality: int
    uniqueness_ratio: float
    missing_ratio: float
    sample_values: list[str]

    # Statistical properties (for numeric columns)
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None

    # Distribution info
    is_normal: bool | None = None
    is_skewed: bool | None = None
    has_outliers: bool = False

    # Temporal properties
    is_sorted: bool | None = None
    has_seasonality: bool | None = None

    # Confidence score for classification
    confidence: float = 1.0


class ColumnProfiler:
    """
    Automatically profile dataset columns to detect semantic types and business entities.

    Uses statistical analysis, pattern matching, and heuristics to understand
    column semantics beyond basic pandas dtypes.
    """

    def __init__(self, sample_size: int = 1000):
        """
        Initialize the column profiler.

        Args:
            sample_size: Number of rows to sample for pattern detection
        """
        self.sample_size = sample_size

        # Pattern matchers for business entities
        self.patterns = {
            BusinessEntity.EMAIL: re.compile(r"^[\w\.-]+@[\w\.-]+\.\w+$"),
            BusinessEntity.PHONE: re.compile(r"^\+?[\d\s\-\(\)]+$"),
            BusinessEntity.PERCENTAGE: re.compile(r"^\d+\.?\d*%$"),
        }

        # Keyword matchers for column names
        self.name_keywords = {
            BusinessEntity.DATE: ["date", "day", "month", "year", "dt", "time"],
            BusinessEntity.TIMESTAMP: ["timestamp", "created_at", "updated_at", "datetime"],
            BusinessEntity.COST: ["cost", "expense", "spending", "spend"],
            BusinessEntity.REVENUE: ["revenue", "sales", "income"],
            BusinessEntity.PRICE: ["price", "amount", "value"],
            BusinessEntity.QUANTITY: ["quantity", "qty", "count", "num", "number"],
            BusinessEntity.CUSTOMER_ID: ["customer_id", "cust_id", "client_id", "user_id"],
            BusinessEntity.PRODUCT_ID: ["product_id", "prod_id", "item_id", "sku"],
            BusinessEntity.TRANSACTION_ID: ["transaction_id", "trans_id", "order_id", "invoice_id"],
            BusinessEntity.CATEGORY: ["category", "type", "class", "segment"],
            BusinessEntity.STATUS: ["status", "state", "flag"],
            BusinessEntity.RATING: ["rating", "score", "stars"],
        }

    def profile_column(self, series: pd.Series) -> ColumnProfile:
        """
        Profile a single column to detect its semantic type and properties.

        Args:
            series: Pandas Series to profile

        Returns:
            ColumnProfile with complete analysis
        """
        logger.debug(f"Profiling column: {series.name}")

        # Basic statistics
        cardinality = series.nunique()
        total_count = len(series)
        missing_count = series.isna().sum()
        uniqueness_ratio = cardinality / max(total_count, 1)
        missing_ratio = missing_count / max(total_count, 1)

        # Sample values for pattern matching
        sample_values = series.dropna().head(min(10, total_count)).astype(str).tolist()

        # Detect semantic type
        semantic_type, confidence = self._detect_semantic_type(series, uniqueness_ratio)

        # Detect business entity
        detected_entity = self._detect_business_entity(series.name, series, semantic_type)

        # Statistical properties for numeric columns (NOT boolean)
        mean, std, min_val, max_val = None, None, None, None
        is_normal, is_skewed, has_outliers = None, None, False

        if semantic_type in [SemanticType.NUMERIC_CONTINUOUS, SemanticType.NUMERIC_DISCRETE]:
            numeric_series = pd.to_numeric(series, errors="coerce").dropna()
            if len(numeric_series) > 0:
                mean = float(numeric_series.mean())
                std = float(numeric_series.std())
                min_val = float(numeric_series.min())
                max_val = float(numeric_series.max())

                # Outlier detection using IQR
                Q1 = numeric_series.quantile(0.25)
                Q3 = numeric_series.quantile(0.75)
                IQR = Q3 - Q1
                has_outliers = ((numeric_series < (Q1 - 1.5 * IQR)) | (numeric_series > (Q3 + 1.5 * IQR))).any()

        # Temporal properties
        is_sorted, has_seasonality = None, None
        if semantic_type == SemanticType.TEMPORAL:
            try:
                dt_series = pd.to_datetime(series, errors="coerce").dropna()
                if len(dt_series) > 1:
                    is_sorted = dt_series.is_monotonic_increasing
            except:
                pass

        return ColumnProfile(
            name=series.name,
            dtype=str(series.dtype),
            semantic_type=semantic_type,
            detected_entity=detected_entity,
            cardinality=cardinality,
            uniqueness_ratio=uniqueness_ratio,
            missing_ratio=missing_ratio,
            sample_values=sample_values,
            mean=mean,
            std=std,
            min=min_val,
            max=max_val,
            is_normal=is_normal,
            is_skewed=is_skewed,
            has_outliers=has_outliers,
            is_sorted=is_sorted,
            has_seasonality=has_seasonality,
            confidence=confidence,
        )

    def _detect_semantic_type(self, series: pd.Series, uniqueness: float) -> tuple[SemanticType, float]:
        """
        Detect the semantic type of a column using heuristics.

        PRIORITY ORDER:
        1. Explicit types (datetime, boolean)
        2. Business entity hints (cost/price/revenue keywords)
        3. Statistical properties (uniqueness, cardinality)

        Args:
            series: Pandas Series to analyze
            uniqueness: Uniqueness ratio (unique_values / total_values)

        Returns:
            Tuple of (SemanticType, confidence_score)
        """
        dtype = series.dtype
        col_name = series.name.lower() if series.name else ""

        # Temporal detection (highest priority)
        if pd.api.types.is_datetime64_any_dtype(series):
            return SemanticType.TEMPORAL, 1.0

        # Try parsing as datetime
        if dtype == "object":
            sample = series.dropna().head(100)
            try:
                pd.to_datetime(sample, errors="raise")
                return SemanticType.TEMPORAL, 0.9
            except:
                pass

        # Boolean - check BEFORE numeric since bool is also numeric
        if pd.api.types.is_bool_dtype(series):
            return SemanticType.BOOLEAN, 1.0

        # Numeric types (after boolean check)
        if pd.api.types.is_numeric_dtype(series):
            # BUSINESS ENTITY PRIORITY: Check for cost/price/revenue keywords
            # This prevents misclassification of financial columns as identifiers
            value_keywords = [
                "cost",
                "price",
                "amount",
                "revenue",
                "sales",
                "spend",
                "expense",
                "payment",
                "value",
                "total",
                "sum",
                "profit",
            ]
            has_value_keyword = any(keyword in col_name for keyword in value_keywords)

            # If column name suggests it's a value/cost, treat as continuous
            if has_value_keyword:
                return SemanticType.NUMERIC_CONTINUOUS, 0.95

            # Check if discrete (integers with low cardinality)
            if pd.api.types.is_integer_dtype(series) and uniqueness < 0.05:
                return SemanticType.NUMERIC_DISCRETE, 0.9

            # Check if likely an identifier (high uniqueness AND sequential pattern)
            # FIXED: Only classify as identifier if extremely high uniqueness (>0.98) OR sequential integers
            if uniqueness > 0.98:
                # Exception: if it's a float, probably not an ID
                if not pd.api.types.is_integer_dtype(series):
                    return SemanticType.NUMERIC_CONTINUOUS, 0.9
                return SemanticType.IDENTIFIER, 0.85

            # Check for sequential IDs (strong identifier signal)
            if pd.api.types.is_integer_dtype(series) and uniqueness > 0.95:
                numeric_series = series.dropna().sort_values()
                if len(numeric_series) > 1:
                    diffs = numeric_series.diff().dropna()
                    if (diffs == 1).sum() / max(len(diffs), 1) > 0.9:
                        return SemanticType.IDENTIFIER, 0.95

            # Otherwise continuous
            return SemanticType.NUMERIC_CONTINUOUS, 0.95

        # Categorical vs Text
        if dtype == "object" or pd.api.types.is_categorical_dtype(series):
            # High uniqueness suggests identifier
            if uniqueness > 0.95:
                return SemanticType.IDENTIFIER, 0.8

            # Low cardinality suggests categorical
            if uniqueness < 0.05 or series.nunique() < 50:
                return SemanticType.CATEGORICAL, 0.9

            # Check average string length
            avg_length = series.dropna().astype(str).str.len().mean()
            if avg_length > 50:
                return SemanticType.TEXT, 0.85
            else:
                return SemanticType.CATEGORICAL, 0.7

        return SemanticType.UNKNOWN, 0.5

    def _detect_business_entity(self, col_name: str, series: pd.Series, semantic_type: SemanticType) -> BusinessEntity:
        """
        Detect business entity based on column name and content patterns.

        Args:
            col_name: Column name
            series: Column data
            semantic_type: Already detected semantic type

        Returns:
            BusinessEntity enum value
        """
        col_lower = col_name.lower()

        # Check column name keywords
        for entity, keywords in self.name_keywords.items():
            if any(keyword in col_lower for keyword in keywords):
                return entity

        # Pattern-based detection on sample values
        sample = series.dropna().head(100).astype(str)

        for entity, pattern in self.patterns.items():
            matches = sample.str.match(pattern).sum()
            if matches / max(len(sample), 1) > 0.8:
                return entity

        # Fallback based on semantic type
        if semantic_type == SemanticType.TEMPORAL:
            return BusinessEntity.DATE
        elif semantic_type == SemanticType.IDENTIFIER:
            return BusinessEntity.TRANSACTION_ID
        elif semantic_type == SemanticType.CATEGORICAL:
            return BusinessEntity.CATEGORY

        return BusinessEntity.UNKNOWN

    def profile_dataset(self, df: pd.DataFrame) -> dict[str, ColumnProfile]:
        """
        Profile all columns in a dataset.

        Args:
            df: Pandas DataFrame to profile

        Returns:
            Dictionary mapping column names to their profiles
        """
        logger.info(f"Profiling dataset with {len(df.columns)} columns, {len(df)} rows")

        profiles = {}
        for col in df.columns:
            try:
                profiles[col] = self.profile_column(df[col])
            except Exception as e:
                logger.error(f"Error profiling column {col}: {e}")
                # Create a minimal profile
                profiles[col] = ColumnProfile(
                    name=col,
                    dtype=str(df[col].dtype),
                    semantic_type=SemanticType.UNKNOWN,
                    detected_entity=BusinessEntity.UNKNOWN,
                    cardinality=0,
                    uniqueness_ratio=0.0,
                    missing_ratio=1.0,
                    sample_values=[],
                    confidence=0.0,
                )

        return profiles


def summarize_profiles(profiles: dict[str, ColumnProfile]) -> dict:
    """
    Create a summary of column profiles for quick overview.

    Args:
        profiles: Dictionary of column profiles

    Returns:
        Summary statistics dictionary
    """
    summary = {
        "total_columns": len(profiles),
        "semantic_types": {},
        "business_entities": {},
        "missing_data_columns": [],
        "high_cardinality_columns": [],
        "identifier_columns": [],
        "temporal_columns": [],
        "numeric_columns": [],
        "categorical_columns": [],
    }

    for col, profile in profiles.items():
        # Count semantic types
        sem_type = profile.semantic_type.value
        summary["semantic_types"][sem_type] = summary["semantic_types"].get(sem_type, 0) + 1

        # Count business entities
        entity = profile.detected_entity.value
        summary["business_entities"][entity] = summary["business_entities"].get(entity, 0) + 1

        # Flag important columns
        if profile.missing_ratio > 0.5:
            summary["missing_data_columns"].append(col)

        if profile.uniqueness_ratio > 0.95:
            summary["high_cardinality_columns"].append(col)

        if profile.semantic_type == SemanticType.IDENTIFIER:
            summary["identifier_columns"].append(col)
        elif profile.semantic_type == SemanticType.TEMPORAL:
            summary["temporal_columns"].append(col)
        elif profile.semantic_type in [SemanticType.NUMERIC_CONTINUOUS, SemanticType.NUMERIC_DISCRETE]:
            summary["numeric_columns"].append(col)
        elif profile.semantic_type == SemanticType.CATEGORICAL:
            summary["categorical_columns"].append(col)

    return summary
