"""
Dataset Classifier - Domain and Task Type Detection

Automatically classifies datasets into business domains and ML task types
using LLM integration and heuristic rules.

Author: Nemo Server ML Team
Date: 2025-11-27
"""

import logging
from dataclasses import dataclass
from enum import Enum

from .schema_intelligence import BusinessEntity, ColumnProfile, SemanticType

logger = logging.getLogger(__name__)


class BusinessDomain(Enum):
    """Common business domains for dataset classification."""

    ECOMMERCE = "ecommerce"
    FINANCE = "finance"
    HEALTHCARE = "healthcare"
    HR = "human_resources"
    MARKETING = "marketing"
    OPERATIONS = "operations"
    SALES = "sales"
    SUPPLY_CHAIN = "supply_chain"
    IOT = "iot"
    TELECOM = "telecommunications"
    RETAIL = "retail"
    UNKNOWN = "unknown"


class MLTaskType(Enum):
    """ML task types that can be performed on the dataset."""

    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    TIME_SERIES = "time_series"
    CLUSTERING = "clustering"
    ASSOCIATION_MINING = "association_mining"
    ANOMALY_DETECTION = "anomaly_detection"
    FORECASTING = "forecasting"
    SURVIVAL_ANALYSIS = "survival_analysis"
    UNKNOWN = "unknown"


@dataclass
class DatasetClassification:
    """Complete classification of a dataset."""

    domain: BusinessDomain
    domain_confidence: float
    task_types: list[MLTaskType]
    task_confidences: dict[MLTaskType, float]
    row_count: int
    column_count: int
    has_target: bool
    potential_targets: list[str]
    has_temporal: bool
    temporal_columns: list[str]
    recommendations: list[str]


class DatasetClassifier:
    """
    Classify datasets into business domains and identify applicable ML task types.

    Uses column profiles, business entity detection, and heuristic rules to determine
    what kind of analysis would be most appropriate.
    """

    def __init__(self):
        """Initialize the dataset classifier."""
        # Domain indicators based on business entities
        self.domain_indicators = {
            BusinessDomain.ECOMMERCE: [
                BusinessEntity.PRODUCT_ID,
                BusinessEntity.CUSTOMER_ID,
                BusinessEntity.PRICE,
                BusinessEntity.QUANTITY,
            ],
            BusinessDomain.FINANCE: [
                BusinessEntity.TRANSACTION_ID,
                BusinessEntity.AMOUNT,
                BusinessEntity.COST,
                BusinessEntity.REVENUE,
            ],
            BusinessDomain.SALES: [BusinessEntity.CUSTOMER_ID, BusinessEntity.REVENUE, BusinessEntity.PRODUCT_ID],
            BusinessDomain.OPERATIONS: [BusinessEntity.TIMESTAMP, BusinessEntity.STATUS, BusinessEntity.QUANTITY],
        }

    def classify(self, profiles: dict[str, ColumnProfile], row_count: int) -> DatasetClassification:
        """
        Classify a dataset based on its column profiles.

        Args:
            profiles: Dictionary of column profiles
            row_count: Number of rows in the dataset

        Returns:
            DatasetClassification with domain and task type predictions
        """
        logger.info(f"Classifying dataset with {len(profiles)} columns, {row_count} rows")

        # Detect business domain
        domain, domain_confidence = self._detect_domain(profiles)

        # Identify potential task types
        task_types, task_confidences = self._identify_tasks(profiles, row_count)

        # Find potential target columns
        has_target, potential_targets = self._find_potential_targets(profiles)

        # Check for temporal data
        temporal_columns = [col for col, prof in profiles.items() if prof.semantic_type == SemanticType.TEMPORAL]
        has_temporal = len(temporal_columns) > 0

        # Generate recommendations
        recommendations = self._generate_recommendations(domain, task_types, has_temporal, row_count, len(profiles))

        return DatasetClassification(
            domain=domain,
            domain_confidence=domain_confidence,
            task_types=task_types,
            task_confidences=task_confidences,
            row_count=row_count,
            column_count=len(profiles),
            has_target=has_target,
            potential_targets=potential_targets,
            has_temporal=has_temporal,
            temporal_columns=temporal_columns,
            recommendations=recommendations,
        )

    def _detect_domain(self, profiles: dict[str, ColumnProfile]) -> tuple[BusinessDomain, float]:
        """
        Detect the business domain based on business entities present.

        Args:
            profiles: Column profiles

        Returns:
            Tuple of (BusinessDomain, confidence_score)
        """
        entity_counts = {}

        # Count entities across all columns
        for prof in profiles.values():
            entity = prof.detected_entity
            entity_counts[entity] = entity_counts.get(entity, 0) + 1

        # Score each domain
        domain_scores = {}
        for domain, indicators in self.domain_indicators.items():
            score = sum(entity_counts.get(entity, 0) for entity in indicators)
            if score > 0:
                domain_scores[domain] = score / len(indicators)

        if not domain_scores:
            return BusinessDomain.UNKNOWN, 0.3

        # Get highest scoring domain
        best_domain = max(domain_scores, key=domain_scores.get)
        confidence = min(domain_scores[best_domain], 1.0)

        return best_domain, confidence

    def _identify_tasks(
        self, profiles: dict[str, ColumnProfile], row_count: int
    ) -> tuple[list[MLTaskType], dict[MLTaskType, float]]:
        """
        Identify applicable ML task types based on data characteristics.

        Args:
            profiles: Column profiles
            row_count: Number of rows

        Returns:
            Tuple of (task_types_list, confidence_dict)
        """
        tasks = []
        confidences = {}

        # Count column types
        has_numeric = any(
            p.semantic_type in [SemanticType.NUMERIC_CONTINUOUS, SemanticType.NUMERIC_DISCRETE]
            for p in profiles.values()
        )
        has_categorical = any(p.semantic_type == SemanticType.CATEGORICAL for p in profiles.values())
        has_temporal = any(p.semantic_type == SemanticType.TEMPORAL for p in profiles.values())
        has_high_card = any(p.uniqueness_ratio > 0.95 for p in profiles.values())

        # Time series / Forecasting
        if has_temporal and has_numeric:
            if row_count > 50:  # Need sufficient history
                tasks.append(MLTaskType.TIME_SERIES)
                confidences[MLTaskType.TIME_SERIES] = 0.9
                tasks.append(MLTaskType.FORECASTING)
                confidences[MLTaskType.FORECASTING] = 0.85

        # Classification / Regression
        if has_categorical or has_numeric:
            # Look for low-cardinality categorical (potential target)
            low_card_cats = [
                p for p in profiles.values() if p.semantic_type == SemanticType.CATEGORICAL and p.cardinality < 20
            ]
            if low_card_cats:
                tasks.append(MLTaskType.CLASSIFICATION)
                confidences[MLTaskType.CLASSIFICATION] = 0.8

            if has_numeric:
                tasks.append(MLTaskType.REGRESSION)
                confidences[MLTaskType.REGRESSION] = 0.75

        # Clustering
        if row_count > 100 and (has_numeric or has_categorical):
            tasks.append(MLTaskType.CLUSTERING)
            confidences[MLTaskType.CLUSTERING] = 0.7

        # Association Mining (for transactional data)
        if has_high_card and has_categorical:
            tasks.append(MLTaskType.ASSOCIATION_MINING)
            confidences[MLTaskType.ASSOCIATION_MINING] = 0.65

        # Anomaly Detection
        if has_numeric and row_count > 100:
            tasks.append(MLTaskType.ANOMALY_DETECTION)
            confidences[MLTaskType.ANOMALY_DETECTION] = 0.7

        return tasks, confidences

    def _find_potential_targets(self, profiles: dict[str, ColumnProfile]) -> tuple[bool, list[str]]:
        """
        Identify columns that could serve as prediction targets.

        Args:
            profiles: Column profiles

        Returns:
            Tuple of (has_target, list_of_target_columns)
        """
        potential_targets = []

        for col, prof in profiles.items():
            # Low cardinality categoricals are good classification targets
            if (
                prof.semantic_type == SemanticType.CATEGORICAL
                and 2 <= prof.cardinality <= 20
                and prof.missing_ratio < 0.1
            ):
                potential_targets.append(col)

            # Continuous numerics are good regression targets
            elif prof.semantic_type == SemanticType.NUMERIC_CONTINUOUS and prof.missing_ratio < 0.1:
                # Prefer columns with business meaning
                if prof.detected_entity in [
                    BusinessEntity.REVENUE,
                    BusinessEntity.COST,
                    BusinessEntity.PRICE,
                    BusinessEntity.RATING,
                ]:
                    potential_targets.insert(0, col)  # Priority
                else:
                    potential_targets.append(col)

        return len(potential_targets) > 0, potential_targets[:5]  # Top 5

    def _generate_recommendations(
        self, domain: BusinessDomain, task_types: list[MLTaskType], has_temporal: bool, row_count: int, col_count: int
    ) -> list[str]:
        """
        Generate human-readable recommendations for analysis.

        Args:
            domain: Detected business domain
            task_types: Applicable ML task types
            has_temporal: Whether dataset has time component
            row_count: Number of rows
            col_count: Number of columns

        Returns:
            List of recommendation strings
        """
        recs = []

        # Domain-specific recommendations
        if domain == BusinessDomain.FINANCE:
            recs.append("💰 Financial data detected - Cost Optimization and Budget Variance engines recommended")
        elif domain == BusinessDomain.ECOMMERCE:
            recs.append("🛒 E-commerce data detected - Market Basket and Customer LTV engines recommended")
        elif domain == BusinessDomain.SALES:
            recs.append("📊 Sales data detected - Revenue Forecasting and Profit Margin engines recommended")

        # Task-based recommendations
        if MLTaskType.TIME_SERIES in task_types:
            recs.append("📈 Time-series analysis possible - Try Revenue Forecasting or Spend Pattern engines")

        if MLTaskType.CLASSIFICATION in task_types:
            recs.append("🎯 Classification task detected - Customer Churn prediction available")

        if MLTaskType.ASSOCIATION_MINING in task_types:
            recs.append("🔗 Transactional patterns detected - Market Basket Analysis recommended")

        # Data quality recommendations
        if row_count < 100:
            recs.append("⚠️ Small dataset (<100 rows) - Some ML models may not be reliable")

        if col_count > 50:
            recs.append("📊 High-dimensional data - Feature selection may improve performance")

        return recs
