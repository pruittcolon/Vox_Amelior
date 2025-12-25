"""
Unit Tests for Schema Intelligence Layer

Tests for ColumnProfiler, DatasetClassifier, and ApplicabilityScorer.

Author: Nemo Server ML Team
Date: 2025-11-27
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from core.applicability_scorer import AnalyticsEngine, ApplicabilityScorer
from core.dataset_classifier import BusinessDomain, DatasetClassifier, MLTaskType
from core.schema_intelligence import BusinessEntity, ColumnProfiler, SemanticType, summarize_profiles


class TestColumnProfiler(unittest.TestCase):
    """Test cases for ColumnProfiler."""

    def setUp(self):
        """Set up test fixtures."""
        self.profiler = ColumnProfiler()

    def test_numeric_continuous_detection(self):
        """Test detection of continuous numeric columns."""
        series = pd.Series([1.5, 2.3, 4.7, 5.9, 3.2], name="revenue")
        profile = self.profiler.profile_column(series)

        self.assertEqual(profile.semantic_type, SemanticType.NUMERIC_CONTINUOUS)
        self.assertIsNotNone(profile.mean)
        self.assertIsNotNone(profile.std)

    def test_categorical_detection(self):
        """Test detection of categorical columns."""
        series = pd.Series(["A", "B", "A", "C", "B", "A"], name="category")
        profile = self.profiler.profile_column(series)

        self.assertEqual(profile.semantic_type, SemanticType.CATEGORICAL)
        self.assertEqual(profile.cardinality, 3)
        self.assertLessEqual(profile.uniqueness_ratio, 0.5)

    def test_temporal_detection(self):
        """Test detection of temporal columns."""
        dates = pd.date_range("2024-01-01", periods=10)
        series = pd.Series(dates, name="date")
        profile = self.profiler.profile_column(series)

        self.assertEqual(profile.semantic_type, SemanticType.TEMPORAL)
        self.assertTrue(profile.is_sorted)

    def test_identifier_detection(self):
        """Test detection of identifier columns."""
        series = pd.Series(range(1000, 1100), name="customer_id")
        profile = self.profiler.profile_column(series)

        self.assertEqual(profile.semantic_type, SemanticType.IDENTIFIER)
        self.assertGreater(profile.uniqueness_ratio, 0.9)

    def test_business_entity_detection_cost(self):
        """Test detection of cost business entity."""
        series = pd.Series([100, 200, 150], name="total_cost")
        profile = self.profiler.profile_column(series)

        self.assertEqual(profile.detected_entity, BusinessEntity.COST)

    def test_business_entity_detection_date(self):
        """Test detection of date business entity."""
        series = pd.Series(pd.date_range("2024-01-01", periods=10), name="transaction_date")
        profile = self.profiler.profile_column(series)

        self.assertEqual(profile.detected_entity, BusinessEntity.DATE)

    def test_missing_data_handling(self):
        """Test handling of missing data."""
        series = pd.Series([1, 2, None, 4, None, 6], name="values")
        profile = self.profiler.profile_column(series)

        self.assertAlmostEqual(profile.missing_ratio, 2 / 6, places=2)

    def test_outlier_detection(self):
        """Test outlier detection."""
        # Create data with clear outliers
        data = [10, 12, 11, 13, 10, 100, 200]  # 100, 200 are outliers
        series = pd.Series(data, name="values")
        profile = self.profiler.profile_column(series)

        self.assertTrue(profile.has_outliers)

    def test_dataset_profiling(self):
        """Test profiling an entire dataset."""
        df = pd.DataFrame(
            {
                "customer_id": range(100),
                "purchase_date": pd.date_range("2024-01-01", periods=100),
                "amount": np.random.uniform(10, 100, 100),
                "category": np.random.choice(["A", "B", "C"], 100),
            }
        )

        profiles = self.profiler.profile_dataset(df)

        self.assertEqual(len(profiles), 4)
        self.assertIn("customer_id", profiles)
        self.assertEqual(profiles["purchase_date"].semantic_type, SemanticType.TEMPORAL)
        self.assertEqual(profiles["amount"].semantic_type, SemanticType.NUMERIC_CONTINUOUS)
        self.assertEqual(profiles["category"].semantic_type, SemanticType.CATEGORICAL)

    def test_summarize_profiles(self):
        """Test profile summarization."""
        df = pd.DataFrame(
            {
                "id": range(50),
                "date": pd.date_range("2024-01-01", periods=50),
                "revenue": np.random.uniform(100, 1000, 50),
                "cost": np.random.uniform(50, 500, 50),
                "category": np.random.choice(["X", "Y"], 50),
            }
        )

        profiles = self.profiler.profile_dataset(df)
        summary = summarize_profiles(profiles)

        self.assertEqual(summary["total_columns"], 5)
        self.assertIn("temporal", summary["semantic_types"])
        self.assertGreater(len(summary["numeric_columns"]), 0)
        self.assertGreater(len(summary["temporal_columns"]), 0)


class TestDatasetClassifier(unittest.TestCase):
    """Test cases for DatasetClassifier."""

    def setUp(self):
        """Set up test fixtures."""
        self.classifier = DatasetClassifier()
        self.profiler = ColumnProfiler()

    def test_finance_domain_detection(self):
        """Test detection of financial domain."""
        df = pd.DataFrame(
            {
                "transaction_id": range(100),
                "date": pd.date_range("2024-01-01", periods=100),
                "amount": np.random.uniform(10, 1000, 100),
                "cost": np.random.uniform(5, 500, 100),
            }
        )

        profiles = self.profiler.profile_dataset(df)
        classification = self.classifier.classify(profiles, len(df))

        self.assertIn(classification.domain, [BusinessDomain.FINANCE, BusinessDomain.SALES])
        self.assertGreater(classification.domain_confidence, 0.3)

    def test_ecommerce_domain_detection(self):
        """Test detection of e-commerce domain."""
        df = pd.DataFrame(
            {
                "customer_id": range(100),
                "product_id": np.random.randint(1, 50, 100),
                "price": np.random.uniform(10, 100, 100),
                "quantity": np.random.randint(1, 10, 100),
            }
        )

        profiles = self.profiler.profile_dataset(df)
        classification = self.classifier.classify(profiles, len(df))

        # Should detect ecommerce or sales domain
        self.assertIn(classification.domain, [BusinessDomain.ECOMMERCE, BusinessDomain.SALES])

    def test_time_series_task_detection(self):
        """Test detection of time series task."""
        df = pd.DataFrame(
            {"date": pd.date_range("2024-01-01", periods=100), "revenue": np.random.uniform(1000, 5000, 100)}
        )

        profiles = self.profiler.profile_dataset(df)
        classification = self.classifier.classify(profiles, len(df))

        self.assertIn(MLTaskType.TIME_SERIES, classification.task_types)
        self.assertIn(MLTaskType.FORECASTING, classification.task_types)

    def test_classification_task_detection(self):
        """Test detection of classification task."""
        df = pd.DataFrame(
            {
                "feature1": np.random.uniform(0, 1, 100),
                "feature2": np.random.uniform(0, 1, 100),
                "target": np.random.choice(["Yes", "No"], 100),
            }
        )

        profiles = self.profiler.profile_dataset(df)
        classification = self.classifier.classify(profiles, len(df))

        self.assertIn(MLTaskType.CLASSIFICATION, classification.task_types)
        self.assertTrue(classification.has_target)
        self.assertIn("target", classification.potential_targets)

    def test_recommendations_generation(self):
        """Test generation of recommendations."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=100),
                "revenue": np.random.uniform(1000, 5000, 100),
                "cost": np.random.uniform(500, 2000, 100),
            }
        )

        profiles = self.profiler.profile_dataset(df)
        classification = self.classifier.classify(profiles, len(df))

        self.assertGreater(len(classification.recommendations), 0)
        self.assertTrue(
            any("time-series" in rec.lower() or "forecast" in rec.lower() for rec in classification.recommendations)
        )


class TestApplicabilityScorer(unittest.TestCase):
    """Test cases for ApplicabilityScorer."""

    def setUp(self):
        """Set up test fixtures."""
        self.scorer = ApplicabilityScorer()
        self.profiler = ColumnProfiler()
        self.classifier = DatasetClassifier()

    def test_cost_optimization_scoring(self):
        """Test scoring for cost optimization engine."""
        df = pd.DataFrame(
            {
                "category": np.random.choice(["A", "B", "C"], 100),
                "cost": np.random.uniform(100, 1000, 100),
                "date": pd.date_range("2024-01-01", periods=100),
            }
        )

        profiles = self.profiler.profile_dataset(df)
        classification = self.classifier.classify(profiles, len(df))
        scores = self.scorer.score_engines(profiles, classification)

        # Should recommend cost optimization
        cost_opt_engines = [s for s in scores if s.engine == AnalyticsEngine.COST_OPTIMIZATION]
        self.assertGreater(len(cost_opt_engines), 0)
        if cost_opt_engines:
            self.assertGreater(cost_opt_engines[0].applicability_score, 0.4)

    def test_revenue_forecast_scoring(self):
        """Test scoring for revenue forecasting engine."""
        df = pd.DataFrame(
            {"date": pd.date_range("2024-01-01", periods=100), "revenue": np.random.uniform(1000, 5000, 100)}
        )

        profiles = self.profiler.profile_dataset(df)
        classification = self.classifier.classify(profiles, len(df))
        scores = self.scorer.score_engines(profiles, classification)

        # Should highly recommend revenue forecasting
        forecast_engines = [s for s in scores if s.engine == AnalyticsEngine.REVENUE_FORECAST]
        if forecast_engines:
            self.assertGreater(forecast_engines[0].applicability_score, 0.5)

    def test_market_basket_scoring(self):
        """Test scoring for market basket analysis."""
        df = pd.DataFrame(
            {
                "transaction_id": np.repeat(range(50), 3),
                "product_id": np.random.choice(["P1", "P2", "P3", "P4", "P5"], 150),
                "customer_id": np.random.randint(1, 20, 150),
            }
        )

        profiles = self.profiler.profile_dataset(df)
        classification = self.classifier.classify(profiles, len(df))
        scores = self.scorer.score_engines(profiles, classification)

        # Should recommend market basket
        basket_engines = [s for s in scores if s.engine == AnalyticsEngine.MARKET_BASKET]
        if basket_engines:
            self.assertGreater(basket_engines[0].applicability_score, 0.3)

    def test_top_recommendations(self):
        """Test getting top N recommendations."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=100),
                "revenue": np.random.uniform(1000, 5000, 100),
                "cost": np.random.uniform(500, 2000, 100),
                "category": np.random.choice(["A", "B"], 100),
            }
        )

        profiles = self.profiler.profile_dataset(df)
        classification = self.classifier.classify(profiles, len(df))

        from core.applicability_scorer import get_top_recommendations

        top_5 = get_top_recommendations(profiles, classification, top_n=5)

        self.assertLessEqual(len(top_5), 5)
        # Should be sorted by score
        for i in range(len(top_5) - 1):
            self.assertGreaterEqual(top_5[i].applicability_score, top_5[i + 1].applicability_score)

    def test_insufficient_data_penalty(self):
        """Test that engines are penalized for insufficient data."""
        df = pd.DataFrame(
            {
                "value": [1, 2, 3]  # Only 3 rows
            }
        )

        profiles = self.profiler.profile_dataset(df)
        classification = self.classifier.classify(profiles, len(df))
        scores = self.scorer.score_engines(profiles, classification)

        # All scores should have reduced confidence due to low row count
        for score in scores:
            self.assertLess(score.confidence, 1.0)


if __name__ == "__main__":
    unittest.main()
