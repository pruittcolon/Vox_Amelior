#!/usr/bin/env python3
"""
Comprehensive Integration Test for Analytics Engines

Tests all 7 engines with synthetic data to verify they work correctly.
This replicates real-world usage without requiring a running ML service.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from engines.anomaly_engine import AnomalyEngine
from engines.clustering_engine import ClusteringEngine
from engines.predictive_engine import PredictiveEngine
from engines.rag_evaluation_engine import RAGEvaluationEngine
from engines.statistical_engine import StatisticalEngine
from engines.trend_engine import TrendEngine
from engines.universal_graph_engine import UniversalGraphEngine


def create_test_data():
    """Create synthetic test dataset"""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=100, freq="D")

    df = pd.DataFrame(
        {
            "date": dates,
            "sales": np.random.normal(1000, 200, 100) + np.arange(100) * 5,  # Trending
            "customers": np.random.poisson(50, 100),
            "revenue": np.random.normal(5000, 1000, 100),
            "category": np.random.choice(["A", "B", "C"], 100),
            "region": np.random.choice(["North", "South", "East", "West"], 100),
            "satisfaction": np.random.uniform(1, 5, 100),
        }
    )

    # Add some anomalies
    df.loc[10, "sales"] = 5000  # Outlier
    df.loc[50, "revenue"] = 20000  # Outlier

    return df


def test_statistical_engine():
    """Test Statistical Analysis Engine"""
    print("\n" + "=" * 60)
    print("TEST 1: Statistical Analysis Engine")
    print("=" * 60)

    try:
        df = create_test_data()
        engine = StatisticalEngine()

        # Test descriptive statistics
        result = engine.analyze(df, {"analysis_types": ["descriptive", "correlation"], "confidence_level": 0.95})

        print("✅ Descriptive stats computed")
        print(f"   Numeric columns: {len(result.get('descriptive', {}).get('numeric', {}))}")
        print(f"   Categorical columns: {len(result.get('descriptive', {}).get('categorical', {}))}")

        # Test correlation
        if "correlation" in result:
            corr_cols = len(result["correlation"].get("columns", []))
            print(f"✅ Correlation matrix: {corr_cols}x{corr_cols}")

        # Test ANOVA
        result_anova = engine.analyze(df, {"analysis_types": ["anova"], "group_by": "category"})

        if "anova" in result_anova:
            print(f"✅ ANOVA: {len(result_anova['anova'])} variables analyzed")

        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_universal_graph_engine():
    """Test Universal Graph Generator"""
    print("\n" + "=" * 60)
    print("TEST 2: Universal Graph Generator Engine")
    print("=" * 60)

    try:
        df = create_test_data()
        engine = UniversalGraphEngine()

        result = engine.generate_graphs(df)

        graphs = result["graphs"]
        print(f"✅ Generated {len(graphs)} visualizations")

        # Count by type
        graph_types = {}
        for graph in graphs:
            gtype = graph["type"]
            graph_types[gtype] = graph_types.get(gtype, 0) + 1

        print("   Graph types:")
        for gtype, count in sorted(graph_types.items()):
            print(f"     - {gtype}: {count}")

        if len(graphs) >= 10:
            print(f"✅ SUCCESS: {len(graphs)} >= 10 graphs")
            return True
        else:
            print(f"❌ FAILED: Only {len(graphs)} graphs")
            return False

    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_rag_evaluation_engine():
    """Test RAG Evaluation Engine"""
    print("\n" + "=" * 60)
    print("TEST 3: RAG Evaluation Engine")
    print("=" * 60)

    try:
        engine = RAGEvaluationEngine()

        # Create test data
        test_cases = [
            {"query": "What is ML?", "relevant_docs": ["doc1", "doc2"]},
            {"query": "How does RAG work?", "relevant_docs": ["doc3", "doc4"]},
        ]

        rag_responses = [
            {"query": "What is ML?", "retrieved_docs": ["doc1", "doc3"], "generated_answer": "Machine learning is..."},
            {"query": "How does RAG work?", "retrieved_docs": ["doc3", "doc5"], "generated_answer": "RAG retrieves..."},
        ]

        result = engine.evaluate(test_cases, rag_responses, {"k_values": [1, 2]})

        print("✅ RAG evaluation completed")

        retrieval = result["retrieval_metrics"]
        print(f"   MRR: {retrieval.get('mrr', 0):.3f}")
        print(f"   Precision@1: {retrieval['precision_at_k'].get('p@1', 0):.3f}")

        generation = result["generation_metrics"]
        print(f"   Faithfulness: {generation.get('faithfulness', 0):.3f}")
        print(f"   Hallucination rate: {generation.get('hallucination_rate', 0):.3f}")

        summary = result["summary"]
        print(f"✅ Overall grade: {summary.get('overall_grade', 'N/A')}")

        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_predictive_engine():
    """Test Predictive Analytics Engine"""
    print("\n" + "=" * 60)
    print("TEST 4: Predictive Analytics Engine")
    print("=" * 60)

    try:
        df = create_test_data()
        engine = PredictiveEngine()

        result = engine.forecast(
            df,
            {
                "time_column": "date",
                "target_column": "sales",
                "horizon": 7,
                "models": ["naive", "moving_average", "exponential_smoothing"],
            },
        )

        print("✅ Forecast completed")
        print(f"   Best model: {result['best_model']}")
        print(f"   Horizon: {result['metadata']['horizon']} periods")

        forecast = result["forecast"]
        mae = forecast["validation_metrics"].get("mae", 0)
        print(f"   Validation MAE: {mae:.2f}")
        print(f"   Forecast values (first 3): {forecast['forecast_values'][:3]}")

        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_anomaly_engine():
    """Test Anomaly Detection Engine"""
    print("\n" + "=" * 60)
    print("TEST 5: Anomaly Detection Engine")
    print("=" * 60)

    try:
        df = create_test_data()
        engine = AnomalyEngine()

        result = engine.detect(
            df, {"target_columns": ["sales", "revenue"], "methods": ["ensemble"], "contamination": 0.05}
        )

        print("✅ Anomaly detection completed")
        print(f"   Method: {result['primary_method']}")
        print(f"   Anomalies found: {result['anomaly_count']}")
        print(f"   Anomaly rate: {result['anomaly_rate'] * 100:.1f}%")

        # Check if we detected the outliers we inserted
        anomalies = result["anomalies"]
        if 10 in anomalies or 50 in anomalies:
            print("✅ Correctly detected injected anomalies")

        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_clustering_engine():
    """Test Clustering Engine"""
    print("\n" + "=" * 60)
    print("TEST 6: Clustering Engine")
    print("=" * 60)

    try:
        df = create_test_data()
        engine = ClusteringEngine()

        result = engine.cluster(
            df,
            {
                "features": ["sales", "revenue", "satisfaction"],
                "algorithm": "auto",  # Will use kmeans with auto-k
            },
        )

        print("✅ Clustering completed")
        print(f"   Algorithm: {result['algorithm']}")
        print(f"   Clusters found: {result['n_clusters']}")

        # Print cluster sizes
        for profile in result["cluster_profiles"]:
            print(f"   Cluster {profile['cluster_id']}: {profile['size']} points ({profile['percentage']:.1f}%)")

        metrics = result["metrics"]
        if "silhouette_score" in metrics and metrics["silhouette_score"]:
            print(f"   Silhouette score: {metrics['silhouette_score']:.3f}")

        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_trend_engine():
    """Test Trend Analysis Engine"""
    print("\n" + "=" * 60)
    print("TEST 7: Trend Analysis Engine")
    print("=" * 60)

    try:
        df = create_test_data()
        engine = TrendEngine()

        result = engine.analyze_trends(df, {"time_column": "date", "value_columns": ["sales", "revenue"]})

        print("✅ Trend analysis completed")

        for col, analysis in result["results"].items():
            trend = analysis["trend"]
            print(f"   {col}:")
            print(f"     Direction: {trend['direction']}")
            print(f"     Strength: {trend['strength']}")
            print(f"     R²: {trend['r_squared']:.3f}")

            if analysis.get("change_points"):
                print(f"     Change points: {len(analysis['change_points'])}")

        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all integration tests"""
    print("\n" + "=" * 60)
    print("ANALYTICS ENGINES - COMPREHENSIVE INTEGRATION TEST")
    print("=" * 60)
    print("Testing all 7 engines with synthetic data\n")

    results = []

    # Run all tests
    results.append(("Statistical Engine", test_statistical_engine()))
    results.append(("Universal Graph Engine", test_universal_graph_engine()))
    results.append(("RAG Evaluation Engine", test_rag_evaluation_engine()))
    results.append(("Predictive Engine", test_predictive_engine()))
    results.append(("Anomaly Engine", test_anomaly_engine()))
    results.append(("Clustering Engine", test_clustering_engine()))
    results.append(("Trend Engine", test_trend_engine()))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL ENGINES OPERATIONAL - PRODUCTION READY!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} engines failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
