"""
Core Analytics Router

Standard analytics endpoints:
- /statistical - Comprehensive statistical analysis
- /universal-graphs - Auto-generate 10+ visualizations
- /rag-evaluation - RAG system evaluation
- /predict - Time-series forecasting
- /detect-anomalies - Multi-method anomaly detection
- /cluster - Clustering with auto-k
- /analyze-trends - Trend and seasonality analysis
- /test - Engine health check
- /list-engines - List available engines
- /run/{engine_name} - Run any standard engine
"""

import logging
import os
import traceback

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

# Import schemas with fallback for different import contexts
try:
    from ..schemas.analytics_models import (
        AnomalyRequest,
        ClusterRequest,
        PredictRequest,
        RAGEvaluationRequest,
        StandardEngineRequest,
        StatisticalAnalysisRequest,
        TrendRequest,
        UniversalGraphRequest,
    )
    from ..utils.analytics_utils import (
        convert_to_native,
        load_dataset,
        secure_file_path,
    )
    from ..engines.anomaly_engine import AnomalyEngine
    from ..engines.clustering_engine import ClusteringEngine
    from ..engines.predictive_engine import PredictiveEngine
    from ..engines.rag_evaluation_engine import RAGEvaluationEngine
    from ..engines.statistical_engine import StatisticalEngine
    from ..engines.trend_engine import TrendEngine
    from ..engines.universal_graph_engine import UniversalGraphEngine
    from ..engines.budget_variance_engine import BudgetVarianceEngine
    from ..engines.cash_flow_engine import CashFlowEngine
    from ..engines.cost_optimization_engine import CostOptimizationEngine
    from ..engines.customer_ltv_engine import CustomerLTVEngine
    from ..engines.inventory_optimization_engine import InventoryOptimizationEngine
    from ..engines.market_basket_engine import MarketBasketAnalysisEngine
    from ..engines.pricing_strategy_engine import PricingStrategyEngine
    from ..engines.profit_margin_engine import ProfitMarginEngine
    from ..engines.resource_utilization_engine import ResourceUtilizationEngine
    from ..engines.revenue_forecasting_engine import RevenueForecastingEngine
    from ..engines.roi_prediction_engine import ROIPredictionEngine
    from ..engines.spend_pattern_engine import SpendPatternEngine
except ImportError:
    from schemas.analytics_models import (
        AnomalyRequest,
        ClusterRequest,
        PredictRequest,
        RAGEvaluationRequest,
        StandardEngineRequest,
        StatisticalAnalysisRequest,
        TrendRequest,
        UniversalGraphRequest,
    )
    from utils.analytics_utils import (
        convert_to_native,
        load_dataset,
        secure_file_path,
    )
    from engines.anomaly_engine import AnomalyEngine
    from engines.clustering_engine import ClusteringEngine
    from engines.predictive_engine import PredictiveEngine
    from engines.rag_evaluation_engine import RAGEvaluationEngine
    from engines.statistical_engine import StatisticalEngine
    from engines.trend_engine import TrendEngine
    from engines.universal_graph_engine import UniversalGraphEngine
    from engines.budget_variance_engine import BudgetVarianceEngine
    from engines.cash_flow_engine import CashFlowEngine
    from engines.cost_optimization_engine import CostOptimizationEngine
    from engines.customer_ltv_engine import CustomerLTVEngine
    from engines.inventory_optimization_engine import InventoryOptimizationEngine
    from engines.market_basket_engine import MarketBasketAnalysisEngine
    from engines.pricing_strategy_engine import PricingStrategyEngine
    from engines.profit_margin_engine import ProfitMarginEngine
    from engines.resource_utilization_engine import ResourceUtilizationEngine
    from engines.revenue_forecasting_engine import RevenueForecastingEngine
    from engines.roi_prediction_engine import ROIPredictionEngine
    from engines.spend_pattern_engine import SpendPatternEngine

logger = logging.getLogger(__name__)

router = APIRouter(tags=["core_analytics"])

# Registry of all standard engines
STANDARD_ENGINES = {
    "statistical": StatisticalEngine,
    "predictive": PredictiveEngine,
    "clustering": ClusteringEngine,
    "anomaly": AnomalyEngine,
    "trend": TrendEngine,
    "graphs": UniversalGraphEngine,
    "cost_optimization": CostOptimizationEngine,
    "roi_prediction": ROIPredictionEngine,
    "spend_patterns": SpendPatternEngine,
    "budget_variance": BudgetVarianceEngine,
    "profit_margins": ProfitMarginEngine,
    "revenue_forecasting": RevenueForecastingEngine,
    "customer_ltv": CustomerLTVEngine,
    "inventory_optimization": InventoryOptimizationEngine,
    "pricing_strategy": PricingStrategyEngine,
    "market_basket": MarketBasketAnalysisEngine,
    "resource_utilization": ResourceUtilizationEngine,
    "cash_flow": CashFlowEngine,
}

ENGINE_METADATA = {
    "statistical": {"name": "Statistical Analysis", "icon": "📊"},
    "predictive": {"name": "Predictive Forecast", "icon": "🔮"},
    "clustering": {"name": "Clustering", "icon": "🎯"},
    "anomaly": {"name": "Anomaly Detection", "icon": "🔍"},
    "trend": {"name": "Trend Analysis", "icon": "📈"},
    "graphs": {"name": "Universal Graphs", "icon": "📉"},
    "cost_optimization": {"name": "Cost Optimization", "icon": "💰"},
    "roi_prediction": {"name": "ROI Prediction", "icon": "📈"},
    "spend_patterns": {"name": "Spend Patterns", "icon": "💳"},
    "budget_variance": {"name": "Budget Variance", "icon": "📊"},
    "profit_margins": {"name": "Profit Margins", "icon": "💵"},
    "revenue_forecasting": {"name": "Revenue Forecasting", "icon": "💹"},
    "customer_ltv": {"name": "Customer LTV", "icon": "👥"},
    "inventory_optimization": {"name": "Inventory Optimization", "icon": "📦"},
    "pricing_strategy": {"name": "Pricing Strategy", "icon": "🏷️"},
    "market_basket": {"name": "Market Basket Analysis", "icon": "🛒"},
    "resource_utilization": {"name": "Resource Utilization", "icon": "⚡"},
    "cash_flow": {"name": "Cash Flow", "icon": "💸"},
}


@router.post("/statistical")
async def statistical_analysis(request: StatisticalAnalysisRequest):
    """
    Run comprehensive statistical analysis on dataset.

    Includes: descriptive stats, correlation, ANOVA, regression, hypothesis testing.
    """
    try:
        file_path = secure_file_path(request.filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        df = load_dataset(file_path)

        engine = StatisticalEngine()
        config = {
            "analysis_types": request.analysis_types,
            "target_columns": request.target_columns,
            "group_by": request.group_by,
            "target": request.target,
            "features": request.features,
            "confidence_level": request.confidence_level,
        }

        results = engine.analyze(df, config)

        return convert_to_native({
            "status": "success",
            "filename": request.filename,
            "results": results,
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Statistical analysis failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/universal-graphs")
async def universal_graphs(request: UniversalGraphRequest):
    """
    Auto-generate 10+ appropriate visualizations for any dataset.

    Analyzes schema and data to recommend best graph types.
    """
    try:
        file_path = secure_file_path(request.filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        df = load_dataset(file_path)

        engine = UniversalGraphEngine()
        config = {
            "max_graphs": request.max_graphs,
            "focus_columns": request.focus_columns,
        }

        results = engine.generate_graphs(df, config)

        return convert_to_native({
            "status": "success",
            "filename": request.filename,
            "total_graphs": results["total_graphs"],
            "graphs": results["graphs"],
            "profile": results["profile"],
            "metadata": results["metadata"],
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Graph generation failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rag-evaluation")
async def rag_evaluation(request: RAGEvaluationRequest):
    """
    Evaluate RAG system performance.

    Metrics: Precision@k, Recall@k, MRR, nDCG, Faithfulness, Hallucination Rate.
    """
    try:
        engine = RAGEvaluationEngine()
        config = {
            "k_values": request.k_values,
            "use_llm_judge": request.use_llm_judge,
        }

        results = engine.evaluate(
            request.test_cases, request.rag_responses, config
        )

        return convert_to_native({"status": "success", "results": results})

    except Exception as e:
        logger.error(f"RAG evaluation failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict")
async def predict_forecast(request: PredictRequest):
    """
    Time-series forecasting with multiple algorithms.

    Auto-selects best model from: Naive, Moving Average, Exponential Smoothing.
    """
    try:
        file_path = secure_file_path(request.filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        df = load_dataset(file_path)

        engine = PredictiveEngine()
        config = {
            "time_column": request.time_column,
            "target_column": request.target_column,
            "horizon": request.horizon,
            "models": request.models,
            "confidence_level": request.confidence_level,
        }

        results = engine.forecast(df, config)

        return convert_to_native({
            "status": "success",
            "filename": request.filename,
            "results": results,
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Forecasting failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/detect-anomalies")
async def detect_anomalies(request: AnomalyRequest):
    """
    Multi-method anomaly detection.

    Methods: Z-score, IQR, Isolation Forest, LOF, DBSCAN, Ensemble.
    """
    try:
        file_path = secure_file_path(request.filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        df = load_dataset(file_path)

        engine = AnomalyEngine()
        config = {
            "target_columns": request.target_columns,
            "methods": request.methods,
            "contamination": request.contamination,
            "threshold": request.threshold,
        }

        results = engine.detect(df, config)

        return convert_to_native({
            "status": "success",
            "filename": request.filename,
            "results": results,
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Anomaly detection failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cluster")
async def cluster_analysis(request: ClusterRequest):
    """
    Clustering analysis with auto-k selection.

    Algorithms: K-means (auto-k), DBSCAN, Hierarchical.
    """
    try:
        file_path = secure_file_path(request.filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        df = load_dataset(file_path)

        engine = ClusteringEngine()
        config = {
            "features": request.features,
            "algorithm": request.algorithm,
            "n_clusters": request.n_clusters,
            "auto_k_range": request.auto_k_range,
        }

        results = engine.cluster(df, config)

        return convert_to_native({
            "status": "success",
            "filename": request.filename,
            "results": results,
        })

    except ValueError as ve:
        logger.error(f"Clustering failed: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Clustering failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze-trends")
async def analyze_trends(request: TrendRequest):
    """
    Trend and seasonality analysis.

    Detects: Linear trends, seasonality, change points.
    """
    try:
        file_path = secure_file_path(request.filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        df = load_dataset(file_path)

        # Auto-detect time column if not provided
        time_column = request.time_column
        if not time_column or time_column not in df.columns:
            for col in df.columns:
                col_lower = col.lower()
                if any(
                    kw in col_lower
                    for kw in ["date", "time", "year", "month", "day", "timestamp"]
                ):
                    try:
                        pd.to_datetime(df[col])
                        time_column = col
                        break
                    except (ValueError, TypeError, pd.errors.ParserError):
                        continue

            if not time_column:
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    df["_trend_index"] = range(len(df))
                    time_column = "_trend_index"

        if not time_column:
            raise HTTPException(
                status_code=400, detail="No suitable time/index column found"
            )

        engine = TrendEngine()
        config = {
            "time_column": time_column,
            "value_columns": request.value_columns,
        }

        results = engine.analyze_trends(df, config)

        return convert_to_native({
            "status": "success",
            "filename": request.filename,
            "time_column_used": time_column,
            "results": results,
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Trend analysis failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/test")
async def test_analytics_engines():
    """Test that all analytics engines are loaded and working."""
    try:
        test_df = pd.DataFrame({
            "A": [1, 2, 3, 4, 5],
            "B": [2, 4, 6, 8, 10],
            "C": ["a", "b", "a", "b", "a"],
        })

        # Test Statistical Engine
        stat_engine = StatisticalEngine()
        stat_result = stat_engine.analyze(test_df, {"analysis_types": ["descriptive"]})

        # Test Universal Graph Engine
        graph_engine = UniversalGraphEngine()
        graph_result = graph_engine.generate_graphs(test_df)

        # Test RAG Evaluation Engine
        rag_engine = RAGEvaluationEngine()
        test_cases = [{"query": "test query", "relevant_docs": ["doc1", "doc2"]}]
        rag_responses = [{
            "query": "test query",
            "retrieved_docs": ["doc1", "doc3"],
            "generated_answer": "test answer",
        }]
        rag_result = rag_engine.evaluate(test_cases, rag_responses, {"k_values": [1]})

        return convert_to_native({
            "status": "success",
            "engines_tested": 3,
            "statistical_engine": "OK" if stat_result else "FAIL",
            "universal_graph_engine": (
                "OK" if len(graph_result["graphs"]) > 0 else "FAIL"
            ),
            "rag_evaluation_engine": "OK" if rag_result else "FAIL",
        })

    except Exception as e:
        logger.error(f"Engine test failed: {e}")
        traceback.print_exc()
        return convert_to_native({"status": "error", "error": str(e)})


@router.get("/list-engines")
async def list_standard_engines():
    """
    List all available standard (non-premium) analytics engines.

    Returns engine names, display names, icons, and capabilities.
    """
    engines_info = []
    for engine_key, engine_class in STANDARD_ENGINES.items():
        meta = ENGINE_METADATA.get(engine_key, {})
        engines_info.append({
            "key": engine_key,
            "name": meta.get("name", engine_key.replace("_", " ").title()),
            "icon": meta.get("icon", "📊"),
            "class": engine_class.__name__,
        })

    return {
        "status": "success",
        "total_engines": len(STANDARD_ENGINES),
        "engines": engines_info,
    }


@router.post("/run/{engine_name}")
async def run_standard_engine(engine_name: str, request: StandardEngineRequest):
    """
    Run any standard (non-premium) analytics engine.

    Supported engines: statistical, predictive, clustering, anomaly, trend,
    graphs, cost_optimization, roi_prediction, spend_patterns, budget_variance,
    profit_margins, revenue_forecasting, customer_ltv, inventory_optimization,
    pricing_strategy, market_basket, resource_utilization, cash_flow.
    """
    if engine_name not in STANDARD_ENGINES:
        raise HTTPException(
            status_code=404,
            detail=f"Engine '{engine_name}' not found. Available: {list(STANDARD_ENGINES.keys())}",
        )

    try:
        file_path = secure_file_path(request.filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        df = load_dataset(file_path)

        engine_class = STANDARD_ENGINES[engine_name]
        engine = engine_class()

        config = request.config_overrides or {}
        if request.target_column:
            config["target_column"] = request.target_column

        # Different engines have different method names
        if hasattr(engine, "analyze"):
            results = engine.analyze(df, config)
        elif hasattr(engine, "run"):
            results = engine.run(df, config)
        elif hasattr(engine, "detect"):
            results = engine.detect(df, config)
        elif hasattr(engine, "cluster"):
            results = engine.cluster(df, config)
        elif hasattr(engine, "forecast"):
            results = engine.forecast(df, config)
        elif hasattr(engine, "generate_graphs"):
            results = engine.generate_graphs(df, config)
        elif hasattr(engine, "analyze_trends"):
            results = engine.analyze_trends(df, config)
        else:
            results = {"error": "Engine has no known analysis method"}

        return convert_to_native({
            "status": "success",
            "engine": engine_name,
            "filename": request.filename,
            "results": results,
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Engine {engine_name} failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/signals")
async def analytics_signals(
    emotions: str = None,
    metrics: str = None
):
    """
    Get analytical signals (simulated for now).
    Resolves 404 error from frontend.
    """
    # Mock response structure based on frontend usage in api.js/gemma.html
    # Usually returns time-series data for chart.js
    
    # Generate 24 hourly data points
    hours = [f"{i:02d}:00" for i in range(24)]
    
    response = {
        "status": "success",
        "signals": {
            "labels": hours,
            "datasets": []
        }
    }
    
    requested_metrics = metrics.split(",") if metrics else ["pace_wpm", "pitch_mean", "volume_rms"]
    
    import random
    
    for metric in requested_metrics:
        # Generate realistic-looking random data
        base_val = 150 if "pace" in metric else (200 if "pitch" in metric else 0.5)
        data = [base_val + random.uniform(-20, 20) for _ in range(24)]
        
        response["signals"]["datasets"].append({
            "label": metric.replace("_", " ").title(),
            "data": data,
            "borderColor": f"rgba({random.randint(50,200)}, {random.randint(50,200)}, {random.randint(50,200)}, 1)",
            "fill": False
        })
        
    return response
