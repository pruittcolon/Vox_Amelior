"""
Financial Analytics Router

Money analytics endpoints:
- /cost-optimization - Cost optimization analysis
- /roi-prediction - ROI prediction
- /spend-patterns - Spend pattern analysis
- /budget-variance - Budget variance analysis
- /profit-margins - Profit margin analysis
- /revenue-forecast - Revenue forecasting
- /customer-ltv - Customer lifetime value
- /inventory-optimization - Inventory optimization
- /pricing-strategy - Pricing strategy analysis
- /market-basket - Market basket analysis
- /resource-utilization - Resource utilization
- /cash-flow - Cash flow analysis
- /financial-dashboard - Full financial dashboard
- /prediction-chart - Premium prediction visualizations
"""

import logging
import os
import traceback
from typing import Any

from fastapi import APIRouter, HTTPException

# Import with fallback for different import contexts
try:
    from ..schemas.analytics_models import (
        FinancialDashboardRequest,
        PredictionVisualizationRequest,
        StatisticalAnalysisRequest,
    )
    from ..utils.analytics_utils import (
        convert_to_native,
        load_dataset,
        secure_file_path,
    )
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
        FinancialDashboardRequest,
        PredictionVisualizationRequest,
        StatisticalAnalysisRequest,
    )
    from utils.analytics_utils import (
        convert_to_native,
        load_dataset,
        secure_file_path,
    )
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

router = APIRouter(tags=["financial_analytics"])

# Financial engines list
FINANCIAL_ENGINES_LIST = [
    "cost_optimization",
    "roi_prediction",
    "spend_patterns",
    "budget_variance",
    "profit_margins",
    "revenue_forecasting",
    "customer_ltv",
    "cash_flow",
    "inventory_optimization",
    "pricing_strategy",
    "market_basket",
    "resource_utilization",
]


@router.post("/cost-optimization")
async def cost_optimization(request: StatisticalAnalysisRequest):
    """Run cost optimization analysis."""
    try:
        file_path = secure_file_path(request.filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        df = load_dataset(file_path)

        config = {}
        if request.cost_column:
            config["cost_column"] = request.cost_column
        if request.category_column:
            config["category_column"] = request.category_column

        engine = CostOptimizationEngine()
        results = engine.analyze(df, config=config if config else None)

        if "error" in results:
            raise HTTPException(status_code=400, detail=results["error"])

        return convert_to_native({
            "status": "success",
            "filename": request.filename,
            "results": results,
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Cost optimization failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/roi-prediction")
async def roi_prediction(request: StatisticalAnalysisRequest):
    """Run ROI prediction analysis."""
    try:
        file_path = secure_file_path(request.filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        df = load_dataset(file_path)

        config = {}
        if request.investment_column:
            config["investment_column"] = request.investment_column
        if request.return_column:
            config["return_column"] = request.return_column

        engine = ROIPredictionEngine()
        results = engine.analyze(df, config=config if config else None)

        if "error" in results:
            raise HTTPException(status_code=400, detail=results["error"])

        return convert_to_native({
            "status": "success",
            "filename": request.filename,
            "results": results,
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"ROI prediction failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/spend-patterns")
async def spend_patterns(request: StatisticalAnalysisRequest):
    """Run spend pattern analysis."""
    try:
        file_path = secure_file_path(request.filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        df = load_dataset(file_path)

        config = {}
        if request.spend_column:
            config["spend_column"] = request.spend_column
        if request.date_column:
            config["date_column"] = request.date_column
        if request.category_column:
            config["category_column"] = request.category_column
        if request.columns and "numerical" in request.columns:
            numerical = request.columns["numerical"]
            if numerical and not config.get("spend_column"):
                config["spend_column"] = numerical[0]

        engine = SpendPatternEngine()
        results = engine.analyze(df, config=config if config else None)

        if "error" in results and not results.get("fallback_used"):
            raise HTTPException(status_code=400, detail=results["error"])

        return convert_to_native({
            "status": "success" if not results.get("fallback_used") else "fallback",
            "filename": request.filename,
            "results": results,
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Spend pattern analysis failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/budget-variance")
async def budget_variance(request: StatisticalAnalysisRequest):
    """Run budget variance analysis."""
    try:
        file_path = secure_file_path(request.filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        df = load_dataset(file_path)

        config = {}
        if request.budget_column:
            config["budget_column"] = request.budget_column
        if request.actual_column:
            config["actual_column"] = request.actual_column
        if request.columns and "numerical" in request.columns:
            numerical = request.columns["numerical"]
            if numerical and len(numerical) >= 2:
                if not config.get("budget_column"):
                    config["budget_column"] = numerical[0]
                if not config.get("actual_column"):
                    config["actual_column"] = numerical[1]

        engine = BudgetVarianceEngine()
        results = engine.analyze(df, config=config if config else None)

        if "error" in results and not results.get("fallback_used"):
            raise HTTPException(status_code=400, detail=results["error"])

        return convert_to_native({
            "status": "success" if not results.get("fallback_used") else "fallback",
            "filename": request.filename,
            "results": results,
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Budget variance analysis failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/profit-margins")
async def profit_margins(request: StatisticalAnalysisRequest):
    """Run profit margin analysis."""
    try:
        file_path = secure_file_path(request.filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        df = load_dataset(file_path)

        config = {}
        if request.revenue_column:
            config["revenue_column"] = request.revenue_column
        if request.cost_column:
            config["cost_column"] = request.cost_column

        engine = ProfitMarginEngine()
        results = engine.analyze(df, config=config if config else None)

        if "error" in results and not results.get("fallback_used"):
            raise HTTPException(status_code=400, detail=results["error"])

        return convert_to_native({
            "status": "success" if not results.get("fallback_used") else "fallback",
            "filename": request.filename,
            "results": results,
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Profit margin analysis failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/revenue-forecast")
async def revenue_forecast(request: StatisticalAnalysisRequest):
    """Run revenue forecasting analysis."""
    try:
        file_path = secure_file_path(request.filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        df = load_dataset(file_path)

        config = {}
        if request.revenue_column:
            config["revenue_column"] = request.revenue_column
        if request.date_column:
            config["date_column"] = request.date_column

        engine = RevenueForecastingEngine()
        results = engine.analyze(df, config=config if config else None)

        if "error" in results and not results.get("fallback_used"):
            raise HTTPException(status_code=400, detail=results["error"])

        return convert_to_native({
            "status": "success",
            "filename": request.filename,
            "results": results,
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Revenue forecasting failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/customer-ltv")
async def customer_ltv(request: StatisticalAnalysisRequest):
    """Run customer lifetime value analysis."""
    try:
        file_path = secure_file_path(request.filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        df = load_dataset(file_path)

        config = {}
        if request.customer_column:
            config["customer_column"] = request.customer_column
        if request.amount_column:
            config["amount_column"] = request.amount_column
        if request.date_column:
            config["date_column"] = request.date_column

        engine = CustomerLTVEngine()
        results = engine.analyze(df, config=config if config else None)

        if "error" in results and not results.get("fallback_used"):
            raise HTTPException(status_code=400, detail=results["error"])

        return convert_to_native({
            "status": "success",
            "filename": request.filename,
            "results": results,
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Customer LTV analysis failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/inventory-optimization")
async def inventory_optimization(request: StatisticalAnalysisRequest):
    """Run inventory optimization analysis."""
    try:
        file_path = secure_file_path(request.filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        df = load_dataset(file_path)

        config = {}
        if request.quantity_column:
            config["quantity_column"] = request.quantity_column
        if request.product_column:
            config["product_column"] = request.product_column

        engine = InventoryOptimizationEngine()
        results = engine.analyze(df, config=config if config else None)

        if "error" in results and not results.get("fallback_used"):
            raise HTTPException(status_code=400, detail=results["error"])

        return convert_to_native({
            "status": "success",
            "filename": request.filename,
            "results": results,
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Inventory optimization failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pricing-strategy")
async def pricing_strategy(request: StatisticalAnalysisRequest):
    """Run pricing strategy analysis."""
    try:
        file_path = secure_file_path(request.filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        df = load_dataset(file_path)

        config = {}
        if request.price_column:
            config["price_column"] = request.price_column
        if request.quantity_column:
            config["demand_column"] = request.quantity_column

        engine = PricingStrategyEngine()
        results = engine.analyze(df, config=config if config else None)

        if "error" in results and not results.get("fallback_used"):
            raise HTTPException(status_code=400, detail=results["error"])

        return convert_to_native({
            "status": "success",
            "filename": request.filename,
            "results": results,
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Pricing strategy failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/market-basket")
async def market_basket(request: StatisticalAnalysisRequest):
    """Run market basket analysis."""
    try:
        file_path = secure_file_path(request.filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        df = load_dataset(file_path)

        config = {}
        if request.product_column:
            config["product_column"] = request.product_column
        if request.customer_column:
            config["transaction_column"] = request.customer_column

        engine = MarketBasketAnalysisEngine()
        results = engine.analyze(df, config=config if config else None)

        if "error" in results and not results.get("fallback_used"):
            raise HTTPException(status_code=400, detail=results["error"])

        return convert_to_native({
            "status": "success",
            "filename": request.filename,
            "results": results,
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Market basket analysis failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/resource-utilization")
async def resource_utilization(request: StatisticalAnalysisRequest):
    """Run resource utilization analysis."""
    try:
        file_path = secure_file_path(request.filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        df = load_dataset(file_path)

        config = {}
        if request.resource_column:
            config["resource_column"] = request.resource_column
        if request.utilization_column:
            config["utilization_column"] = request.utilization_column

        engine = ResourceUtilizationEngine()
        results = engine.analyze(df, config=config if config else None)

        if "error" in results and not results.get("fallback_used"):
            raise HTTPException(status_code=400, detail=results["error"])

        return convert_to_native({
            "status": "success",
            "filename": request.filename,
            "results": results,
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Resource utilization failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cash-flow")
async def cash_flow(request: StatisticalAnalysisRequest):
    """Run cash flow analysis."""
    try:
        file_path = secure_file_path(request.filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        df = load_dataset(file_path)

        config = {}
        if request.amount_column:
            config["amount_column"] = request.amount_column
        if request.date_column:
            config["date_column"] = request.date_column

        engine = CashFlowEngine()
        results = engine.analyze(df, config=config if config else None)

        if "error" in results and not results.get("fallback_used"):
            raise HTTPException(status_code=400, detail=results["error"])

        return convert_to_native({
            "status": "success",
            "filename": request.filename,
            "results": results,
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Cash flow analysis failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/financial-engines")
async def list_financial_engines():
    """List all available financial analytics engines."""
    return {
        "status": "success",
        "engines": FINANCIAL_ENGINES_LIST,
        "total": len(FINANCIAL_ENGINES_LIST),
    }
