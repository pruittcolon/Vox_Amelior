"""
Analytics Pydantic Request/Response Models

Centralized schema definitions for all analytics endpoints.
Extracted from analytics_routes.py for modular architecture.
"""

from typing import Any

from pydantic import BaseModel


# =============================================================================
# CORE ANALYTICS MODELS
# =============================================================================


class ColumnClassificationRequest(BaseModel):
    """Request for column classification using the statistical classifier."""

    filename: str



class StatisticalAnalysisRequest(BaseModel):
    """Request for comprehensive statistical analysis."""

    filename: str
    analysis_types: list[str] = ["all"]
    target_columns: list[str] | None = None
    group_by: str | None = None
    target: str | None = None
    features: list[str] | None = None
    confidence_level: float = 0.95
    # Column mapping hints for schema-agnostic engines
    columns: dict[str, Any] | None = None
    spend_column: str | None = None
    date_column: str | None = None
    revenue_column: str | None = None
    budget_column: str | None = None
    actual_column: str | None = None
    customer_column: str | None = None
    amount_column: str | None = None
    category_column: str | None = None
    # Additional column hints for more engines
    cost_column: str | None = None
    price_column: str | None = None
    quantity_column: str | None = None
    product_column: str | None = None
    investment_column: str | None = None
    return_column: str | None = None
    resource_column: str | None = None
    utilization_column: str | None = None


class UniversalGraphRequest(BaseModel):
    """Request for auto-generating visualizations."""

    filename: str
    max_graphs: int | None = None
    focus_columns: list[str] | None = None


class RAGEvaluationRequest(BaseModel):
    """Request for RAG system evaluation."""

    test_cases: list[dict[str, Any]]
    rag_responses: list[dict[str, Any]]
    k_values: list[int] = [1, 3, 5, 10]
    use_llm_judge: bool = False


class PredictRequest(BaseModel):
    """Request for time-series forecasting."""

    filename: str
    time_column: str | None = None
    target_column: str | None = None
    horizon: int = 30
    models: list[str] = ["auto"]
    confidence_level: float = 0.95


class AnomalyRequest(BaseModel):
    """Request for anomaly detection."""

    filename: str
    target_columns: list[str] | None = None
    methods: list[str] = ["ensemble"]
    contamination: float = 0.05
    threshold: float = 3.0


class ClusterRequest(BaseModel):
    """Request for clustering analysis."""

    filename: str
    features: list[str] | None = None
    algorithm: str = "auto"
    n_clusters: int = 3
    auto_k_range: list[int] = [2, 10]


class TrendRequest(BaseModel):
    """Request for trend analysis."""

    filename: str
    time_column: str | None = None
    value_columns: list[str] | None = None


class QuickAnalysisRequest(BaseModel):
    """Request for quick analysis with layman summary."""

    filename: str
    target_column: str | None = None


class StandardEngineRequest(BaseModel):
    """Request for running any standard engine."""

    filename: str
    target_column: str | None = None
    config_overrides: dict[str, Any] | None = None


# =============================================================================
# PREMIUM/TITAN MODELS
# =============================================================================


class TitanPremiumRequest(BaseModel):
    """Request for Titan Premium analysis with Gemma ranking."""

    filename: str
    target_column: str | None = None
    n_variants: int = 10
    page: int = 1
    page_size: int = 1
    holdout_ratio: float = 0.0
    enable_gemma_ranking: bool = False
    config_overrides: dict[str, Any] | None = None


class TitanPremiumNextRequest(BaseModel):
    """Request for next variant in pagination."""

    session_id: str


class TitanPremiumConfigRequest(BaseModel):
    """Request to re-run analysis with custom config."""

    session_id: str
    config_overrides: dict[str, Any]


class PremiumAnalysisRequest(BaseModel):
    """Request model for unified premium analysis."""

    filename: str
    target_column: str | None = None
    config_overrides: dict[str, Any] | None = None


# =============================================================================
# FINANCIAL ANALYTICS MODELS
# =============================================================================


class FinancialDashboardRequest(BaseModel):
    """Request model for the premium financial dashboard.

    Runs multiple financial engines and formats results for
    enterprise-grade visualizations.
    """

    filename: str
    engines: list[str] = [
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
    currency: str = "USD"
    include_recommendations: bool = True


class PredictionVisualizationRequest(BaseModel):
    """Request for generating premium prediction visualizations."""

    filename: str
    target_column: str
    time_column: str | None = None
    prediction_horizon: int = 15
    confidence_level: float = 0.95
    include_variance_bands: bool = True


# =============================================================================
# SMART ANALYZE MODELS
# =============================================================================


class SmartAnalyzeRequest(BaseModel):
    """Request for smart-analyze with Gemma-powered recommendations."""

    filename: str
    target_column: str | None = None
    priority: str = "insight"  # insight, speed, accuracy
    include_gemma_summary: bool = True


class SmartAnalyzeNextRequest(BaseModel):
    """Request for next variant in smart-analyze session."""

    session_id: str
    engine_name: str
    page_size: int = 1
