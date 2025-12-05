"""
Engine Registry - Modular Engine Execution System

Provides a simple, extensible registry for running ML engines in order.
Supports:
- Sequential execution (default) for debugging one at a time
- Parallel execution (--parallel flag) for production speed
- Skip flags (--no_titan, --no_regression, etc.) for selective testing

Usage:
    from engines.engine_registry import ENGINE_REGISTRY, run_engines_in_order
    
    # Run all engines on a dataset
    results = run_engines_in_order(df, target_column="price")
    
    # Skip specific engines
    results = run_engines_in_order(df, skip_engines=["titan", "regression"])
    
    # Run in parallel (production mode)
    results = await run_engines_in_order_async(df, parallel=True
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable, Type
from enum import Enum
import pandas as pd
import traceback

logger = logging.getLogger(__name__)


# =============================================================================
# ENGINE METADATA
# =============================================================================

class EngineCategory(str, Enum):
    """Categories for organizing engines"""
    AUTOML = "automl"           # Full AutoML pipelines (Titan)
    PREDICTIVE = "predictive"   # Forecasting, regression
    CLUSTERING = "clustering"   # Segmentation, clustering
    ANOMALY = "anomaly"         # Anomaly detection
    STATISTICAL = "statistical" # Statistical analysis
    FINANCIAL = "financial"     # Financial/business metrics
    GRAPH = "graph"             # Visualization engines
    ADVANCED = "advanced"       # Research/experimental


@dataclass
class EngineInfo:
    """Metadata for a registered engine"""
    name: str                           # Short name (e.g., "titan")
    display_name: str                   # UI display name
    description: str                    # What this engine does
    category: EngineCategory            # Engine category
    engine_class: Type                  # The actual engine class
    priority: int = 100                 # Execution order (lower = first)
    requires_target: bool = True        # Needs target column?
    min_rows: int = 10                  # Minimum rows required
    min_numeric_cols: int = 1           # Minimum numeric columns
    enabled: bool = True                # Is this engine active?
    gpu_capable: bool = False           # Can use GPU acceleration?
    tags: List[str] = field(default_factory=list)


# =============================================================================
# ENGINE REGISTRY - Add new engines here!
# =============================================================================

def _build_registry() -> Dict[str, EngineInfo]:
    """
    Build the engine registry. Add new engines here!
    
    Pattern for adding a new engine:
    1. Import your engine class
    2. Add an EngineInfo entry to the registry
    3. The engine will automatically be available with --no_<name> flag
    """
    registry = {}
    
    # Import engines lazily to avoid circular imports
    try:
        from .titan_engine import TitanEngine
        registry["titan"] = EngineInfo(
            name="titan",
            display_name="Titan AutoML",
            description="Enterprise-grade AutoML with nested CV, stability selection, and Gemma ranking",
            category=EngineCategory.AUTOML,
            engine_class=TitanEngine,
            priority=10,  # Run first - it's the flagship
            requires_target=True,
            gpu_capable=True,
            tags=["premium", "automl", "classification", "regression"]
        )
    except ImportError as e:
        logger.warning(f"Could not import TitanEngine: {e}")
    
    try:
        from .predictive_engine import PredictiveEngine
        registry["predictive"] = EngineInfo(
            name="predictive",
            display_name="Predictive Forecasting",
            description="Time-series forecasting with multiple algorithms",
            category=EngineCategory.PREDICTIVE,
            engine_class=PredictiveEngine,
            priority=20,
            requires_target=False,  # Auto-detects time series
            tags=["forecasting", "timeseries"]
        )
    except ImportError as e:
        logger.warning(f"Could not import PredictiveEngine: {e}")
    
    try:
        from .clustering_engine import ClusteringEngine
        registry["clustering"] = EngineInfo(
            name="clustering",
            display_name="Clustering Analysis",
            description="Multi-algorithm clustering (K-Means, DBSCAN, Hierarchical)",
            category=EngineCategory.CLUSTERING,
            engine_class=ClusteringEngine,
            priority=30,
            requires_target=False,
            tags=["segmentation", "unsupervised"]
        )
    except ImportError as e:
        logger.warning(f"Could not import ClusteringEngine: {e}")
    
    try:
        from .anomaly_engine import AnomalyEngine
        registry["anomaly"] = EngineInfo(
            name="anomaly",
            display_name="Anomaly Detection",
            description="Multi-method anomaly detection with ensemble voting",
            category=EngineCategory.ANOMALY,
            engine_class=AnomalyEngine,
            priority=40,
            requires_target=False,
            tags=["outliers", "fraud"]
        )
    except ImportError as e:
        logger.warning(f"Could not import AnomalyEngine: {e}")
    
    try:
        from .statistical_engine import StatisticalEngine
        registry["statistical"] = EngineInfo(
            name="statistical",
            display_name="Statistical Analysis",
            description="Comprehensive statistical tests and correlations",
            category=EngineCategory.STATISTICAL,
            engine_class=StatisticalEngine,
            priority=50,
            requires_target=False,
            tags=["statistics", "correlation", "hypothesis"]
        )
    except ImportError as e:
        logger.warning(f"Could not import StatisticalEngine: {e}")
    
    try:
        from .trend_engine import TrendEngine
        registry["trend"] = EngineInfo(
            name="trend",
            display_name="Trend Analysis",
            description="Trend detection and seasonality decomposition",
            category=EngineCategory.PREDICTIVE,
            engine_class=TrendEngine,
            priority=60,
            requires_target=False,
            tags=["trends", "seasonality"]
        )
    except ImportError as e:
        logger.warning(f"Could not import TrendEngine: {e}")
    
    try:
        from .universal_graph_engine import UniversalGraphEngine
        registry["graphs"] = EngineInfo(
            name="graphs",
            display_name="Universal Graphs",
            description="Auto-generate appropriate visualizations",
            category=EngineCategory.GRAPH,
            engine_class=UniversalGraphEngine,
            priority=70,
            requires_target=False,
            tags=["visualization", "charts"]
        )
    except ImportError as e:
        logger.warning(f"Could not import UniversalGraphEngine: {e}")
    
    # Financial engines
    try:
        from .cost_optimization_engine import CostOptimizationEngine
        registry["cost"] = EngineInfo(
            name="cost",
            display_name="Cost Optimization",
            description="Cost reduction and optimization analysis",
            category=EngineCategory.FINANCIAL,
            engine_class=CostOptimizationEngine,
            priority=100,
            requires_target=False,
            tags=["financial", "cost"]
        )
    except ImportError:
        pass
    
    try:
        from .roi_prediction_engine import ROIPredictionEngine
        registry["roi"] = EngineInfo(
            name="roi",
            display_name="ROI Prediction",
            description="Return on investment prediction",
            category=EngineCategory.FINANCIAL,
            engine_class=ROIPredictionEngine,
            priority=110,
            requires_target=False,
            tags=["financial", "roi", "standard"]
        )
    except ImportError:
        pass
    
    # Spend Pattern Analysis
    try:
        from .spend_pattern_engine import SpendPatternEngine
        registry["spend_patterns"] = EngineInfo(
            name="spend_patterns",
            display_name="Spend Pattern Analysis",
            description="Analyze spending patterns and detect anomalies",
            category=EngineCategory.FINANCIAL,
            engine_class=SpendPatternEngine,
            priority=115,
            requires_target=False,
            tags=["financial", "spending", "standard"]
        )
    except ImportError:
        pass
    
    # Budget Variance Analysis
    try:
        from .budget_variance_engine import BudgetVarianceEngine
        registry["budget_variance"] = EngineInfo(
            name="budget_variance",
            display_name="Budget Variance Analysis",
            description="Budget vs actual spending analysis",
            category=EngineCategory.FINANCIAL,
            engine_class=BudgetVarianceEngine,
            priority=120,
            requires_target=False,
            tags=["financial", "budget", "standard"]
        )
    except ImportError:
        pass
    
    # Profit Margin Analysis
    try:
        from .profit_margin_engine import ProfitMarginEngine
        registry["profit_margins"] = EngineInfo(
            name="profit_margins",
            display_name="Profit Margin Analysis",
            description="Analyze profit margins by product and segment",
            category=EngineCategory.FINANCIAL,
            engine_class=ProfitMarginEngine,
            priority=125,
            requires_target=False,
            tags=["financial", "profit", "standard"]
        )
    except ImportError:
        pass
    
    # Revenue Forecasting
    try:
        from .revenue_forecasting_engine import RevenueForecastingEngine
        registry["revenue_forecasting"] = EngineInfo(
            name="revenue_forecasting",
            display_name="Revenue Forecasting",
            description="Predict future revenue using time-series analysis",
            category=EngineCategory.FINANCIAL,
            engine_class=RevenueForecastingEngine,
            priority=130,
            requires_target=False,
            tags=["financial", "forecasting", "standard"]
        )
    except ImportError:
        pass
    
    # Customer LTV
    try:
        from .customer_ltv_engine import CustomerLTVEngine
        registry["customer_ltv"] = EngineInfo(
            name="customer_ltv",
            display_name="Customer LTV",
            description="Customer lifetime value calculation and segmentation",
            category=EngineCategory.FINANCIAL,
            engine_class=CustomerLTVEngine,
            priority=135,
            requires_target=False,
            tags=["financial", "customer", "standard"]
        )
    except ImportError:
        pass
    
    # Inventory Optimization
    try:
        from .inventory_optimization_engine import InventoryOptimizationEngine
        registry["inventory_optimization"] = EngineInfo(
            name="inventory_optimization",
            display_name="Inventory Optimization",
            description="Optimize inventory levels and reorder points",
            category=EngineCategory.FINANCIAL,
            engine_class=InventoryOptimizationEngine,
            priority=140,
            requires_target=False,
            tags=["financial", "inventory", "standard"]
        )
    except ImportError:
        pass
    
    # Pricing Strategy
    try:
        from .pricing_strategy_engine import PricingStrategyEngine
        registry["pricing_strategy"] = EngineInfo(
            name="pricing_strategy",
            display_name="Pricing Strategy",
            description="Price optimization and elasticity analysis",
            category=EngineCategory.FINANCIAL,
            engine_class=PricingStrategyEngine,
            priority=145,
            requires_target=False,
            tags=["financial", "pricing", "standard"]
        )
    except ImportError:
        pass
    
    # Market Basket Analysis
    try:
        from .market_basket_engine import MarketBasketAnalysisEngine
        registry["market_basket"] = EngineInfo(
            name="market_basket",
            display_name="Market Basket Analysis",
            description="Product association and cross-sell recommendations",
            category=EngineCategory.FINANCIAL,
            engine_class=MarketBasketAnalysisEngine,
            priority=150,
            requires_target=False,
            tags=["financial", "associations", "standard"]
        )
    except ImportError:
        pass
    
    # Resource Utilization
    try:
        from .resource_utilization_engine import ResourceUtilizationEngine
        registry["resource_utilization"] = EngineInfo(
            name="resource_utilization",
            display_name="Resource Utilization",
            description="Resource efficiency and bottleneck analysis",
            category=EngineCategory.FINANCIAL,
            engine_class=ResourceUtilizationEngine,
            priority=155,
            requires_target=False,
            tags=["financial", "resources", "standard"]
        )
    except ImportError:
        pass
    
    # Cash Flow Analysis
    try:
        from .cash_flow_engine import CashFlowEngine
        registry["cash_flow"] = EngineInfo(
            name="cash_flow",
            display_name="Cash Flow Analysis",
            description="Cash flow prediction and analysis",
            category=EngineCategory.FINANCIAL,
            engine_class=CashFlowEngine,
            priority=160,
            requires_target=False,
            tags=["financial", "cash", "standard"]
        )
    except ImportError:
        pass
    
    # RAG Evaluation Engine
    try:
        from .rag_evaluation_engine import RAGEvaluationEngine
        registry["rag_evaluation"] = EngineInfo(
            name="rag_evaluation",
            display_name="RAG Evaluation",
            description="Evaluate RAG system retrieval and generation quality",
            category=EngineCategory.ADVANCED,
            engine_class=RAGEvaluationEngine,
            priority=170,
            requires_target=False,
            tags=["evaluation", "rag", "standard"]
        )
    except ImportError:
        pass
    
    # Advanced/Research engines
    try:
        from .chaos_engine import ChaosEngine
        registry["chaos"] = EngineInfo(
            name="chaos",
            display_name="Chaos Engine",
            description="Non-linear relationship detection",
            category=EngineCategory.ADVANCED,
            engine_class=ChaosEngine,
            priority=200,
            requires_target=True,
            tags=["advanced", "nonlinear"]
        )
    except ImportError:
        pass
    
    try:
        from .oracle_engine import OracleEngine
        registry["oracle"] = EngineInfo(
            name="oracle",
            display_name="Oracle Causal",
            description="Granger causality analysis",
            category=EngineCategory.ADVANCED,
            engine_class=OracleEngine,
            priority=210,
            requires_target=False,
            tags=["advanced", "causality"]
        )
    except ImportError:
        pass
    
    # Premium Time Series Engine - Chronos
    try:
        from .chronos_engine import ChronosEngine
        registry["chronos"] = EngineInfo(
            name="chronos",
            display_name="Chronos Time Series",
            description="Advanced time-series forecasting with Prophet variants",
            category=EngineCategory.PREDICTIVE,
            engine_class=ChronosEngine,
            priority=15,  # High priority - premium engine
            requires_target=False,  # Auto-detects time columns
            gpu_capable=False,
            tags=["premium", "timeseries", "forecasting", "prophet"]
        )
    except ImportError as e:
        logger.warning(f"Could not import ChronosEngine: {e}")
    
    return registry


# Global registry instance
ENGINE_REGISTRY: Dict[str, EngineInfo] = _build_registry()


# =============================================================================
# ENGINE EXECUTION RESULTS
# =============================================================================

@dataclass
class EngineResult:
    """Result from running a single engine"""
    engine_name: str
    success: bool
    duration_seconds: float
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    skipped: bool = False
    skip_reason: Optional[str] = None


@dataclass 
class BatchResult:
    """Results from running multiple engines"""
    total_engines: int
    successful: int
    failed: int
    skipped: int
    total_duration_seconds: float
    results: Dict[str, EngineResult] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_engines": self.total_engines,
            "successful": self.successful,
            "failed": self.failed,
            "skipped": self.skipped,
            "total_duration_seconds": round(self.total_duration_seconds, 2),
            "results": {
                name: {
                    "success": r.success,
                    "duration": round(r.duration_seconds, 2),
                    "skipped": r.skipped,
                    "error": r.error,
                    "result": r.result if r.success else None
                }
                for name, r in self.results.items()
            }
        }


# =============================================================================
# ENGINE EXECUTION FUNCTIONS
# =============================================================================

def run_single_engine(
    engine_info: EngineInfo,
    df: pd.DataFrame,
    target_column: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> EngineResult:
    """
    Run a single engine on a dataset.
    
    Args:
        engine_info: EngineInfo metadata
        df: Input DataFrame
        target_column: Target column for supervised learning
        config: Engine-specific configuration
        
    Returns:
        EngineResult with success/failure and timing
    """
    start_time = time.time()
    
    # Check minimum requirements
    if len(df) < engine_info.min_rows:
        return EngineResult(
            engine_name=engine_info.name,
            success=False,
            duration_seconds=0,
            skipped=True,
            skip_reason=f"Dataset has {len(df)} rows, minimum is {engine_info.min_rows}"
        )
    
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) < engine_info.min_numeric_cols:
        return EngineResult(
            engine_name=engine_info.name,
            success=False,
            duration_seconds=0,
            skipped=True,
            skip_reason=f"Dataset has {len(numeric_cols)} numeric columns, minimum is {engine_info.min_numeric_cols}"
        )
    
    if engine_info.requires_target and not target_column:
        # Try to auto-detect target
        target_column = _auto_detect_target(df)
        if not target_column:
            return EngineResult(
                engine_name=engine_info.name,
                success=False,
                duration_seconds=0,
                skipped=True,
                skip_reason="Engine requires target column but none provided or detected"
            )
    
    try:
        # Instantiate engine
        engine = engine_info.engine_class()
        
        # Build config
        run_config = config or {}
        
        # Call analyze method (standard interface)
        if hasattr(engine, 'analyze'):
            if engine_info.requires_target:
                result = engine.analyze(df, target_column, run_config)
            else:
                result = engine.analyze(df, run_config)
        elif hasattr(engine, 'generate_graphs'):
            # Graph engine has different interface
            result = engine.generate_graphs(df, run_config)
        elif hasattr(engine, 'forecast'):
            result = engine.forecast(df, run_config)
        elif hasattr(engine, 'cluster'):
            result = engine.cluster(df, run_config)
        elif hasattr(engine, 'detect'):
            result = engine.detect(df, run_config)
        else:
            raise ValueError(f"Engine {engine_info.name} has no standard analyze method")
        
        duration = time.time() - start_time
        
        return EngineResult(
            engine_name=engine_info.name,
            success=True,
            duration_seconds=duration,
            result=result
        )
        
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Engine {engine_info.name} failed: {e}")
        logger.debug(traceback.format_exc())
        
        return EngineResult(
            engine_name=engine_info.name,
            success=False,
            duration_seconds=duration,
            error=str(e)
        )


def run_engines_in_order(
    df: pd.DataFrame,
    target_column: Optional[str] = None,
    skip_engines: Optional[List[str]] = None,
    only_engines: Optional[List[str]] = None,
    config: Optional[Dict[str, Any]] = None,
    verbose: bool = True
) -> BatchResult:
    """
    Run engines sequentially in priority order.
    
    This is the main entry point for modular engine execution.
    
    Args:
        df: Input DataFrame
        target_column: Target column for supervised learning
        skip_engines: List of engine names to skip (e.g., ["titan", "clustering"])
        only_engines: If provided, ONLY run these engines
        config: Engine-specific configs keyed by engine name
        verbose: Print progress messages
        
    Returns:
        BatchResult with all engine results
        
    Example:
        # Run all engines except titan
        results = run_engines_in_order(df, skip_engines=["titan"])
        
        # Run only titan and clustering
        results = run_engines_in_order(df, only_engines=["titan", "clustering"])
    """
    skip_engines = skip_engines or []
    config = config or {}
    
    # Get engines sorted by priority
    engines_to_run = sorted(
        ENGINE_REGISTRY.values(),
        key=lambda e: e.priority
    )
    
    # Filter by only_engines if provided
    if only_engines:
        engines_to_run = [e for e in engines_to_run if e.name in only_engines]
    
    # Filter out disabled and skipped engines
    engines_to_run = [
        e for e in engines_to_run 
        if e.enabled and e.name not in skip_engines
    ]
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Running {len(engines_to_run)} engines on dataset ({len(df)} rows)")
        print(f"{'='*60}\n")
    
    results: Dict[str, EngineResult] = {}
    total_start = time.time()
    successful = 0
    failed = 0
    skipped = 0
    
    for i, engine_info in enumerate(engines_to_run, 1):
        if verbose:
            print(f"[{i}/{len(engines_to_run)}] Running {engine_info.display_name}...", end=" ", flush=True)
        
        engine_config = config.get(engine_info.name, {})
        result = run_single_engine(engine_info, df, target_column, engine_config)
        results[engine_info.name] = result
        
        if result.skipped:
            skipped += 1
            if verbose:
                print(f"⏭️  SKIPPED ({result.skip_reason})")
        elif result.success:
            successful += 1
            if verbose:
                print(f"✅ OK ({result.duration_seconds:.2f}s)")
        else:
            failed += 1
            if verbose:
                print(f"❌ FAILED ({result.error})")
    
    total_duration = time.time() - total_start
    
    batch_result = BatchResult(
        total_engines=len(engines_to_run),
        successful=successful,
        failed=failed,
        skipped=skipped,
        total_duration_seconds=total_duration,
        results=results
    )
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Complete: {successful} passed, {failed} failed, {skipped} skipped")
        print(f"Total time: {total_duration:.2f}s")
        print(f"{'='*60}\n")
    
    return batch_result


async def run_engines_in_order_async(
    df: pd.DataFrame,
    target_column: Optional[str] = None,
    skip_engines: Optional[List[str]] = None,
    only_engines: Optional[List[str]] = None,
    config: Optional[Dict[str, Any]] = None,
    parallel: bool = False,
    max_workers: int = 4,
    verbose: bool = True
) -> BatchResult:
    """
    Async version with optional parallel execution.
    
    Args:
        parallel: If True, run engines concurrently (faster but harder to debug)
        max_workers: Maximum concurrent engines when parallel=True
        (other args same as run_engines_in_order)
        
    Returns:
        BatchResult with all engine results
    """
    if not parallel:
        # Run sequentially (wraps sync function)
        return run_engines_in_order(
            df, target_column, skip_engines, only_engines, config, verbose
        )
    
    # Parallel execution
    skip_engines = skip_engines or []
    config = config or {}
    
    engines_to_run = sorted(
        [e for e in ENGINE_REGISTRY.values() if e.enabled and e.name not in skip_engines],
        key=lambda e: e.priority
    )
    
    if only_engines:
        engines_to_run = [e for e in engines_to_run if e.name in only_engines]
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Running {len(engines_to_run)} engines in PARALLEL (max {max_workers} workers)")
        print(f"{'='*60}\n")
    
    total_start = time.time()
    
    # Use semaphore to limit concurrency
    semaphore = asyncio.Semaphore(max_workers)
    
    async def run_with_semaphore(engine_info: EngineInfo) -> EngineResult:
        async with semaphore:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            engine_config = config.get(engine_info.name, {})
            return await loop.run_in_executor(
                None,
                run_single_engine,
                engine_info, df, target_column, engine_config
            )
    
    # Run all engines concurrently
    tasks = [run_with_semaphore(e) for e in engines_to_run]
    engine_results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    results: Dict[str, EngineResult] = {}
    successful = failed = skipped = 0
    
    for engine_info, result in zip(engines_to_run, engine_results):
        if isinstance(result, Exception):
            results[engine_info.name] = EngineResult(
                engine_name=engine_info.name,
                success=False,
                duration_seconds=0,
                error=str(result)
            )
            failed += 1
        else:
            results[engine_info.name] = result
            if result.skipped:
                skipped += 1
            elif result.success:
                successful += 1
            else:
                failed += 1
    
    total_duration = time.time() - total_start
    
    batch_result = BatchResult(
        total_engines=len(engines_to_run),
        successful=successful,
        failed=failed,
        skipped=skipped,
        total_duration_seconds=total_duration,
        results=results
    )
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Complete: {successful} passed, {failed} failed, {skipped} skipped")
        print(f"Total time: {total_duration:.2f}s (parallel)")
        print(f"{'='*60}\n")
    
    return batch_result


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _auto_detect_target(df: pd.DataFrame) -> Optional[str]:
    """
    Auto-detect a likely target column for supervised learning.
    
    Heuristics:
    1. Last numeric column that's not an ID
    2. Columns named 'target', 'label', 'y', 'class', etc.
    """
    # Known target column names
    target_names = ['target', 'label', 'y', 'class', 'outcome', 'result', 'price', 'value']
    
    for name in target_names:
        matches = [c for c in df.columns if name.lower() in c.lower()]
        if matches:
            return matches[0]
    
    # Fall back to last numeric column that's not an ID
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    non_id_cols = [c for c in numeric_cols if 'id' not in c.lower()]
    
    if non_id_cols:
        return non_id_cols[-1]
    
    return None


def get_available_engines() -> List[Dict[str, Any]]:
    """Get list of all available engines with their metadata."""
    return [
        {
            "name": info.name,
            "display_name": info.display_name,
            "description": info.description,
            "category": info.category.value,
            "priority": info.priority,
            "requires_target": info.requires_target,
            "gpu_capable": info.gpu_capable,
            "enabled": info.enabled,
            "tags": info.tags
        }
        for info in sorted(ENGINE_REGISTRY.values(), key=lambda e: e.priority)
    ]


def get_engine_info(engine_name: str) -> Optional[EngineInfo]:
    """Get info for a specific engine."""
    return ENGINE_REGISTRY.get(engine_name)


def list_engine_names() -> List[str]:
    """Get list of all engine names (for CLI flags)."""
    return list(ENGINE_REGISTRY.keys())
