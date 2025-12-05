"""
Advanced Analytics Engines for ML Service

This package contains specialized analytics engines for comprehensive
business intelligence and machine learning analysis.
"""

from .statistical_engine import StatisticalEngine
from .universal_graph_engine import UniversalGraphEngine
from .rag_evaluation_engine import RAGEvaluationEngine
from .predictive_engine import PredictiveEngine
from .anomaly_engine import AnomalyEngine
from .clustering_engine import ClusteringEngine
from .trend_engine import TrendEngine

# Engine registry for modular execution
from .engine_registry import (
    ENGINE_REGISTRY,
    EngineInfo,
    EngineCategory,
    EngineResult,
    BatchResult,
    run_engines_in_order,
    run_engines_in_order_async,
    run_single_engine,
    get_available_engines,
    get_engine_info,
    list_engine_names
)

# Re-export legacy engines from engines.py for backward compatibility
import sys
import os
# Import from the parent directory's engines.py file
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

try:
    # Import from engines.py (the standalone file, not this package)
    import importlib.util
    _engines_file = os.path.join(_parent_dir, 'engines.py')
    if os.path.exists(_engines_file):
        _spec = importlib.util.spec_from_file_location("engines_legacy", _engines_file)
        _engines_module = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_engines_module)
        
        SemanticMapper = _engines_module.SemanticMapper
        AnalyticEngine = _engines_module.AnalyticEngine
        VectorEngine = _engines_module.VectorEngine
        RecipeEngine = _engines_module.RecipeEngine
        TimeSeriesEngine = _engines_module.TimeSeriesEngine
        QueryEngine = _engines_module.QueryEngine
    else:
        # Fallback: create stubs if file doesn't exist
        SemanticMapper = None
        AnalyticEngine = None
        VectorEngine = None
        RecipeEngine = None
        TimeSeriesEngine = None
        QueryEngine = None
except Exception as e:
    print(f"Warning: Could not load legacy engines: {e}")
    SemanticMapper = None
    AnalyticEngine = None
    VectorEngine = None
    RecipeEngine = None
    TimeSeriesEngine = None
    QueryEngine = None

__all__ = [
    'StatisticalEngine',
    'UniversalGraphEngine',
    'RAGEvaluationEngine',
    'PredictiveEngine',
    'AnomalyEngine',
    'ClusteringEngine',
    'TrendEngine',
    # Engine registry
    'ENGINE_REGISTRY',
    'EngineInfo',
    'EngineCategory',
    'EngineResult',
    'BatchResult',
    'run_engines_in_order',
    'run_engines_in_order_async',
    'run_single_engine',
    'get_available_engines',
    'get_engine_info',
    'list_engine_names',
    # Legacy engines
    'SemanticMapper',
    'AnalyticEngine',
    'VectorEngine',
    'RecipeEngine',
    'TimeSeriesEngine',
    'QueryEngine',
]
