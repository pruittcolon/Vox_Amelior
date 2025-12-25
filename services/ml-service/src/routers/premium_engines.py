"""
Premium Engines Router

Flagship premium engine endpoints:
- /titan-premium - Titan AutoML with Gemma ranking
- /titan-premium/{session_id}/next - Pagination
- /titan-premium/{session_id}/all - Get all variants
- /titan-premium/{session_id}/explain - Explainability
- /titan-premium/rerun - Re-run with config changes
- /titan-premium-schema - Get config schema
- /premium/{engine_name} - Run any premium engine
- /premium/engines - List available premium engines
- /run-engine/{engine_name} - Dynamic engine runner
"""

import asyncio
import logging
import os
import traceback
import uuid
from datetime import datetime, timedelta
from typing import Any

from fastapi import APIRouter, HTTPException
from starlette.concurrency import run_in_threadpool

# Import with fallback for different import contexts
try:
    from ..schemas.analytics_models import (
        PremiumAnalysisRequest,
        TitanPremiumConfigRequest,
        TitanPremiumRequest,
    )
    from ..utils.analytics_utils import (
        AnalyticsGemmaClient,
        convert_to_native,
        load_dataset,
        secure_file_path,
        UPLOAD_DIR,
    )
    from ..engines.chaos_engine import ChaosEngine
    from ..engines.chronos_engine import ChronosEngine
    from ..engines.deep_feature_engine import DeepFeatureEngine
    from ..engines.flash_engine import FlashEngine
    from ..engines.galileo_engine import GalileoEngine
    from ..engines.mirror_engine import MirrorEngine
    from ..engines.newton_engine import NewtonEngine
    from ..engines.oracle_engine import OracleEngine
    from ..engines.scout_engine import ScoutEngine
    from ..engines.titan_engine import TITAN_CONFIG_SCHEMA, TitanEngine
except ImportError:
    from schemas.analytics_models import (
        PremiumAnalysisRequest,
        TitanPremiumConfigRequest,
        TitanPremiumRequest,
    )
    from utils.analytics_utils import (
        AnalyticsGemmaClient,
        convert_to_native,
        load_dataset,
        secure_file_path,
        UPLOAD_DIR,
    )
    from engines.chaos_engine import ChaosEngine
    from engines.chronos_engine import ChronosEngine
    from engines.deep_feature_engine import DeepFeatureEngine
    from engines.flash_engine import FlashEngine
    from engines.galileo_engine import GalileoEngine
    from engines.mirror_engine import MirrorEngine
    from engines.newton_engine import NewtonEngine
    from engines.oracle_engine import OracleEngine
    from engines.scout_engine import ScoutEngine
    from engines.titan_engine import TITAN_CONFIG_SCHEMA, TitanEngine

# Import history service
try:
    from ..analysis_history import historyService
except ImportError:
    try:
        from analysis_history import historyService
    except ImportError:
        historyService = None

logger = logging.getLogger(__name__)

router = APIRouter(tags=["premium_engines"])

# Premium engines registry
PREMIUM_ENGINES = {
    "titan": TitanEngine,
    "chaos": ChaosEngine,
    "scout": ScoutEngine,
    "oracle": OracleEngine,
    "newton": NewtonEngine,
    "flash": FlashEngine,
    "mirror": MirrorEngine,
    "chronos": ChronosEngine,
    "deep_feature": DeepFeatureEngine,
    "galileo": GalileoEngine,
}

# Session storage for Titan Premium
_titan_sessions: dict[str, dict[str, Any]] = {}
SESSION_TIMEOUT_MINUTES = 30


def _cleanup_expired_sessions():
    """Remove expired sessions."""
    now = datetime.now()
    expired = [
        sid
        for sid, data in _titan_sessions.items()
        if now - data.get("created_at", now) > timedelta(minutes=SESSION_TIMEOUT_MINUTES)
    ]
    for sid in expired:
        del _titan_sessions[sid]


def _create_session(
    results: dict[str, Any], filename: str, config: dict[str, Any]
) -> str:
    """Create new session and return session_id."""
    _cleanup_expired_sessions()
    session_id = str(uuid.uuid4())
    _titan_sessions[session_id] = {
        "results": results,
        "filename": filename,
        "config": config,
        "current_index": 0,
        "created_at": datetime.now(),
    }
    return session_id


def _get_session(session_id: str) -> dict[str, Any] | None:
    """Get session by ID, return None if expired/not found."""
    _cleanup_expired_sessions()
    return _titan_sessions.get(session_id)


@router.post("/titan-premium")
async def titan_premium_analysis(request: TitanPremiumRequest):
    """
    Run Titan Premium AutoML analysis with multi-variant generation and optional Gemma ranking.

    Returns the top-ranked result first, with session_id for pagination ("show more").
    """
    try:
        file_path = secure_file_path(request.filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        df = load_dataset(file_path)

        config = {
            "target_column": request.target_column,
            "n_variants": request.n_variants,
            "holdout_ratio": request.holdout_ratio,
            "enable_gemma_ranking": request.enable_gemma_ranking,
        }

        if request.config_overrides:
            for key, value in request.config_overrides.items():
                if key in TITAN_CONFIG_SCHEMA:
                    config[key] = value

        engine = TitanEngine()
        results = engine.analyze(df, config)

        if "error" in results:
            raise HTTPException(status_code=400, detail=results["message"])

        session_id = _create_session(results, request.filename, config)

        variants = results.get("variants", [])
        total_variants = len(variants)

        start_idx = (request.page - 1) * request.page_size
        end_idx = start_idx + request.page_size
        current_variants = variants[start_idx:end_idx]

        return convert_to_native({
            "status": "success",
            "session_id": session_id,
            "filename": request.filename,
            "task_type": results.get("task_type"),
            "target_column": results.get("target_column"),
            "current_variant": current_variants[0] if current_variants else None,
            "current_page": request.page,
            "total_variants": total_variants,
            "remaining": total_variants - end_idx,
            "has_more": end_idx < total_variants,
            "insights": results.get("insights", []),
            "stable_features": results.get("stable_features", []),
            "feature_importance": results.get("feature_importance", {}),
            "holdout_validation": results.get("holdout_validation"),
            "provenance": results.get("provenance"),
            "config_schema": results.get("config_schema"),
            "layman_summary": results.get("layman_summary"),
            "graph_data": results.get("graph_data"),
            "rows": len(df),
            "columns": len(df.columns),
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Titan Premium analysis failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/titan-premium/{session_id}/next")
async def titan_premium_next(session_id: str, page_size: int = 1):
    """Get next variant(s) from a Titan Premium session."""
    try:
        session = _get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found or expired")

        results = session["results"]
        variants = results.get("variants", [])
        current_idx = session["current_index"]

        start_idx = current_idx + page_size
        end_idx = start_idx + page_size

        if start_idx >= len(variants):
            return convert_to_native({
                "status": "complete",
                "message": "No more variants available",
                "session_id": session_id,
                "total_variants": len(variants),
                "has_more": False,
            })

        session["current_index"] = start_idx
        current_variants = variants[start_idx:end_idx]

        return convert_to_native({
            "status": "success",
            "session_id": session_id,
            "current_variant": current_variants[0] if current_variants else None,
            "variants": current_variants,
            "current_index": start_idx,
            "total_variants": len(variants),
            "remaining": len(variants) - end_idx,
            "has_more": end_idx < len(variants),
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Titan Premium next failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/titan-premium/{session_id}/all")
async def titan_premium_all(session_id: str):
    """Get all variants from a Titan Premium session."""
    try:
        session = _get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found or expired")

        results = session["results"]

        return convert_to_native({
            "status": "success",
            "session_id": session_id,
            "filename": session["filename"],
            "variants": results.get("variants", []),
            "total_variants": len(results.get("variants", [])),
            "insights": results.get("insights", []),
            "provenance": results.get("provenance"),
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Titan Premium all failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/titan-premium/{session_id}/explain")
async def titan_premium_explain(session_id: str):
    """Get detailed explainability/provenance for a Titan Premium session."""
    try:
        session = _get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found or expired")

        results = session["results"]
        provenance = results.get("provenance", {})

        return convert_to_native({
            "status": "success",
            "session_id": session_id,
            "methodology": provenance.get("methodology", {}),
            "pipeline_steps": provenance.get("pipeline_steps", []),
            "configuration_used": provenance.get("configuration_used", {}),
            "feature_stability_scores": provenance.get("feature_stability_scores", {}),
            "data_summary": provenance.get("data_summary", {}),
            "holdout_validation": provenance.get("holdout_validation"),
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Titan Premium explain failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/titan-premium/rerun")
async def titan_premium_rerun(request: TitanPremiumConfigRequest):
    """Re-run Titan Premium analysis with modified configuration."""
    try:
        session = _get_session(request.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found or expired")

        file_path = os.path.join(UPLOAD_DIR, session["filename"])
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Original file not found")

        df = load_dataset(file_path)

        config = session["config"].copy()
        for key, value in request.config_overrides.items():
            if key in TITAN_CONFIG_SCHEMA:
                schema = TITAN_CONFIG_SCHEMA[key]
                param_type = schema.get("type", "str")

                if param_type == "int":
                    value = int(value)
                elif param_type == "float":
                    value = float(value)
                elif param_type == "bool":
                    value = bool(value)

                config[key] = value

        engine = TitanEngine()
        results = engine.analyze(df, config)

        if "error" in results:
            raise HTTPException(status_code=400, detail=results["message"])

        new_session_id = _create_session(results, session["filename"], config)
        variants = results.get("variants", [])

        return convert_to_native({
            "status": "success",
            "session_id": new_session_id,
            "previous_session_id": request.session_id,
            "filename": session["filename"],
            "config_applied": config,
            "current_variant": variants[0] if variants else None,
            "total_variants": len(variants),
            "insights": results.get("insights", []),
            "provenance": results.get("provenance"),
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Titan Premium rerun failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/titan-premium-schema")
async def titan_premium_schema():
    """Return the JSON schema for Titan configuration."""
    return TITAN_CONFIG_SCHEMA


@router.post("/premium/{engine_name}")
async def run_premium_analysis(engine_name: str, request: PremiumAnalysisRequest):
    """
    Run Premium analysis on any flagship engine.

    Supported engines: titan, chaos, scout, oracle, newton, flash, mirror, chronos, deep_feature, galileo
    """
    HEAVY_ENGINES = {"mirror", "titan", "oracle"}
    ENGINE_TIMEOUT = 120 if engine_name.lower() in HEAVY_ENGINES else 60

    try:
        engine_name_lower = engine_name.lower().replace("-", "_")
        if engine_name_lower not in PREMIUM_ENGINES:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown engine: {engine_name}. Available: {list(PREMIUM_ENGINES.keys())}",
            )

        file_path = secure_file_path(request.filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {request.filename}")

        df = load_dataset(file_path)

        config = {"target_column": request.target_column}
        if request.config_overrides:
            config.update(request.config_overrides)

        EngineClass = PREMIUM_ENGINES[engine_name_lower]

        if engine_name_lower == "titan":
            gemma_client = AnalyticsGemmaClient()
            engine = EngineClass(gemma_client=gemma_client)
        else:
            engine = EngineClass()

        async def run_engine_async():
            if hasattr(engine, "run_premium_async"):
                result = await engine.run_premium_async(df.copy(), config)
                if hasattr(result, "to_dict"):
                    return result.to_dict()
                return result
            if hasattr(engine, "analyze_async"):
                return await engine.analyze_async(df.copy(), config)
            return None

        def run_engine_sync():
            if hasattr(engine, "run_premium"):
                result = engine.run_premium(df.copy(), config)
                if hasattr(result, "to_dict"):
                    return result.to_dict()
                return result
            if hasattr(engine, "analyze"):
                return engine.analyze(df.copy(), config)
            return {"status": "error", "message": "No analyze method"}

        try:
            result = await asyncio.wait_for(run_engine_async(), timeout=ENGINE_TIMEOUT)
            if result is None:
                result = await asyncio.wait_for(
                    run_in_threadpool(run_engine_sync), timeout=ENGINE_TIMEOUT
                )
        except TimeoutError:
            raise HTTPException(
                status_code=504,
                detail=f"Engine {engine_name} timed out after {ENGINE_TIMEOUT}s.",
            )

        return convert_to_native({
            "status": "success",
            "engine": engine_name_lower,
            "filename": request.filename,
            **result,
        })

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Premium analysis failed for {engine_name}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/premium/engines")
async def list_premium_engines():
    """List all available premium flagship engines."""
    engine_info = {
        "titan": {"name": "Titan Engine", "description": "Universal AutoML with Gemma ranking", "icon": "🔱"},
        "chaos": {"name": "Chaos Engine", "description": "Monte Carlo simulations", "icon": "🎲"},
        "scout": {"name": "Scout Engine", "description": "Feature exploration and ranking", "icon": "🔍"},
        "oracle": {"name": "Oracle Engine", "description": "Ensemble forecasting", "icon": "🔮"},
        "newton": {"name": "Newton Engine", "description": "Gradient optimization", "icon": "🍎"},
        "flash": {"name": "Flash Engine", "description": "Quick-fit pattern detection", "icon": "⚡"},
        "mirror": {"name": "Mirror Engine", "description": "Synthetic data generation", "icon": "🪞"},
        "chronos": {"name": "Chronos Engine", "description": "Advanced time series", "icon": "⏰"},
        "deep_feature": {"name": "Deep Feature Engine", "description": "Neural feature extraction", "icon": "🧠"},
        "galileo": {"name": "Galileo Engine", "description": "Observational ML insights", "icon": "🔭"},
    }

    return {
        "status": "success",
        "total_engines": len(PREMIUM_ENGINES),
        "engines": [
            {"key": key, **info}
            for key, info in engine_info.items()
        ],
    }


# =============================================================================
# DYNAMIC ENGINE ENDPOINT - For predictions.html compatibility
# =============================================================================

# Extended engine registry mapping names to endpoints
EXTENDED_ENGINE_MAP = {
    # Core premium engines
    "titan": "titan",
    "predictive": "titan",  # Alias
    "clustering": "scout",  
    "anomaly": "chaos",
    "statistical": "newton",
    "trend": "chronos",
    "graphs": "galileo",
    # Financial engines
    "cost": "flash",
    "roi": "flash",
    "spend_patterns": "chronos",
    "budget_variance": "newton",
    "profit_margins": "flash",
    "revenue_forecasting": "chronos",
    "customer_ltv": "oracle",
    "cash_flow": "chronos",
    # Operations engines
    "inventory_optimization": "flash",
    "pricing_strategy": "oracle",
    "market_basket": "galileo",
    "resource_utilization": "flash",
    # Advanced engines
    "rag_evaluation": "scout",
    "chaos": "chaos",
    "oracle": "oracle",
}


@router.post("/run-engine/{engine_name}")
async def run_single_engine(engine_name: str, request: dict):
    """
    Run a single ML engine by name.
    
    This endpoint provides compatibility with the predictions.html frontend
    by mapping engine names to the appropriate premium engine.
    
    Args:
        engine_name: Name of engine to run (e.g., 'titan', 'clustering', 'anomaly')
        request: JSON body with {filename, target_column, config, use_vectorization}
    
    Returns:
        Engine analysis results
    """
    filename = request.get("filename")
    target_column = request.get("target_column")
    config = request.get("config") or {}
    use_vectorization = request.get("use_vectorization", False)
    
    try:
        # Validate file
        if not filename:
            raise HTTPException(status_code=400, detail={"error": "filename is required"})
        
        file_path = secure_file_path(filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail={"error": f"File not found: {filename}"})
        
        # Map engine name to premium engine
        engine_name_lower = engine_name.lower().replace("-", "_")
        mapped_engine = EXTENDED_ENGINE_MAP.get(engine_name_lower, engine_name_lower)
        
        # Check if it's a valid premium engine
        if mapped_engine not in PREMIUM_ENGINES:
            # Return a helpful error with available engines
            available = list(EXTENDED_ENGINE_MAP.keys()) + list(PREMIUM_ENGINES.keys())
            raise HTTPException(
                status_code=404,
                detail={
                    "error": f"Engine '{engine_name}' not found",
                    "mapped_to": mapped_engine,
                    "available_engines": sorted(set(available)),
                },
            )
        
        # Load dataset
        df = load_dataset(file_path)
        
        # Build config
        engine_config = {
            "target_column": target_column,
            "use_vectorization": use_vectorization,
            **config,
        }
        
        # Get engine class and instantiate
        EngineClass = PREMIUM_ENGINES[mapped_engine]
        
        if mapped_engine == "titan":
            gemma_client = AnalyticsGemmaClient()
            engine = EngineClass(gemma_client=gemma_client)
        else:
            engine = EngineClass()
        
        # Run analysis with timeout
        HEAVY_ENGINES = {"mirror", "titan", "oracle"}
        ENGINE_TIMEOUT = 120 if mapped_engine in HEAVY_ENGINES else 60
        
        async def run_engine_async():
            if hasattr(engine, "run_premium_async"):
                result = await engine.run_premium_async(df.copy(), engine_config)
                if hasattr(result, "to_dict"):
                    return result.to_dict()
                return result
            if hasattr(engine, "analyze_async"):
                return await engine.analyze_async(df.copy(), engine_config)
            return None
        
        def run_engine_sync():
            if hasattr(engine, "run_premium"):
                result = engine.run_premium(df.copy(), engine_config)
                if hasattr(result, "to_dict"):
                    return result.to_dict()
                return result
            if hasattr(engine, "analyze"):
                return engine.analyze(df.copy(), engine_config)
            return {"status": "error", "message": "No analyze method"}
        
        try:
            result = await asyncio.wait_for(run_engine_async(), timeout=ENGINE_TIMEOUT)
            if result is None:
                result = await asyncio.wait_for(
                    run_in_threadpool(run_engine_sync), timeout=ENGINE_TIMEOUT
                )
        except TimeoutError:
            raise HTTPException(
                status_code=504,
                detail=f"Engine {engine_name} timed out after {ENGINE_TIMEOUT}s.",
            )
        
        # Add metadata
        result_dict = convert_to_native(result) if result else {}
        result_dict["_engine_metadata"] = {
            "requested_engine": engine_name,
            "mapped_to": mapped_engine,
            "filename": filename,
            "use_vectorization": use_vectorization,
        }
        
        return result_dict
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error running engine {engine_name}: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail={
                "engine": engine_name,
                "error": str(e),
                "type": type(e).__name__,
            },
        )
