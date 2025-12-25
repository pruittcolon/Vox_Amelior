"""
Analysis History Router

Session and run management endpoints:
- /history/sessions - List all sessions
- /history/sessions/{session_id} - Get session details
- /history/sessions (POST) - Create session
- /history/list-sessions - List active smart-analyze sessions
"""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException

# Import with fallback for different import contexts
try:
    from ..utils.analytics_utils import convert_to_native
except ImportError:
    from utils.analytics_utils import convert_to_native

# Import history service
try:
    from ..analysis_history import historyService
except ImportError:
    try:
        from analysis_history import historyService
    except ImportError:
        historyService = None

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/history", tags=["analysis_history"])


@router.get("/sessions")
async def get_all_history_sessions():
    """
    Get all analysis sessions with run counts.
    Returns list of sessions sorted by most recent first.
    """
    if not historyService:
        raise HTTPException(status_code=503, detail="History service unavailable")

    try:
        sessions = historyService.getAllSessions()
        return convert_to_native({
            "status": "success",
            "sessions": sessions,
            "total": len(sessions),
        })
    except Exception as e:
        logger.error(f"Failed to get sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}")
async def get_session_details(session_id: str):
    """
    Get detailed session info including run summary per engine.
    """
    if not historyService:
        raise HTTPException(status_code=503, detail="History service unavailable")

    try:
        session = historyService.getSession(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        return convert_to_native({
            "status": "success",
            "session": session,
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sessions")
async def create_session(request: dict[str, Any]):
    """
    Create or update a session record.
    Called when user uploads a file for analysis.
    """
    if not historyService:
        raise HTTPException(status_code=503, detail="History service unavailable")

    try:
        session_id = request.get("session_id")
        filename = request.get("filename")
        columns = request.get("columns", [])
        row_count = request.get("row_count", 0)

        if not session_id or not filename:
            raise HTTPException(
                status_code=400, detail="session_id and filename required"
            )

        historyService.saveSession(session_id, filename, columns, row_count)

        return {
            "status": "success",
            "session_id": session_id,
            "message": "Session created/updated",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}/engines/{engine_name}")
async def get_engine_runs(session_id: str, engine_name: str):
    """
    Get all runs for a specific engine in a session.
    Used for run navigation (< 1 of 3 >).
    """
    if not historyService:
        raise HTTPException(status_code=503, detail="History service unavailable")

    try:
        runs = historyService.getEngineRuns(session_id, engine_name)
        return convert_to_native({
            "status": "success",
            "session_id": session_id,
            "engine_name": engine_name,
            "runs": runs,
            "total_runs": len(runs),
        })
    except Exception as e:
        logger.error(f"Failed to get engine runs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}/engines/{engine_name}/runs/{run_index}")
async def get_specific_run(session_id: str, engine_name: str, run_index: int):
    """
    Get a specific run by index.
    Used when navigating between runs.
    """
    if not historyService:
        raise HTTPException(status_code=503, detail="History service unavailable")

    try:
        run = historyService.getRun(session_id, engine_name, run_index)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")

        return convert_to_native({
            "status": "success",
            "session_id": session_id,
            "engine_name": engine_name,
            "run_index": run_index,
            "run": run,
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get run: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/save-run")
async def save_run(request: dict[str, Any]):
    """
    Save an analysis run result.
    Called by frontend after each engine completes.
    
    Expected request body:
    {
        "session_id": str,
        "engine_name": str,
        "target_column": str (optional),
        "filename": str,
        "result": dict,
        "run_id": str (optional)
    }
    """
    if not historyService:
        raise HTTPException(status_code=503, detail="History service unavailable")

    try:
        session_id = request.get("session_id")
        engine_name = request.get("engine_name")
        filename = request.get("filename")
        result = request.get("result", {})
        target_column = request.get("target_column")
        
        if not session_id or not engine_name:
            raise HTTPException(
                status_code=400, 
                detail="session_id and engine_name are required"
            )
        
        # Ensure session exists
        if filename:
            columns = result.get("columns", [])
            row_count = result.get("rows", 0)
            historyService.saveSession(session_id, filename, columns, row_count)
        
        # Get next run index
        run_index = historyService.getNextRunIndex(session_id, engine_name)
        
        # Extract relevant fields from result
        gemma_summary = result.get("gemma_summary") or result.get("gemmaSummary")
        feature_columns = result.get("features_used") or result.get("feature_columns")
        score = result.get("best_score") or result.get("cv_score") or result.get("score")
        
        # Save the run
        historyService.saveRun(
            sessionId=session_id,
            engineName=engine_name,
            runIndex=run_index,
            results=result,
            targetColumn=target_column,
            featureColumns=feature_columns if isinstance(feature_columns, list) else None,
            gemmaSummary=gemma_summary,
            config=result.get("config"),
            score=float(score) if score else None,
        )
        
        return {
            "status": "success",
            "session_id": session_id,
            "engine_name": engine_name,
            "run_index": run_index,
            "message": "Run saved successfully",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to save run: {e}")
        raise HTTPException(status_code=500, detail=str(e))

