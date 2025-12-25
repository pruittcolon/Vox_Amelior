import json
import logging
import os
from typing import Any

import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.automl import AutoMLService

from shared.analytics import AnalyticsEngine
from shared.security.secrets_manager import get_secret

# --- Models ---


class GemmaCRMRequest(BaseModel):
    question: str
    context: dict[str, Any]


# --- App Setup ---

logger = logging.getLogger("insights-service")

APP_ENV = os.getenv("APP_ENV", "local")
DB_PATH = os.getenv("INSIGHTS_DB_PATH", "/app/instance/rag.db")
try:
    DB_KEY = get_secret("rag_db_key")
    if DB_KEY:
        logger.info("[Insights] Loaded rag_db_key secret for analytics access")
    else:
        logger.warning("[Insights] rag_db_key secret not found; falling back to plaintext analytics DB")
except Exception as exc:
    DB_KEY = None
    logger.error("[Insights] Failed to load rag_db_key secret: %s", exc)

engine = AnalyticsEngine(DB_PATH, encryption_key=DB_KEY)
automl = AutoMLService(DB_PATH, db_key=DB_KEY)

logger.info("[Insights] Starting in %s mode using DB %s", APP_ENV, DB_PATH)
if not engine.db_path.exists():
    logger.warning("[Insights] DB path %s does not exist", engine.db_path)

app = FastAPI(title="Insights Service", version="1.0.0")

# Phase 2: Tighten CORS - only allow internal service origins
ALLOWED_ORIGINS = os.getenv(
    "INSIGHTS_ALLOWED_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000,https://localhost,https://127.0.0.1"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID", "X-Service-Token"],
)

# Phase 2: Service-to-Service Authentication Middleware
try:
    from shared.security.service_auth import ServiceAuthMiddleware, _is_test_mode, load_service_jwt_keys

    _jwt_keys = load_service_jwt_keys("insights-service")
    app.add_middleware(
        ServiceAuthMiddleware,
        service_secret=_jwt_keys,
        exempt_paths=["/health", "/docs", "/openapi.json", "/"],
        enabled=not _is_test_mode(),
    )
    logger.info("✅ ServiceAuthMiddleware enabled on Insights service")
except Exception as _auth_err:
    logger.warning(f"⚠️ ServiceAuthMiddleware not loaded: {_auth_err}")


@app.get("/health", tags=["health"])
def healthcheck():
    return {
        "status": "ok",
        "env": APP_ENV,
        "db_path": str(engine.db_path),
        "db_exists": engine.db_path.exists(),
    }


@app.get("/automl/hypotheses", tags=["automl"])
def get_automl_hypotheses():
    """Generate AutoML hypotheses"""
    try:
        # Ensure data is loaded
        automl.load_data()
        return {"hypotheses": automl.generate_hypotheses()}
    except Exception as e:
        logger.error(f"AutoML error: {e}")
        return {"error": str(e), "hypotheses": []}


@app.post("/automl/run", tags=["automl"])
def run_automl_experiment(payload: dict):
    """Run an AutoML experiment"""
    try:
        experiment_id = payload.get("experiment_id")
        return automl.run_experiment(experiment_id)
    except Exception as e:
        logger.error(f"AutoML run error: {e}")
        return {"status": "error", "message": str(e)}


@app.get("/analytics/signals", tags=["analytics"])
def analytics_signals(
    start_date: str | None = Query(None),
    end_date: str | None = Query(None),
    speakers: str | None = Query(None, description="Comma-separated speaker list"),
    emotions: str | None = Query(None, description="Comma-separated emotion list"),
    metrics: str | None = Query(None, description="Comma-separated speech metrics"),
):
    speaker_list: list[str] | None = [s.strip() for s in speakers.split(",") if s.strip()] if speakers else None
    emotion_list: list[str] | None = [e.strip() for e in emotions.split(",") if e.strip()] if emotions else None
    metric_list: list[str] | None = [m.strip() for m in metrics.split(",") if m.strip()] if metrics else None

    try:
        payload = engine.query_signals(
            start_date=start_date,
            end_date=end_date,
            speakers=speaker_list,
            emotions=emotion_list,
            metrics=metric_list,
        )
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.exception("[Insights] analytics_signals failed")
        payload = engine.empty_payload(
            metrics=metric_list,
            fallback_reason="insights_error",
            error=str(exc),
        )
    if not payload.get("summary"):
        # Ensure frontend always receives a complete structure.
        payload = engine.empty_payload(
            metrics=metric_list,
            fallback_reason=payload.get("fallback_reason") or "insights_empty",
            error=str(payload.get("error")) if payload.get("error") else None,
        )
    return payload


@app.get("/analytics/segments", tags=["analytics"])
def analytics_segments(
    start_date: str | None = Query(None),
    end_date: str | None = Query(None),
    speakers: str | None = Query(None, description="Comma-separated speaker list"),
    emotions: str | None = Query(None, description="Comma-separated emotion list"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    order: str = Query("desc"),
):
    speaker_list: list[str] | None = [s.strip() for s in speakers.split(",") if s.strip()] if speakers else None
    emotion_list: list[str] | None = [e.strip() for e in emotions.split(",") if e.strip()] if emotions else None

    try:
        payload = engine.query_segments(
            start_date=start_date,
            end_date=end_date,
            speakers=speaker_list,
            emotions=emotion_list,
            limit=limit,
            offset=offset,
            order=order,
        )
    except Exception as exc:  # pragma: no cover
        logger.exception("[Insights] analytics_segments failed")
        payload = {"items": [], "count": 0, "grouped_by_speaker": [], "error": str(exc)}
    return payload


@app.post("/salesforce/gemma-insights", tags=["salesforce"])
def gemma_crm_insights(request: GemmaCRMRequest):
    """
    Generate AI insights for CRM data using Gemma.
    """
    try:
        # 1. Construct Prompt
        # We assume the context contains Leads, Opportunities, and Summary data
        context_str = json.dumps(request.context, indent=2)

        prompt = f"""You are an advanced CRM AI Assistant. Analyze the following Salesforce data and answer the user's question.

USER QUESTION: {request.question}

CRM DATA CONTEXT:
{context_str}

INSTRUCTIONS:
- Be concise, professional, and actionable.
- Focus on key metrics (Pipeline Value, Win Probability, Hot Leads).
- Highlight risks (stalled deals, low scores).
- If suggesting actions, be specific (e.g., "Schedule a demo with TechCorp", "Follow up on the $50k deal").
- Do not mention that you are analyzing JSON data, just speak naturally as a CRM assistant.
"""

        # 2. Call Gemma Service
        # We try localhost:8001 (internal) or standard service URL
        gemma_url = os.getenv("GEMMA_SERVICE_URL", "http://localhost:8001")

        # Use simple generate endpoint
        try:
            resp = requests.post(
                f"{gemma_url}/generate", json={"prompt": prompt, "max_tokens": 512, "temperature": 0.5}, timeout=30
            )

            if resp.status_code == 200:
                answer = resp.json().get("text", "I analyzed the data but couldn't generate a response.")
                return {"answer": answer}
            else:
                logger.error(f"Gemma Error {resp.status_code}: {resp.text}")
                # Fallback response if Gemma fails
                return {
                    "answer": "I'm currently unable to connect to the AI engine. However, based on the data, I recommend reviewing your 'Red' health opportunities immediately."
                }

        except requests.exceptions.RequestException as e:
            logger.error(f"Gemma Connection Error: {e}")
            return {"answer": "AI Engine unavailable. Please check the dashboard manually for high-risk items."}

    except Exception as e:
        logger.exception("CRM Insights Failed")
        raise HTTPException(status_code=500, detail=str(e))


class UnifiedContextRequest(BaseModel):
    """Request for unified Salesforce + Fiserv insights."""
    question: str
    salesforce_context: dict[str, Any] | None = None
    fiserv_context: dict[str, Any] | None = None
    member_id: str | None = None


@app.post("/enterprise/unified-insights", tags=["enterprise"])
def unified_enterprise_insights(request: UnifiedContextRequest):
    """
    Generate AI insights with BOTH Salesforce CRM and Fiserv Banking context.
    Phase 4: Unified Context Service implementation.
    """
    try:
        # 1. Build unified context
        context_parts = []
        
        if request.salesforce_context:
            context_parts.append("=== SALESFORCE CRM DATA ===")
            context_parts.append(json.dumps(request.salesforce_context, indent=2))
        
        if request.fiserv_context:
            context_parts.append("\n=== FISERV BANKING DATA ===")
            context_parts.append(json.dumps(request.fiserv_context, indent=2))
        
        # 2. If member_id provided, try to fetch Fiserv data
        if request.member_id and not request.fiserv_context:
            fiserv_url = os.getenv("FISERV_SERVICE_URL", "http://fiserv-service:8015")
            try:
                member_resp = requests.get(
                    f"{fiserv_url}/api/v1/datasets/member/{request.member_id}",
                    timeout=10
                )
                if member_resp.status_code == 200:
                    context_parts.append("\n=== FISERV MEMBER DATA ===")
                    context_parts.append(json.dumps(member_resp.json(), indent=2))
            except Exception as e:
                logger.warning(f"Fiserv context fetch failed: {e}")
        
        context_str = "\n".join(context_parts) if context_parts else "No context provided"
        
        # 3. Construct enhanced prompt
        prompt = f"""You are an Enterprise AI Assistant with access to BOTH CRM and Banking data.
Analyze the unified context and answer the user's question with actionable insights.

USER QUESTION: {request.question}

{context_str}

INSTRUCTIONS:
- Cross-reference CRM opportunities with banking member data when available
- Identify patterns (e.g., "High-value CRM lead is also a banking member with strong balances")
- Highlight risks across both systems
- Be specific with recommendations
- Do not mention JSON or data structures, speak naturally as an enterprise advisor
"""

        # 4. Call Gemma Service
        gemma_url = os.getenv("GEMMA_SERVICE_URL", "http://gemma-service:8008")
        
        try:
            resp = requests.post(
                f"{gemma_url}/generate",
                json={"prompt": prompt, "max_tokens": 768, "temperature": 0.5},
                timeout=45
            )
            
            if resp.status_code == 200:
                answer = resp.json().get("text", "Analysis complete but response parsing failed.")
                return {
                    "answer": answer,
                    "sources": ["Salesforce CRM", "Fiserv Banking"] if request.fiserv_context or request.member_id else ["Salesforce CRM"],
                    "context_merged": bool(request.salesforce_context and (request.fiserv_context or request.member_id))
                }
            else:
                logger.error(f"Gemma Error {resp.status_code}: {resp.text}")
                return {
                    "answer": "AI analysis unavailable. Review your CRM pipeline for red-health deals and check banking member status manually.",
                    "sources": [],
                    "context_merged": False
                }
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Gemma Connection Error: {e}")
            return {
                "answer": "Enterprise AI Engine offline. Manual review recommended for cross-system analysis.",
                "sources": [],
                "context_merged": False
            }
            
    except Exception as e:
        logger.exception("Unified Insights Failed")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8010")),
        reload=False,
    )
