"""
Emotion Analysis Service
Wraps existing emotion_analyzer.py (CPU-only)
"""

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware

# Add parent directories to path for imports
# Add root directory to path to access shared modules
# In Docker, shared modules are copied to /app/ directly
app_dir = str(Path(__file__).parent.parent)
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)
sys.path.insert(0, "/app")

# Import existing emotion_analyzer (unchanged)
try:
    import emotion_analyzer

    analyze_emotion = emotion_analyzer.analyze_emotion
    initialize_emotion_classifier = emotion_analyzer.initialize_emotion_classifier
    print("[EMOTION] Successfully imported emotion_analyzer")
except ImportError as e:
    print(f"[EMOTION] ERROR: Failed to import emotion_analyzer: {e}")
    emotion_analyzer = None
    analyze_emotion = None
    initialize_emotion_classifier = None

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Configuration
JWT_ONLY = os.getenv("JWT_ONLY", "false").lower() in {"1", "true", "yes"}

# Global emotion classifier
classifier = None

# Global auth
service_auth = None


# ============================================================================
# Service Initialization
# ============================================================================


class ServiceAuthMiddleware(BaseHTTPMiddleware):
    """Middleware for JWT service authentication (Phase 3: Enforce JWT-only + replay)."""

    async def dispatch(self, request: Request, call_next):
        # Skip auth for health checks
        if request.url.path == "/health":
            return await call_next(request)

        jwt_token = request.headers.get("X-Service-Token")
        if not jwt_token or not service_auth:
            logger.error(f"❌ Missing JWT for {request.url.path}")
            return JSONResponse(status_code=401, content={"detail": "Missing service token"})

        try:
            # Allowed callers: gateway, transcription-service
            allowed = ["gateway", "transcription-service"]
            payload = service_auth.verify_token(jwt_token, allowed_services=allowed, expected_aud="internal")

            # Note: Replay protection is already handled inside verify_token()
            # No need for additional check here

            rid_short = str(payload.get("request_id", ""))[:8]
            logger.info(f"✅ JWT OK s={payload.get('service_id')} aud=internal rid={rid_short} path={request.url.path}")
            return await call_next(request)
        except Exception as e:
            logger.error(f"❌ JWT rejected: {e} path={request.url.path}")
            return JSONResponse(status_code=401, content={"detail": "Invalid service token"})


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup and shutdown"""
    global classifier, service_auth

    # Startup
    logger.info("Starting Emotion Service...")

    # ISO 27002: Fail-closed security check
    from shared.security.startup_checks import assert_secure_mode

    assert_secure_mode()  # Blocks startup if SECURE_MODE=true with unsafe flags

    # Initialize service auth (Phase 3)
    try:
        from shared.security.service_auth import get_service_auth, load_service_jwt_keys

        jwt_keys = load_service_jwt_keys("emotion-service")
        service_auth = get_service_auth(service_id="emotion-service", service_secret=jwt_keys)
        logger.info(
            "✅ JWT service auth initialized (enforcing JWT-only, aud=internal, replay protected, keys=%s)",
            len(jwt_keys),
        )
    except Exception as e:
        logger.error(f"❌ JWT service auth initialization failed: {e}")
        raise

    if analyze_emotion is None or initialize_emotion_classifier is None:
        logger.error("emotion_analyzer not available!")
        raise RuntimeError("Cannot start Emotion service without emotion_analyzer")

    try:
        # Initialize classifier (CPU-only)
        initialize_emotion_classifier()
        # Get classifier from the module (it's set as a global there)
        classifier = emotion_analyzer.emotion_classifier

        logger.info("Emotion Service started successfully (CPU-only)")

    except Exception as e:
        logger.error(f"Failed to start Emotion service: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down Emotion Service...")
    if classifier:
        del classifier
    logger.info("Emotion Service shutdown complete")


# Initialize FastAPI
app = FastAPI(title="Emotion Service", description="Text Emotion Analysis Service", version="1.0.0", lifespan=lifespan)

# Add JWT middleware (Phase 2: Permissive)
app.add_middleware(ServiceAuthMiddleware)


# ============================================================================
# Request/Response Models
# ============================================================================


class EmotionAnalyzeRequest(BaseModel):
    text: str


class EmotionBatchRequest(BaseModel):
    texts: list[str]


# ============================================================================
# Health & Status Endpoints
# ============================================================================


@app.get("/health")
def health() -> dict[str, Any]:
    """Health check endpoint"""
    if classifier is None:
        return {"status": "unhealthy", "error": "Classifier not initialized"}

    return {
        "status": "healthy",
        "classifier_loaded": classifier is not None,
        "model": "j-hartmann/emotion-english-distilroberta-base",
    }


# ============================================================================
# Emotion Analysis Endpoints
# ============================================================================


@app.post("/analyze")
def analyze_emotion_endpoint(request: EmotionAnalyzeRequest) -> dict[str, Any]:
    """Analyze emotion in text"""
    if classifier is None:
        raise HTTPException(status_code=503, detail="Classifier not initialized")

    try:
        result = analyze_emotion(request.text)
        # Return flat structure for compatibility
        return {
            "success": True,
            "emotion": result["dominant_emotion"],
            "confidence": result["confidence"],
            "scores": result["emotions"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/batch")
def analyze_batch(request: EmotionBatchRequest) -> dict[str, Any]:
    """Batch analyze emotions"""
    if classifier is None:
        raise HTTPException(status_code=503, detail="Classifier not initialized")

    try:
        results = []
        for text in request.texts:
            result = analyze_emotion(text)
            results.append(result)

        return {"success": True, "emotions": results, "count": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/cli/test")
def run_self_test() -> dict[str, Any]:
    """Self-test endpoint for CLI testing"""
    test_results = []

    # Test 1: Classifier loaded
    test_results.append(
        {
            "test": "classifier_loaded",
            "passed": classifier is not None,
            "details": f"Emotion classifier initialized: {classifier is not None}",
        }
    )

    # Test 2: Test emotion analysis
    if classifier is not None:
        try:
            test_result = analyze_emotion("I am very happy today!")
            test_results.append(
                {
                    "test": "test_emotion_analysis",
                    "passed": "dominant_emotion" in test_result,
                    "details": f"Test analysis successful, detected: {test_result.get('dominant_emotion')}",
                }
            )
        except Exception as e:
            test_results.append(
                {"test": "test_emotion_analysis", "passed": False, "details": f"Test analysis failed: {str(e)[:50]}"}
            )
    else:
        test_results.append({"test": "test_emotion_analysis", "passed": False, "details": "Classifier not loaded"})

    passed = sum(1 for t in test_results if t["passed"])
    total = len(test_results)

    return {
        "test_suite": "emotion_service",
        "summary": {"total": total, "passed": passed, "failed": total - passed},
        "results": test_results,
        "overall_passed": passed == total,
    }



if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8005, log_level="info")

