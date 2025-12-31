"""
API Gateway Service - Slim Refactored Version
==============================================
Handles authentication, routing, and frontend serving.

This is a clean refactored version that delegates all routes to domain-specific routers.
The original main.py (~4000 lines) has been decomposed into:
- Core: This file (imports, config, middleware, lifespan, router includes)
- Routers: src/routers/*.py (domain-specific API routes)
- Middleware: Middlewares are defined inline here for simplicity

Phase 5.7+ Refactor: December 2024
"""

import base64
import json
import logging
import os
import secrets
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

# Add root directory to path to access shared modules
root_dir = str(Path(__file__).parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# =============================================================================
# Auth Module Import (Fail-Closed)
# =============================================================================
try:
    from src.auth.auth_manager import AuthManager
    from src.auth.permissions import Session, require_auth

    _auth_loaded = True
except ImportError as e:
    _auth_import_error = str(e)
    _auth_loaded = False
    AuthManager = None
    Session = None
    require_auth = None

# =============================================================================
# Logging Configuration
# =============================================================================
try:
    from shared.logging.structured import setup_structured_logging

    if os.getenv("STRUCTURED_LOGGING", "true").lower() in ("true", "1", "yes"):
        setup_structured_logging("api-gateway")
        logger = logging.getLogger(__name__)
        logger.info("✅ Structured logging enabled for API Gateway")
    else:
        raise ImportError("Structured logging disabled via env")
except (ImportError, Exception) as e:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger = logging.getLogger(__name__)
    logger.info(f"Standard logging enabled (Structured logging skipped: {e})")

if _auth_loaded:
    logger.info("Auth modules loaded")
else:
    logger.warning("Auth import failed: %s", _auth_import_error)

# =============================================================================
# Configuration
# =============================================================================
from src.config import SecurityConfig as SecConf

from shared.analysis.fallback_store import AnalysisFallbackStore

# Service URLs
GEMMA_URL = os.getenv("GEMMA_URL", "http://gemma-service:8001")
RAG_URL = os.getenv("RAG_URL", "http://rag-service:8004")
EMOTION_URL = os.getenv("EMOTION_URL", "http://emotion-service:8005")
TRANSCRIPTION_URL = os.getenv("TRANSCRIPTION_URL", "http://transcription-service:8003")
INSIGHTS_URL = os.getenv("INSIGHTS_URL", "http://insights-service:8010")
ML_SERVICE_URL = os.getenv("ML_SERVICE_URL", "http://localhost:8006")

# Security & CORS
ALLOWED_ORIGINS = [
    o.strip() for o in os.getenv("ALLOWED_ORIGINS", "http://127.0.0.1,http://localhost").split(",") if o.strip()
]
SESSION_COOKIE_NAME = SecConf.SESSION_COOKIE_NAME
CSRF_COOKIE_NAME = SecConf.CSRF_COOKIE_NAME

# Directories
APP_INSTANCE_DIR = Path(os.getenv("APP_INSTANCE_DIR", "/app/instance"))
try:
    APP_INSTANCE_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    APP_INSTANCE_DIR = Path("instance")
    APP_INSTANCE_DIR.mkdir(parents=True, exist_ok=True)

FRONTEND_DIR = (
    Path("/app/frontend")
    if Path("/app/frontend").exists()
    else Path(__file__).parent.parent.parent.parent.parent / "frontend"
)
FAVICON_PATH = FRONTEND_DIR / "assets" / "images" / "icons" / "favicon.png"

# Fallback store for analysis
ANALYSIS_FALLBACK_DIR = Path(os.getenv("ANALYSIS_FALLBACK_DIR", str(APP_INSTANCE_DIR / "analysis_fallback")))
fallback_store = AnalysisFallbackStore(
    base_dir=ANALYSIS_FALLBACK_DIR,
    legacy_file=APP_INSTANCE_DIR / "analysis_fallback.json",
    max_per_user=200,
)

# Rate limiting
RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "true").lower() in {"1", "true", "yes"}

# =============================================================================
# Global State
# =============================================================================
auth_manager = None
service_auth = None
personality_jobs: dict[str, dict[str, Any]] = {}
analysis_jobs: dict[str, dict[str, Any]] = {}


# =============================================================================
# Helper Functions (Used by routers via lazy import)
# =============================================================================
def _decode_session_key(raw_value: str | None) -> bytes | None:
    """Decode a base64/urlsafe base64 encoded 32-byte key."""
    if not raw_value:
        return None
    raw_value = raw_value.strip()
    padded = raw_value + "=" * (-len(raw_value) % 4)
    for decoder in (base64.urlsafe_b64decode, base64.b64decode):
        try:
            decoded = decoder(padded.encode("utf-8"))
            if len(decoded) == 32:
                return decoded
        except Exception:
            continue
    return None


# =============================================================================
# Lifespan (Startup/Shutdown)
# =============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global auth_manager, service_auth
    logger.info("Starting API Gateway (Slim)...")

    # PHASE 0: Fail-closed security checks
    from shared.security.startup_checks import assert_secure_mode

    assert_secure_mode()

    # Block startup if auth failed
    if not _auth_loaded:
        raise RuntimeError(f"SECURITY: AuthManager failed to import: {_auth_import_error}")

    # Initialize auth manager
    if AuthManager:
        from src.auth.auth_manager import init_auth_manager

        from shared.security.secrets_manager import get_secret

        session_key_bytes = _decode_session_key(get_secret("session_key"))
        if not session_key_bytes:
            session_key_bytes = _decode_session_key(os.getenv("SESSION_KEY_B64") or os.getenv("SESSION_KEY"))
        if not session_key_bytes:
            logger.warning("Session key not found; generating ephemeral key")
            session_key_bytes = secrets.token_bytes(32)

        users_db_key = get_secret("users_db_key")
        auth_manager = init_auth_manager(
            secret_key=session_key_bytes,
            db_path=str(APP_INSTANCE_DIR / "users.db"),
            db_encryption_key=users_db_key,
        )
        logger.info("✅ Auth manager initialized")

    # Initialize service auth for inter-service JWTs
    try:
        from shared.security.service_auth import get_service_auth, load_service_jwt_keys

        jwt_keys = load_service_jwt_keys("gateway")
        service_auth = get_service_auth(service_id="gateway", service_secret=jwt_keys)
        logger.info("✅ JWT service auth initialized for gateway")
    except Exception as e:
        raise RuntimeError(f"SECURITY: ServiceAuth failed: {e}") from e

    yield
    logger.info("API Gateway shutdown complete")


# =============================================================================
# App Initialization
# =============================================================================
app = FastAPI(title="API Gateway", version="2.0.0", lifespan=lifespan)

# =============================================================================
# Middleware: CORS
# =============================================================================
app.add_middleware(
    CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)


# =============================================================================
# Middleware: Rate Limiting
# =============================================================================
class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self.window = int(os.getenv("RATE_LIMIT_WINDOW_SEC", "60"))
        self.limit = int(os.getenv("RATE_LIMIT_DEFAULT", "120"))
        self.buckets: dict[str, dict[str, int]] = {}

    async def dispatch(self, request: Request, call_next):
        client_ip = request.headers.get("x-forwarded-for", "").split(",")[0].strip() or (
            request.client.host if request.client else "unknown"
        )
        path = request.url.path

        # Skip rate limiting for health and static
        if path.startswith(("/health", "/ui/", "/favicon")):
            return await call_next(request)

        now = int(time.time())
        key = f"{client_ip}:{path}"
        bucket = self.buckets.get(key, {"count": 0, "start": now})

        if now - bucket["start"] > self.window:
            bucket = {"count": 1, "start": now}
        else:
            bucket["count"] += 1

        self.buckets[key] = bucket

        if bucket["count"] > self.limit:
            return Response(status_code=429, content="Too Many Requests")

        return await call_next(request)


if RATE_LIMIT_ENABLED:
    app.add_middleware(RateLimitMiddleware)


# =============================================================================
# Middleware: Security Headers (Enhanced for Enterprise)
# =============================================================================
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Enhanced security headers middleware per OWASP and ISO 27002 guidelines.
    
    Adds comprehensive security headers to ALL responses:
    - X-Content-Type-Options: nosniff (prevent MIME sniffing)
    - X-Frame-Options: DENY (prevent clickjacking)
    - X-XSS-Protection: 0 (deprecated but included for legacy browsers)
    - Referrer-Policy: strict-origin-when-cross-origin
    - Permissions-Policy: restrict sensitive APIs
    - Strict-Transport-Security: HSTS with preload
    - Content-Security-Policy: strict CSP with frame-ancestors
    - X-Permitted-Cross-Domain-Policies: none
    - Cache-Control: no-store for API responses
    
    Also removes potentially leaky headers (Server, X-Powered-By).
    """
    
    def __init__(self, app):
        super().__init__(app)
        # Paths exempt from Cache-Control: no-store (static assets)
        self.cache_exempt_paths = ("/ui/", "/assets/", "/static/", "/favicon")
        # Force HSTS even in development
        self.force_hsts = os.getenv("FORCE_HSTS", "true").lower() in ("true", "1", "yes")
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        path = request.url.path
        
        # Core security headers (applied to ALL responses)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "0"  # Disabled per OWASP - can cause XSS in older browsers
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["X-Permitted-Cross-Domain-Policies"] = "none"
        
        # Permissions Policy - restrict sensitive browser APIs
        response.headers["Permissions-Policy"] = (
            "accelerometer=(), "
            "autoplay=(), "
            "camera=(), "
            "cross-origin-isolated=(), "
            "display-capture=(), "
            "encrypted-media=(), "
            "fullscreen=(self), "
            "geolocation=(), "
            "gyroscope=(), "
            "keyboard-map=(), "
            "magnetometer=(), "
            "microphone=(self), "
            "midi=(), "
            "payment=(), "
            "picture-in-picture=(), "
            "publickey-credentials-get=(), "
            "screen-wake-lock=(), "
            "sync-xhr=(), "
            "usb=(), "
            "xr-spatial-tracking=()"
        )
        
        # HSTS with preload - max-age 1 year, includeSubDomains, preload
        if self.force_hsts:
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains; preload"
            )
        
        # Content Security Policy - strict with frame-ancestors
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'wasm-unsafe-eval' "
            "https://unpkg.com https://cdn.jsdelivr.net https://cdn.plot.ly https://cdnjs.cloudflare.com; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://cdnjs.cloudflare.com; "
            "font-src 'self' https://fonts.gstatic.com data:; "
            "img-src 'self' data: blob: https:; "
            "connect-src 'self' ws: wss: http://localhost:* http://127.0.0.1:* https://unpkg.com https://cdn.jsdelivr.net; "
            "frame-ancestors 'none'; "
            "base-uri 'self'; "
            "form-action 'self'; "
            "object-src 'none'; "
            "upgrade-insecure-requests"
        )
        
        # Cache-Control: no-store for API responses (not static assets)
        if not any(path.startswith(exempt) for exempt in self.cache_exempt_paths):
            if path.startswith("/api/") or path.startswith("/health"):
                response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
                response.headers["Pragma"] = "no-cache"
        
        # Remove potentially leaky headers
        if "Server" in response.headers:
            del response.headers["Server"]
        if "X-Powered-By" in response.headers:
            del response.headers["X-Powered-By"]
        
        return response


app.add_middleware(SecurityHeadersMiddleware)


# =============================================================================
# Middleware: CSRF Protection (Double-Submit Cookie Pattern)
# =============================================================================
class CSRFMiddleware(BaseHTTPMiddleware):
    """
    CSRF protection using double-submit cookie pattern.
    
    Supports:
    - Exempt paths (e.g., /api/auth/login)
    - Bearer token authentication for mobile clients
    - Header/cookie token comparison
    """
    
    def __init__(self, app):
        super().__init__(app)
        self.exempt_paths = [
            "/api/auth/login",
            "/api/auth/register", 
            "/health",
            "/favicon.ico",
            "/upload",      # Multipart file uploads don't include CSRF headers
            "/api/upload",  # Alias for /upload with /api prefix
            "/api/public/chat",  # Public Gemma chat endpoint
            # Gemma endpoints - internal API calls
            "/api/gemma/generate",
            "/api/gemma/warmup",
            "/api/gemma/chat",
            "/api/gemma/stats",
            "/api/gemma/release-session",
            # Transcript endpoints - internal API calls
            "/api/transcripts/count",
            "/api/transcripts/query",
            "/api/transcripts/recent",
            "/api/transcripts/speakers",
            # RAG/vectorize endpoints
            "/vectorize/database",
            "/vectorize/status",
            "/embed",
            "/ask",
            "/databases",
            "/api/databases",
            # SCIM endpoints - IdP provisioning
            "/scim/Users",
            "/scim/Groups",
            "/scim/ServiceProviderConfig",
            "/scim/Schemas",
            "/scim/ResourceTypes",
        ]
        self.exempt_prefixes = [
            "/analytics/",           # ML analytics endpoints
            "/api/analytics/",       # ML analytics with /api prefix
            "/vectorize/",           # Vectorization endpoints
            "/database-scoring/",    # Database quality scoring
            "/quality-insights/",    # Quality Intelligence 3D Dashboard
        ]
        self.bearer_auth_paths = [
            "/api/mobile/",
            "/api/v1/mobile/",
        ]
        self.auth_limit = int(os.getenv("RATE_LIMIT_AUTH", "10"))  # Auth endpoint rate limit

    async def dispatch(self, request: Request, call_next):
        # Skip exempt paths (exact match)
        if request.url.path in self.exempt_paths:
            return await call_next(request)
        
        # Skip exempt prefixes
        if any(request.url.path.startswith(p) for p in self.exempt_prefixes):
            return await call_next(request)
        
        # Skip for Bearer token authenticated paths (mobile clients)
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer ") and any(
            request.url.path.startswith(p) for p in self.bearer_auth_paths
        ):
            return await call_next(request)
        
        # Skip safe methods (GET, HEAD, OPTIONS)
        if request.method in ("GET", "HEAD", "OPTIONS"):
            return await call_next(request)
        
        # Double-submit CSRF validation
        header_token = request.headers.get(CSRF_COOKIE_NAME) or request.headers.get("X-CSRF-Token")
        cookie_token = request.cookies.get(CSRF_COOKIE_NAME)
        
        if not header_token or not cookie_token:
            return Response(status_code=403, content="CSRF token missing")
        
        if not secrets.compare_digest(header_token, cookie_token):
            return Response(status_code=403, content="CSRF token mismatch")
        
        return await call_next(request)


# Enable CSRF protection (disabled in TEST_MODE for easier testing)
if not SecConf.TEST_MODE:
    app.add_middleware(CSRFMiddleware)


# =============================================================================
# Static Files & Favicon
# =============================================================================
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    if FAVICON_PATH.exists():
        return FileResponse(str(FAVICON_PATH), media_type="image/png")
    return Response(status_code=204)


@app.get("/", response_class=HTMLResponse)
async def root_redirect():
    return RedirectResponse(url="/ui/login.html")


# Custom no-cache static files
class NoCacheStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        return response


if FRONTEND_DIR.exists():
    app.mount("/ui", NoCacheStaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
    logger.info(f"✅ Frontend mounted from {FRONTEND_DIR}")


# =============================================================================
# Health Check
# =============================================================================
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "api-gateway", "version": "2.0.0"}


# =============================================================================
# Proxy Helper (Used by all routers)
# =============================================================================
import httpx


async def proxy_request(
    url: str,
    method: str = "POST",
    json: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
    files: dict[str, Any] | None = None,
    extra_headers: dict[str, str] | None = None,
    timeout_seconds: float = 600.0,  # Increased from 120 for large file uploads
):
    """Proxy request to backend service with JWT authentication."""
    headers: dict[str, str] = {}

    # Add service JWT
    if service_auth:
        try:
            token = service_auth.create_token(expires_in=60, aud="internal")
            headers["X-Service-Token"] = token
        except Exception as e:
            logger.error(f"Failed to create service token: {e}")
            raise HTTPException(status_code=503, detail="Service authentication unavailable")

    if extra_headers:
        headers.update(extra_headers)

    try:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            if method == "GET":
                response = await client.get(url, headers=headers, params=params)
            elif method == "POST":
                if files:
                    response = await client.post(url, headers=headers, files=files, data=json or {})
                else:
                    response = await client.post(url, headers=headers, json=json)
            elif method == "PUT":
                response = await client.put(url, headers=headers, json=json)
            elif method == "DELETE":
                response = await client.delete(url, headers=headers)
            else:
                raise ValueError(f"Unsupported method: {method}")

            response.raise_for_status()
            return response.json()
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail=f"Service timeout: {url}")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unavailable: {str(e)}")


def _service_jwt_headers(expires_in: int = 60) -> dict[str, str]:
    """Helper to mint short-lived service JWTs."""
    if not service_auth:
        raise HTTPException(status_code=503, detail="Service authentication unavailable")
    token = service_auth.create_token(expires_in=expires_in, aud="internal")
    return {"X-Service-Token": token}


def format_sse(event: str, payload: dict[str, Any]) -> str:
    """Format a Server-Sent Event."""
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


# =============================================================================
# Gemma Generate with Fallback (Used by gemma router)
# =============================================================================
async def _gemma_generate_with_fallback(
    body: dict[str, Any],
    headers: dict[str, str] | None,
    *,
    fallback_suffix: str = "\n\nRespond with at least one clear sentence.",
    min_temperature: float = 0.55,
):
    """Invoke Gemma /generate and retry once if no text is produced."""

    async def _invoke(payload: dict[str, Any]):
        result = await proxy_request(f"{GEMMA_URL}/generate", "POST", json=payload, extra_headers=headers)
        return result

    def _extract_text(resp: dict[str, Any] | None) -> str:
        if not resp:
            return ""
        return (resp.get("text") or resp.get("response") or "").strip()

    response = await _invoke(body)
    text = _extract_text(response)

    if text:
        return text, response

    # Retry with adjusted parameters
    retry_body = body.copy()
    retry_body["prompt"] = body.get("prompt", "") + fallback_suffix
    retry_body["temperature"] = max(body.get("temperature", 0.4), min_temperature)

    response = await _invoke(retry_body)
    text = _extract_text(response)
    return text, response


# =============================================================================
# Router Imports & Registration
# =============================================================================
logger.info("Loading routers...")

# Core routers
from src.routers.analysis import router as analysis_router
from src.routers.auth import router as auth_router
from src.routers.email_analyzer import router as email_router
from src.routers.enrollment import router as enrollment_router
from src.routers.gemma import router as gemma_router
from src.routers.health import router as health_router
from src.routers.ml import router as ml_router
from src.routers.rag import router as rag_router
from src.routers.transcription import router as transcription_router
from src.routers.transcripts import router as transcripts_router
from src.routers.websocket import router as websocket_router
from src.routers.emotions import router as emotions_router

# SCIM router (Phase 2 - Enterprise Identity)
try:
    from src.routers.scim import router as scim_router
    _scim_available = True
except ImportError:
    scim_router = None
    _scim_available = False

# Enterprise routers
try:
    from src.routers.enterprise import router as enterprise_router

    _enterprise_available = True
except ImportError:
    enterprise_router = None
    _enterprise_available = False

# Banking & Fiserv routers
try:
    from src.routers.banking import router as banking_router
    from src.routers.fiserv import router as fiserv_router

    _banking_available = True
except ImportError:
    banking_router = None
    fiserv_router = None
    _banking_available = False

# Call Intelligence routers
try:
    from src.routers.call_intelligence import router as call_intelligence_router
    from src.routers.call_lifecycle import router as call_lifecycle_router

    _call_available = True
except ImportError:
    call_intelligence_router = None
    call_lifecycle_router = None
    _call_available = False

# Salesforce router
try:
    from src.routers.salesforce import router as salesforce_router

    _salesforce_available = True
except ImportError:
    salesforce_router = None
    _salesforce_available = False

# Phase 4 - Workflow Orchestration routers
try:
    from src.routers.automation import router as automation_router
    from src.routers.models import router as models_router
    from src.routers.prompts import router as prompts_router

    _phase4_available = True
except ImportError:
    automation_router = None
    models_router = None
    prompts_router = None
    _phase4_available = False

# Phase 5 - Reliability and FinOps routers
try:
    from src.routers.analytics import router as analytics_router

    _phase5_available = True
except ImportError:
    analytics_router = None
    _phase5_available = False

# =============================================================================
# Register All Routers
# =============================================================================
app.include_router(auth_router)
logger.info("✅ Auth Router mounted")

app.include_router(health_router)
logger.info("✅ Health Router mounted")

app.include_router(gemma_router)
logger.info("✅ Gemma Router mounted")

app.include_router(rag_router)
logger.info("RAG Router mounted")

app.include_router(transcripts_router)
logger.info("Transcripts Router mounted")

app.include_router(ml_router)
logger.info("ML Router mounted")

app.include_router(email_router)
logger.info("Email Analyzer Router mounted")

app.include_router(transcription_router)
logger.info("Transcription Router mounted")

app.include_router(analysis_router)
logger.info("✅ Analysis Router mounted")

app.include_router(emotions_router)
logger.info("✅ Emotions Router mounted")

app.include_router(enrollment_router)
logger.info("✅ Enrollment Router mounted")

app.include_router(websocket_router)
logger.info("✅ WebSocket Router mounted")

# SCIM Router (Phase 2 - Enterprise Identity)
if _scim_available:
    app.include_router(scim_router)
    logger.info("✅ SCIM Router mounted")

if _enterprise_available:
    app.include_router(enterprise_router)
    logger.info("✅ Enterprise Router mounted")

if _banking_available:
    app.include_router(banking_router)
    app.include_router(fiserv_router)
    logger.info("✅ Banking & Fiserv Routers mounted")

if _call_available:
    app.include_router(call_intelligence_router)
    app.include_router(call_lifecycle_router)
    logger.info("✅ Call Intelligence Routers mounted")

if _salesforce_available:
    app.include_router(salesforce_router)
    logger.info("✅ Salesforce Router mounted")

# Phase 4 - Workflow Orchestration
if _phase4_available:
    app.include_router(automation_router)
    app.include_router(models_router)
    app.include_router(prompts_router)
    logger.info("✅ Phase 4 Routers (automation, models, prompts) mounted")

# Phase 5 - Reliability and FinOps
if _phase5_available:
    app.include_router(analytics_router)
    logger.info("✅ Phase 5 Analytics Router mounted")

# =============================================================================
# API Versioning
# =============================================================================
v1 = APIRouter(prefix="/api/v1", tags=["v1"])
app.include_router(v1)
logger.info("✅ API v1 Router mounted")

# =============================================================================
# Startup Complete
# =============================================================================
router_count = 15 + (3 if _phase4_available else 0) + (1 if _phase5_available else 0)
logger.info("🚀 API Gateway (Slim) initialized with %d routers", router_count)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
