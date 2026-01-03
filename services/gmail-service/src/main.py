"""
Gmail Automation Service - Main FastAPI Application.

Provides OAuth, email fetching, and Gemma-powered analysis endpoints
for Gmail integration with the Vox Amelior platform.
"""

import logging
import os
import secrets
import sys
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware

# Add shared modules to path
sys.path.insert(0, "/app")

from .email_processor import EmailProcessor, format_sse_event
from .gmail_client import GmailClient
from .models import (
    AnalysisPreset,
    EmailAnalyzeRequest,
    EmailAnalyzeResponse,
    EmailFetchRequest,
    EmailFetchResponse,
    OAuthCallbackRequest,
    OAuthDisconnectResponse,
    OAuthStatusResponse,
    OAuthURLResponse,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("gmail-service")

# Configuration
SERVICE_PORT = int(os.getenv("SERVICE_PORT", "8016"))
GEMMA_SERVICE_URL = os.getenv("GEMMA_SERVICE_URL", "http://gemma-service:8001")
RAG_SERVICE_URL = os.getenv("RAG_SERVICE_URL", "http://rag-service:8004")

# Feature flag
GMAIL_SERVICE_ENABLED = os.getenv("GMAIL_SERVICE_ENABLED", "true").lower() in {"1", "true", "yes"}

# Service auth
service_auth = None


def _ensure_gmail_enabled() -> None:
    """Raise 503 if Gmail service is disabled."""
    if not GMAIL_SERVICE_ENABLED:
        raise HTTPException(
            status_code=503,
            detail="Gmail service is not enabled",
        )


class ServiceAuthMiddleware(BaseHTTPMiddleware):
    """Middleware for JWT service authentication."""

    async def dispatch(self, request: Request, call_next):
        # Skip auth for health checks and SSE streams (browser EventSource cannot send headers)
        exempt_paths = ["/health", "/healthz", "/emails/analyze/stream"]
        if request.url.path in exempt_paths:
            return await call_next(request)

        # Allow localhost for internal testing
        client_host = request.client.host if request.client else None
        if client_host in ["127.0.0.1", "localhost", "::1"]:
            return await call_next(request)

        jwt_token = request.headers.get("X-Service-Token")
        if not jwt_token or not service_auth:
            logger.warning(f"Missing JWT for {request.url.path}")
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing service token"},
            )

        try:
            allowed = ["gateway", "api-gateway"]
            payload = service_auth.verify_token(
                jwt_token,
                allowed_services=allowed,
                expected_aud="internal",
            )
            logger.debug(f"JWT OK for {request.url.path}")
            return await call_next(request)
        except Exception as e:
            logger.error(f"JWT rejected: {e}")
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid service token"},
            )


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Adds request logging with latency tracking."""

    async def dispatch(self, request: Request, call_next):
        import time

        req_id = request.headers.get("X-Request-Id") or uuid.uuid4().hex[:12]
        start = time.monotonic()
        path = request.url.path
        method = request.method

        try:
            response = await call_next(request)
            duration_ms = int((time.monotonic() - start) * 1000)
            response.headers["X-Request-Id"] = req_id
            logger.info(f"[ACCESS] {method} {path} {response.status_code} rid={req_id} {duration_ms}ms")
            return response
        except Exception as exc:
            duration_ms = int((time.monotonic() - start) * 1000)
            logger.error(f"[ERROR] {method} {path} rid={req_id} {duration_ms}ms err={exc}")
            raise


# Store OAuth state tokens (in production, use Redis)
_oauth_states: dict[str, str] = {}


def get_user_id(request: Request) -> str:
    """Extract user ID from request headers or session.

    In production, this would come from the JWT token validated
    by the API Gateway.
    """
    user_id = request.headers.get("X-User-Id")
    if not user_id:
        # Fallback for development
        user_id = "default-user"
    return user_id


def get_gmail_client(request: Request) -> GmailClient:
    """Dependency to get Gmail client for current user."""
    user_id = get_user_id(request)
    return GmailClient(user_id)


def get_email_processor(request: Request) -> EmailProcessor:
    """Dependency to get email processor with service headers."""
    service_headers = {}
    service_token = request.headers.get("X-Service-Token")
    if service_token:
        service_headers["X-Service-Token"] = service_token
    return EmailProcessor(service_headers)


# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup and shutdown."""
    logger.info("Starting Gmail Automation Service...")
    logger.info("=" * 60)

    # Initialize service auth
    global service_auth
    try:
        from shared.security.service_auth import get_service_auth, load_service_jwt_keys

        jwt_keys = load_service_jwt_keys("gmail-service")
        service_auth = get_service_auth(
            service_id="gmail-service",
            service_secret=jwt_keys,
        )
        logger.info("JWT service auth initialized")
    except Exception as e:
        logger.warning(f"JWT auth initialization failed: {e} - continuing without auth")

    logger.info(f"Gmail service enabled: {GMAIL_SERVICE_ENABLED}")
    logger.info(f"Gemma service URL: {GEMMA_SERVICE_URL}")
    logger.info("Gmail Automation Service started successfully")
    logger.info("=" * 60)

    yield

    logger.info("Shutting down Gmail Automation Service...")


# Initialize FastAPI
app = FastAPI(
    title="Gmail Automation Service",
    description="Gmail API integration with Gemma-powered email analysis",
    version="1.0.0",
    lifespan=lifespan,
)

# Add middleware
app.add_middleware(ServiceAuthMiddleware)
app.add_middleware(RequestLoggingMiddleware)

# CORS configuration
CORS_ORIGINS = [
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "https://localhost",
    "https://127.0.0.1",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# =============================================================================
# Health Endpoints
# =============================================================================


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "gmail-service",
        "enabled": GMAIL_SERVICE_ENABLED,
    }


@app.get("/healthz")
async def healthz():
    """Kubernetes health probe."""
    return {"status": "ok"}


# =============================================================================
# OAuth Endpoints
# =============================================================================


@app.get("/oauth/url", response_model=OAuthURLResponse)
async def get_oauth_url(
    request: Request,
    gmail: GmailClient = Depends(get_gmail_client),
):
    """Generate Google OAuth authorization URL.

    Returns:
        OAuth URL and state token for redirect.
    """
    _ensure_gmail_enabled()

    # Generate CSRF state token
    state = secrets.token_urlsafe(32)
    user_id = get_user_id(request)
    _oauth_states[state] = user_id

    try:
        auth_url, _ = gmail.get_authorization_url(state=state)
        return OAuthURLResponse(
            authorization_url=auth_url,
            state=state,
        )
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/oauth/callback")
async def oauth_callback(
    request: Request,
    callback: OAuthCallbackRequest,
):
    """Handle OAuth callback from Google.

    Stores encrypted tokens and returns connection details.
    """
    _ensure_gmail_enabled()

    # Verify state token
    stored_user_id = _oauth_states.pop(callback.state, None)
    if not stored_user_id:
        raise HTTPException(
            status_code=400,
            detail="Invalid or expired state token",
        )

    gmail = GmailClient(stored_user_id)

    try:
        result = gmail.handle_callback(callback.code, callback.state)
        return {
            "success": True,
            "email": result.get("email"),
            "message": "Gmail connected successfully",
        }
    except Exception as e:
        logger.error(f"OAuth callback failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/oauth/status", response_model=OAuthStatusResponse)
async def get_oauth_status(
    gmail: GmailClient = Depends(get_gmail_client),
):
    """Check Gmail OAuth connection status."""
    _ensure_gmail_enabled()

    status = gmail.get_status()
    return OAuthStatusResponse(**status)


@app.post("/oauth/disconnect", response_model=OAuthDisconnectResponse)
async def disconnect_gmail(
    gmail: GmailClient = Depends(get_gmail_client),
):
    """Disconnect Gmail and revoke tokens."""
    _ensure_gmail_enabled()

    success = gmail.disconnect()
    return OAuthDisconnectResponse(
        success=success,
        message="Gmail disconnected" if success else "Disconnect failed",
    )


# =============================================================================
# Email Endpoints
# =============================================================================


@app.post("/emails/fetch", response_model=EmailFetchResponse)
async def fetch_emails(
    request: EmailFetchRequest,
    gmail: GmailClient = Depends(get_gmail_client),
):
    """Fetch emails from Gmail with filters.

    Requires active OAuth connection.
    """
    _ensure_gmail_enabled()

    if not gmail.is_connected():
        raise HTTPException(
            status_code=401,
            detail="Gmail not connected. Please authenticate first.",
        )

    try:
        emails = gmail.fetch_emails(request)
        start_date, end_date = gmail._calculate_date_range(request)

        return EmailFetchResponse(
            success=True,
            emails=emails,
            total_count=len(emails),
            fetched_count=len(emails),
            timeframe_start=start_date,
            timeframe_end=end_date,
        )
    except Exception as e:
        logger.error(f"Email fetch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/emails/analyze", response_model=EmailAnalyzeResponse)
async def analyze_emails(
    request: Request,
    analyze_request: EmailAnalyzeRequest,
    gmail: GmailClient = Depends(get_gmail_client),
    processor: EmailProcessor = Depends(get_email_processor),
):
    """Analyze emails with Gemma LLM.

    Fetches specified emails and runs analysis prompt on each.
    """
    _ensure_gmail_enabled()

    if not gmail.is_connected():
        raise HTTPException(
            status_code=401,
            detail="Gmail not connected. Please authenticate first.",
        )

    # Validate custom prompt if needed
    if analyze_request.preset == AnalysisPreset.CUSTOM:
        if not analyze_request.custom_prompt:
            raise HTTPException(
                status_code=400,
                detail="custom_prompt required for CUSTOM preset",
            )

    # Fetch the requested emails
    emails = []
    for email_id in analyze_request.email_ids:
        email = gmail.get_email_by_id(email_id)
        if email:
            emails.append(email)

    if not emails:
        raise HTTPException(
            status_code=404,
            detail="No valid emails found for analysis",
        )

    try:
        import time

        start_time = time.monotonic()

        results = await processor.analyze_batch(emails, analyze_request)

        total_time_ms = int((time.monotonic() - start_time) * 1000)
        total_tokens = sum(r.tokens_used for r in results)

        # Store results if requested
        if analyze_request.store_results:
            user_id = get_user_id(request)
            await processor.store_results_in_rag(results, user_id)

        return EmailAnalyzeResponse(
            success=True,
            results=results,
            total_emails=len(results),
            total_tokens=total_tokens,
            total_time_ms=total_time_ms,
        )

    except Exception as e:
        logger.error(f"Email analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/emails/analyze/stream")
async def analyze_emails_stream(
    request: Request,
    email_ids: str = Query(..., description="Comma-separated email IDs"),
    preset: AnalysisPreset = Query(AnalysisPreset.SUMMARIZE),
    custom_prompt: str | None = Query(None),
    max_tokens: int = Query(512, ge=64, le=2048),
    temperature: float = Query(0.4, ge=0.0, le=1.0),
    gmail: GmailClient = Depends(get_gmail_client),
    processor: EmailProcessor = Depends(get_email_processor),
):
    """Stream email analysis via Server-Sent Events.

    Provides real-time progress as each email is analyzed.
    """
    _ensure_gmail_enabled()

    if not gmail.is_connected():
        raise HTTPException(
            status_code=401,
            detail="Gmail not connected",
        )

    # Parse email IDs
    ids = [id.strip() for id in email_ids.split(",") if id.strip()]
    if not ids:
        raise HTTPException(status_code=400, detail="No email IDs provided")

    # Validate custom prompt
    if preset == AnalysisPreset.CUSTOM and not custom_prompt:
        raise HTTPException(
            status_code=400,
            detail="custom_prompt required for CUSTOM preset",
        )

    # Fetch emails
    emails = [gmail.get_email_by_id(id) for id in ids]
    emails = [e for e in emails if e is not None]

    if not emails:
        raise HTTPException(status_code=404, detail="No valid emails found")

    # Build request
    analyze_request = EmailAnalyzeRequest(
        email_ids=ids,
        preset=preset,
        custom_prompt=custom_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    async def event_generator():
        try:
            async for event in processor.analyze_stream(emails, analyze_request):
                yield format_sse_event(event)
        except Exception as e:
            from .models import AnalysisStreamEvent

            error_event = AnalysisStreamEvent(
                event_type="error",
                error=str(e),
            )
            yield format_sse_event(error_event)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# =============================================================================
# Main Entry Point
# =============================================================================


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=SERVICE_PORT,
        reload=True,
    )
