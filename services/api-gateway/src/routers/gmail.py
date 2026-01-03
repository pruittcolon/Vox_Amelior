"""
Gmail Automation Router - Proxies requests to gmail-service.

Provides OAuth flow, email fetching, and Gemma-powered analysis endpoints
following the gateway proxy pattern.
"""

import json
import logging
import os
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

try:
    from src.auth.permissions import Session, require_auth
except ImportError:

    def require_auth():
        return None

    Session = None

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/gmail", tags=["gmail"])

GMAIL_SERVICE_URL = os.getenv("GMAIL_SERVICE_URL", "http://gmail-service:8016")

# Feature flag
GMAIL_ENABLED = os.getenv("GMAIL_SERVICE_ENABLED", "true").lower() in {"1", "true", "yes"}


def _get_proxy_request():
    """Lazy import to avoid circular dependencies."""
    from src.main import proxy_request

    return proxy_request


def _get_service_headers():
    """Get service JWT headers for internal requests."""
    try:
        from src.main import get_service_headers

        return get_service_headers()
    except ImportError:
        return {}


def _ensure_gmail_enabled():
    """Raise 503 if Gmail service is disabled."""
    if not GMAIL_ENABLED:
        raise HTTPException(
            status_code=503,
            detail="Gmail automation is not enabled",
        )


def _add_user_header(request: Request, headers: dict[str, str]) -> dict[str, str]:
    """Add user ID header for gmail-service to identify the user."""
    # Get user ID from session or header
    user_id = request.headers.get("X-User-Id")
    if not user_id:
        # Try to get from auth session
        try:
            from src.auth.permissions import get_current_user

            user = get_current_user(request)
            if user:
                user_id = str(user.id)
        except Exception:
            pass

    if user_id:
        headers["X-User-Id"] = user_id

    return headers


# =============================================================================
# OAuth Endpoints
# =============================================================================


@router.get("/oauth/url")
async def get_oauth_url(request: Request, session: Session = Depends(require_auth)):
    """Generate Google OAuth authorization URL.

    Returns the URL to redirect the user to for Gmail authentication.
    """
    _ensure_gmail_enabled()
    proxy_request = _get_proxy_request()

    headers = _add_user_header(request, _get_service_headers())

    return await proxy_request(
        f"{GMAIL_SERVICE_URL}/oauth/url",
        "GET",
        extra_headers=headers,
    )


@router.get("/oauth/callback")
async def oauth_callback_get(
    request: Request,
    code: str = Query(...),
    state: str = Query(...),
    scope: str = Query(None),
    session: Session = Depends(require_auth),
):
    """Handle OAuth callback redirect from Google (GET request).

    Query params:
        code: Authorization code from Google
        state: CSRF state token
        scope: Granted scopes
    """
    _ensure_gmail_enabled()
    proxy_request = _get_proxy_request()

    headers = _add_user_header(request, _get_service_headers())

    # Forward to gmail-service as POST with JSON body
    result = await proxy_request(
        f"{GMAIL_SERVICE_URL}/oauth/callback",
        "POST",
        json={"code": code, "state": state},
        extra_headers=headers,
    )

    # Redirect to Gmail dashboard on success
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/ui/gmail.html?connected=true", status_code=302)


@router.post("/oauth/callback")
async def oauth_callback_post(
    request: Request,
    payload: dict[str, Any],
    session: Session = Depends(require_auth),
):
    """Handle OAuth callback from frontend (POST request).

    Body:
        code: Authorization code from Google
        state: CSRF state token
    """
    _ensure_gmail_enabled()
    proxy_request = _get_proxy_request()

    headers = _add_user_header(request, _get_service_headers())

    return await proxy_request(
        f"{GMAIL_SERVICE_URL}/oauth/callback",
        "POST",
        json=payload,
        extra_headers=headers,
    )


@router.get("/oauth/status")
async def get_oauth_status(request: Request, session: Session = Depends(require_auth)):
    """Check Gmail OAuth connection status.

    Returns whether Gmail is connected and the connected email address.
    """
    _ensure_gmail_enabled()
    proxy_request = _get_proxy_request()

    headers = _add_user_header(request, _get_service_headers())

    return await proxy_request(
        f"{GMAIL_SERVICE_URL}/oauth/status",
        "GET",
        extra_headers=headers,
    )


@router.post("/oauth/disconnect")
async def disconnect_gmail(request: Request, session: Session = Depends(require_auth)):
    """Disconnect Gmail and revoke OAuth tokens."""
    _ensure_gmail_enabled()
    proxy_request = _get_proxy_request()

    headers = _add_user_header(request, _get_service_headers())

    return await proxy_request(
        f"{GMAIL_SERVICE_URL}/oauth/disconnect",
        "POST",
        extra_headers=headers,
    )


# =============================================================================
# Email Endpoints
# =============================================================================


@router.post("/emails/fetch")
async def fetch_emails(
    request: Request,
    payload: dict[str, Any],
    session: Session = Depends(require_auth),
):
    """Fetch emails from Gmail with filters.

    Body:
        timeframe: "24h", "7d", "30d", "90d", or "custom"
        start_date: ISO date (for custom timeframe)
        end_date: ISO date (for custom timeframe)
        labels: List of Gmail labels to filter
        max_results: Maximum emails to fetch (1-500)
        query: Gmail search query
    """
    _ensure_gmail_enabled()
    proxy_request = _get_proxy_request()

    headers = _add_user_header(request, _get_service_headers())

    return await proxy_request(
        f"{GMAIL_SERVICE_URL}/emails/fetch",
        "POST",
        json=payload,
        extra_headers=headers,
    )


@router.post("/emails/analyze")
async def analyze_emails(
    request: Request,
    payload: dict[str, Any],
    session: Session = Depends(require_auth),
):
    """Analyze emails with Gemma LLM.

    Body:
        email_ids: List of Gmail message IDs
        preset: "summarize", "action_items", "sender_intent", "sentiment", "custom"
        custom_prompt: Custom analysis prompt (required for "custom" preset)
        max_tokens: Maximum tokens per response (64-2048)
        temperature: LLM temperature (0.0-1.0)
        store_results: Whether to store in RAG service
    """
    _ensure_gmail_enabled()
    proxy_request = _get_proxy_request()

    headers = _add_user_header(request, _get_service_headers())

    return await proxy_request(
        f"{GMAIL_SERVICE_URL}/emails/analyze",
        "POST",
        json=payload,
        extra_headers=headers,
    )


@router.get("/emails/analyze/stream")
async def analyze_emails_stream(
    request: Request,
    email_ids: str = Query(..., description="Comma-separated email IDs"),
    preset: str = Query("summarize"),
    custom_prompt: str | None = Query(None),
    max_tokens: int = Query(512, ge=64, le=2048),
    temperature: float = Query(0.4, ge=0.0, le=1.0),
    session: Session = Depends(require_auth),
):
    """Stream email analysis via Server-Sent Events.

    Provides real-time progress as each email is analyzed.
    """
    _ensure_gmail_enabled()

    import httpx

    headers = _add_user_header(request, _get_service_headers())

    params = {
        "email_ids": email_ids,
        "preset": preset,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if custom_prompt:
        params["custom_prompt"] = custom_prompt

    async def event_generator():
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                async with client.stream(
                    "GET",
                    f"{GMAIL_SERVICE_URL}/emails/analyze/stream",
                    params=params,
                    headers=headers,
                ) as response:
                    async for line in response.aiter_lines():
                        yield line + "\n"
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'detail': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


logger.info("Gmail Automation Router initialized")
