"""
Transcription Router - Audio transcription and emotion endpoints.

Provides file transcription, streaming transcription, and emotion analysis.
"""

import logging
import os
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile

try:
    from src.auth.permissions import Session, require_auth
except ImportError:

    def require_auth():
        return None

    Session = None

logger = logging.getLogger(__name__)

router = APIRouter(tags=["transcription"])

TRANSCRIPTION_URL = os.getenv("TRANSCRIPTION_URL", "http://transcription-service:8003")


def _get_proxy_request():
    """Lazy import to avoid circular dependencies."""
    from src.main import proxy_request

    return proxy_request


# =============================================================================
# Emotion Analysis
# =============================================================================


@router.post("/api/emotion/analyze")
async def emotion_analyze(request: dict[str, Any], session: Session = Depends(require_auth)):
    """Analyze emotion from audio or text."""
    proxy_request = _get_proxy_request()
    return await proxy_request(f"{TRANSCRIPTION_URL}/emotion/analyze", "POST", json=request)


# =============================================================================
# Transcription Endpoints
# =============================================================================


@router.post("/api/transcription/transcribe")
async def transcription_transcribe(file: UploadFile = File(...), session: Session = Depends(require_auth)):
    """Transcribe an audio file."""
    proxy_request = _get_proxy_request()

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    # Read file content
    file_content = await file.read()

    # Validate file size (max 100MB)
    if len(file_content) > 100 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 100MB)")

    files = {"file": (file.filename, file_content, file.content_type or "audio/wav")}
    return await proxy_request(f"{TRANSCRIPTION_URL}/transcribe", "POST", files=files)


@router.post("/api/transcription/stream")
async def transcription_stream(request: dict[str, Any], session: Session = Depends(require_auth)):
    """Start streaming transcription session."""
    proxy_request = _get_proxy_request()
    return await proxy_request(f"{TRANSCRIPTION_URL}/stream", "POST", json=request)


# =============================================================================
# Direct Transcribe Endpoints (Flutter app compatibility)
# =============================================================================


@router.post("/transcribe")
async def transcribe_direct(request: Request, session: Session = Depends(require_auth)):
    """Direct transcribe endpoint - forwards to transcription service (flexible input)."""
    proxy_request = _get_proxy_request()

    content_type = request.headers.get("content-type", "")

    if "multipart" in content_type:
        # Handle file upload
        form = await request.form()
        file = form.get("file") or form.get("audio")
        if file and hasattr(file, "read"):
            file_content = await file.read()
            files = {"file": (file.filename, file_content, file.content_type or "audio/wav")}
            return await proxy_request(f"{TRANSCRIPTION_URL}/transcribe", "POST", files=files)
        else:
            raise HTTPException(status_code=400, detail="No file in form data")
    elif "application/json" in content_type:
        # Handle JSON body (base64 audio or config)
        body = await request.json()
        return await proxy_request(f"{TRANSCRIPTION_URL}/transcribe", "POST", json=body)
    else:
        # Assume raw audio bytes
        body = await request.body()
        if not body:
            raise HTTPException(status_code=400, detail="No audio data provided")
        files = {"file": ("audio.wav", body, "audio/wav")}
        return await proxy_request(f"{TRANSCRIPTION_URL}/transcribe", "POST", files=files)


@router.post("/api/transcribe")
async def api_transcribe(request: Request, session: Session = Depends(require_auth)):
    """Alias route for frontend clients that prefix endpoints with /api."""
    return await transcribe_direct(request, session)


logger.info("✅ Transcription Router initialized with audio endpoints")
