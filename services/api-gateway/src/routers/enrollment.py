"""
Speaker Enrollment Router - Voice profile enrollment endpoints.

Provides speaker enrollment for voice recognition and profile management.
"""

import logging
import os
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

try:
    from src.auth.permissions import Session, require_auth
except ImportError:

    def require_auth():
        return None

    Session = None

logger = logging.getLogger(__name__)

router = APIRouter(tags=["enrollment"])

TRANSCRIPTION_URL = os.getenv("TRANSCRIPTION_URL", "http://transcription-service:8003")

# Enrollment directory
INSTANCE_DIR = Path(os.getenv("INSTANCE_DIR", "/app/instance"))
ENROLLMENT_DIR = INSTANCE_DIR / "enrollment"


def _get_proxy_request():
    """Lazy import to avoid circular dependencies."""
    from src.main import proxy_request

    return proxy_request


# =============================================================================
# Speaker Enrollment Endpoints
# =============================================================================


@router.post("/enroll/upload")
async def enroll_upload(
    audio: UploadFile = File(...), speaker: str = Form(...), session: Session = Depends(require_auth)
):
    """Upload enrollment audio for speaker voice profile.

    Creates a speaker embedding from 90-120 seconds of voice audio.
    Saves to instance/enrollment/{speaker_name}/
    """
    proxy_request = _get_proxy_request()

    if not speaker or len(speaker.strip()) < 2:
        raise HTTPException(status_code=400, detail="Speaker name must be at least 2 characters")

    # Sanitize speaker name
    safe_speaker = "".join(c for c in speaker if c.isalnum() or c in "-_").strip()
    if not safe_speaker:
        raise HTTPException(status_code=400, detail="Invalid speaker name")

    # Read audio content
    audio_content = await audio.read()

    # Validate file size (min 1MB for ~90 seconds, max 50MB)
    if len(audio_content) < 100 * 1024:  # 100KB minimum
        raise HTTPException(status_code=400, detail="Audio too short (minimum ~30 seconds required)")
    if len(audio_content) > 50 * 1024 * 1024:  # 50MB max
        raise HTTPException(status_code=400, detail="Audio too large (max 50MB)")

    # Forward to transcription service for embedding creation
    files = {"audio": (audio.filename, audio_content, audio.content_type or "audio/wav")}
    data = {"speaker": safe_speaker}

    try:
        result = await proxy_request(
            f"{TRANSCRIPTION_URL}/enroll/upload",
            "POST",
            files=files,
            json=data,
        )

        return {
            "success": True,
            "speaker": safe_speaker,
            "message": f"Enrollment submitted for speaker '{safe_speaker}'",
            "result": result,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Enrollment failed for speaker {safe_speaker}: {e}")
        raise HTTPException(status_code=500, detail=f"Enrollment failed: {str(e)}")


@router.get("/enroll/speakers")
async def list_enrolled_speakers(session: Session = Depends(require_auth)):
    """List all enrolled speakers."""
    proxy_request = _get_proxy_request()

    try:
        result = await proxy_request(f"{TRANSCRIPTION_URL}/enroll/speakers", "GET")
        return result
    except HTTPException as e:
        if e.status_code == 404:
            # Fallback: check local enrollment directory
            speakers = []
            if ENROLLMENT_DIR.exists():
                for path in ENROLLMENT_DIR.iterdir():
                    if path.is_dir():
                        speakers.append(
                            {
                                "name": path.name,
                                "files": len(list(path.glob("*.wav"))),
                            }
                        )
            return {"success": True, "speakers": speakers}
        raise


# Alias endpoints with /api prefix for frontend compatibility
@router.post("/api/enroll/upload")
async def api_enroll_upload(
    audio: UploadFile = File(...), speaker: str = Form(...), session: Session = Depends(require_auth)
):
    """Alias for /enroll/upload to support /api prefix."""
    return await enroll_upload(audio, speaker, session)


@router.get("/api/enroll/speakers")
async def api_list_enrolled_speakers(session: Session = Depends(require_auth)):
    """Alias for /enroll/speakers to support /api prefix."""
    return await list_enrolled_speakers(session)


logger.info("✅ Enrollment Router initialized with speaker profile endpoints")
