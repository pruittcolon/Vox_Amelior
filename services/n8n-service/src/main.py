"""
n8n Integration Service - Main Application

FastAPI service for processing transcript segments and triggering n8n workflows.
"""

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .command_queue import get_command_queue
from .command_registry import get_command_registry
from .config import LOG_LEVEL, SERVICE_PORT
from .emotion_tracker import get_emotion_tracker
from .voice_monkey import get_voice_monkey_client
from .webhook_client import get_webhook_client

# Configure logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# Pydantic Models
# ============================================================================


class TranscriptSegment(BaseModel):
    """A single transcript segment from the transcription service"""

    text: str
    speaker: str = "unknown"
    verified_speaker: str | None = None
    start_time: float = 0.0
    end_time: float = 0.0
    emotion: str | None = None
    emotion_confidence: float | None = None

    class Config:
        extra = "allow"  # Allow additional fields


class ProcessRequest(BaseModel):
    """Request to process transcript segments"""

    segments: list[TranscriptSegment]
    job_id: str | None = None
    session_id: str | None = None


class ProcessResponse(BaseModel):
    """Response from processing segments"""

    processed_segments: int
    voice_commands_triggered: int
    emotion_alerts_triggered: int
    details: list[dict[str, Any]] = Field(default_factory=list)


class RegisterCommandRequest(BaseModel):
    """Request to register a new voice command"""

    command_id: str
    pattern: str
    description: str
    n8n_action: str
    enabled: bool = True


class CommandResponse(BaseModel):
    """Response for command operations"""

    command_id: str
    pattern: str
    description: str
    n8n_action: str
    enabled: bool


# ============================================================================
# Application Lifespan
# ============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup and shutdown"""
    logger.info("Starting n8n Integration Service...")

    # Initialize singletons
    registry = get_command_registry()
    tracker = get_emotion_tracker()
    client = get_webhook_client()
    voice_monkey = get_voice_monkey_client()

    # Initialize command queue with VoiceMonkey client
    queue = get_command_queue()
    queue.set_voice_monkey_client(voice_monkey)

    logger.info(f"✅ Loaded {len(registry.list_all())} voice commands")
    logger.info(f"✅ Tracking emotions for speakers: {tracker.tracked_speakers}")
    logger.info(f"✅ Webhook base URL: {client.base_url}")
    logger.info(f"✅ Voice Monkey configured: {voice_monkey.config.is_configured()}")
    logger.info(f"✅ Command queue: {queue.get_status()['cooldown_seconds']}s cooldown")

    yield

    logger.info("Shutting down n8n Integration Service...")


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="n8n Integration Service",
    description="Process transcript segments and trigger n8n workflows",
    version="1.0.0",
    lifespan=lifespan,
)

# =============================================================================
# ISO 27002 5.17 / OWASP API2:2023: Service-to-Service Authentication
# =============================================================================
import os
import sys

# Add shared module path for imports
sys.path.insert(0, "/app/shared")

SECURE_MODE = os.getenv("SECURE_MODE", "false").lower() in {"1", "true", "yes"}

if SECURE_MODE:
    try:
        from shared.security.service_auth import ServiceAuthMiddleware, load_service_jwt_keys
        from shared.security.startup_checks import assert_secure_mode

        # Fail-closed: Block startup if secure mode requirements not met
        assert_secure_mode()

        jwt_keys = load_service_jwt_keys("n8n-service")
        app.add_middleware(
            ServiceAuthMiddleware,
            service_secret=jwt_keys,
            exempt_paths=["/health", "/docs", "/openapi.json"],
            enabled=True,
        )
        logger.info("✅ S2S authentication enabled for n8n-service (SECURE_MODE=true)")
    except Exception as e:
        logger.critical("🚫 Failed to initialize S2S auth in SECURE_MODE: %s", e)
        raise RuntimeError(f"SECURITY: S2S auth required but failed: {e}") from e
else:
    logger.warning("⚠️ S2S authentication DISABLED (SECURE_MODE=false)")


# ============================================================================
# Endpoints
# ============================================================================


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    client = get_webhook_client()
    voice_monkey = get_voice_monkey_client()
    n8n_reachable = await client.health_check()

    return {
        "status": "healthy",
        "n8n_reachable": n8n_reachable,
        "voice_monkey_configured": voice_monkey.config.is_configured(),
        "commands_loaded": len(get_command_registry().list_all()),
        "tracking_speakers": get_emotion_tracker().tracked_speakers,
        "emotion_model": "j-hartmann/emotion-english-distilroberta-base",
    }


@app.get("/queue/status")
async def get_queue_status():
    """
    Get the current status of the VoiceMonkey command queue.

    Returns:
        - queue_length: Number of commands waiting
        - processing: Whether queue is actively processing
        - cooldown_seconds: Delay between commands (10s)
        - cooldown_remaining: Seconds until next command can be sent
        - queued_devices: List of devices waiting in queue
    """
    queue = get_command_queue()
    return queue.get_status()


@app.post("/process", response_model=ProcessResponse)
async def process_segments(request: ProcessRequest):
    """
    Process transcript segments for voice commands and emotion alerts.

    This endpoint is called by the transcription service after enriching segments.
    """
    registry = get_command_registry()
    tracker = get_emotion_tracker()
    client = get_webhook_client()

    voice_commands_triggered = 0
    emotion_alerts_triggered = 0
    details = []

    for segment in request.segments:
        # Get effective speaker (prefer verified)
        speaker = segment.verified_speaker or segment.speaker

        # Check for voice commands
        matched_command = registry.match(segment.text)
        if matched_command:
            # Send to n8n webhook
            webhook_success = await client.send_voice_command(
                command=matched_command,
                speaker=speaker,
                text=segment.text,
                segment_metadata={
                    "job_id": request.job_id,
                    "session_id": request.session_id,
                    "start_time": segment.start_time,
                    "end_time": segment.end_time,
                },
            )

            # Queue command for VoiceMonkey (handles rate limiting)
            # Commands are spaced 10 seconds apart to prevent VoiceMonkey issues
            queue = get_command_queue()
            vm_result = await queue.enqueue(device=matched_command.n8n_action, preset=matched_command.n8n_action)

            voice_commands_triggered += 1
            details.append(
                {
                    "type": "voice_command",
                    "command_id": matched_command.command_id,
                    "speaker": speaker,
                    "text": segment.text[:50],
                    "voice_monkey_queued": vm_result.get("queued", False),
                    "voice_monkey_executed": vm_result.get("executed_immediately", False),
                    "queue_position": vm_result.get("queue_position"),
                    "webhook_sent": webhook_success,
                }
            )

        # Track emotions
        if segment.emotion:
            alert = tracker.add_emotion(speaker, segment.emotion)
            if alert:
                success = await client.send_emotion_alert(
                    alert=alert, segment_metadata={"job_id": request.job_id, "session_id": request.session_id}
                )
                if success:
                    emotion_alerts_triggered += 1
                    details.append(
                        {
                            "type": "emotion_alert",
                            "speaker": alert.speaker,
                            "emotion": alert.emotion,
                            "count": alert.consecutive_count,
                        }
                    )

    return ProcessResponse(
        processed_segments=len(request.segments),
        voice_commands_triggered=voice_commands_triggered,
        emotion_alerts_triggered=emotion_alerts_triggered,
        details=details,
    )


# ============================================================================
# Command Registry Endpoints
# ============================================================================


@app.get("/commands", response_model=list[CommandResponse])
async def list_commands():
    """List all registered voice commands"""
    registry = get_command_registry()
    return [
        CommandResponse(
            command_id=cmd.command_id,
            pattern=cmd.pattern,
            description=cmd.description,
            n8n_action=cmd.n8n_action,
            enabled=cmd.enabled,
        )
        for cmd in registry.list_all()
    ]


@app.post("/commands", response_model=CommandResponse)
async def register_command(request: RegisterCommandRequest):
    """Register a new voice command"""
    registry = get_command_registry()

    try:
        command = registry.register(
            command_id=request.command_id,
            pattern=request.pattern,
            description=request.description,
            n8n_action=request.n8n_action,
            enabled=request.enabled,
        )
        return CommandResponse(
            command_id=command.command_id,
            pattern=command.pattern,
            description=command.description,
            n8n_action=command.n8n_action,
            enabled=command.enabled,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/commands/{command_id}")
async def delete_command(command_id: str):
    """Delete a voice command"""
    registry = get_command_registry()

    if registry.unregister(command_id):
        return {"message": f"Command '{command_id}' deleted"}
    else:
        raise HTTPException(status_code=404, detail=f"Command '{command_id}' not found")


@app.patch("/commands/{command_id}/enable")
async def enable_command(command_id: str, enabled: bool = True):
    """Enable or disable a voice command"""
    registry = get_command_registry()

    if registry.set_enabled(command_id, enabled):
        return {"message": f"Command '{command_id}' {'enabled' if enabled else 'disabled'}"}
    else:
        raise HTTPException(status_code=404, detail=f"Command '{command_id}' not found")


# ============================================================================
# Emotion Tracking Endpoints
# ============================================================================


@app.get("/alerts/status")
async def get_alert_status():
    """Get current emotion tracking status"""
    tracker = get_emotion_tracker()
    return tracker.get_all_status()


@app.get("/alerts/history")
async def get_alert_history(limit: int = 50):
    """Get emotion alert history"""
    tracker = get_emotion_tracker()
    return tracker.get_alert_history(limit)


@app.post("/alerts/reset/{speaker}")
async def reset_speaker_tracking(speaker: str):
    """Reset emotion tracking for a specific speaker"""
    tracker = get_emotion_tracker()

    if tracker.reset_speaker(speaker):
        return {"message": f"Reset tracking for '{speaker}'"}
    else:
        return {"message": f"No tracking state found for '{speaker}'"}


@app.post("/alerts/reset")
async def reset_all_tracking():
    """Reset all emotion tracking"""
    tracker = get_emotion_tracker()
    tracker.reset_all()
    return {"message": "Reset all emotion tracking"}


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=SERVICE_PORT)
