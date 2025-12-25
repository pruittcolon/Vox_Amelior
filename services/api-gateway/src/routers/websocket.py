"""
WebSocket Router - Real-time streaming proxy endpoints.

Provides WebSocket proxy to transcription service for real-time audio streaming.
"""

import asyncio
import logging
import os

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

try:
    from src.auth.manager import AuthManager
    from src.auth.permissions import Session
except ImportError:
    Session = None
    AuthManager = None

logger = logging.getLogger(__name__)

router = APIRouter(tags=["websocket"])

TRANSCRIPTION_URL = os.getenv("TRANSCRIPTION_URL", "http://transcription-service:8003")
TRANSCRIPTION_WS_URL = TRANSCRIPTION_URL.replace("http://", "ws://").replace("https://", "wss://")

# Allowed WebSocket origins (configurable via env)
WS_ALLOWED_ORIGINS = set(
    filter(
        None,
        os.getenv("WS_ALLOWED_ORIGINS", "http://localhost,http://127.0.0.1,https://localhost,https://127.0.0.1").split(
            ","
        ),
    )
)


def _get_auth_manager():
    """Lazy import auth manager."""
    try:
        from src.main import auth_manager

        return auth_manager
    except ImportError:
        return None


# =============================================================================
# WebSocket Streaming Proxy
# =============================================================================


@router.websocket("/stream")
async def websocket_stream_proxy(websocket: WebSocket, token: str | None = Query(None)):
    """WebSocket proxy to transcription service for real-time streaming.

    Mobile app connects to ws://gateway:8000/stream?token=<session_token>
    Gateway proxies to ws://transcription-service:8003/stream

    Security:
    - JWT authentication required (via query param or cookie)
    - Origin validation for browser clients
    """

    # Validate origin for browser clients
    origin = websocket.headers.get("origin", "")
    if origin and origin not in WS_ALLOWED_ORIGINS:
        # Check if it's a subdomain of allowed origins
        allowed = any(
            origin.endswith(ao.replace("http://", ".").replace("https://", ".")) for ao in WS_ALLOWED_ORIGINS if ao
        )
        if not allowed:
            logger.warning(f"WebSocket connection rejected - invalid origin: {origin}")
            await websocket.close(code=4003, reason="Invalid origin")
            return

    # Authenticate
    auth_manager = _get_auth_manager()
    ws_token = token or websocket.cookies.get("nemo_session")

    if not ws_token:
        logger.warning("WebSocket connection rejected - no token provided")
        await websocket.close(code=4001, reason="Authentication required")
        return

    if auth_manager:
        session = auth_manager.validate_session(ws_token)
        if not session:
            logger.warning("WebSocket connection rejected - invalid session")
            await websocket.close(code=4001, reason="Invalid session")
            return
        user_id = getattr(session, "user_id", "anonymous")
    else:
        user_id = "anonymous"
        logger.warning("Auth manager not available, allowing anonymous WebSocket")

    await websocket.accept()
    logger.info(f"WebSocket connected: user={user_id}")

    # Connect to backend transcription service
    backend_url = f"{TRANSCRIPTION_WS_URL}/stream"

    try:
        import websockets

        async with websockets.connect(backend_url) as backend_ws:

            async def forward_to_backend():
                """Forward messages from client to backend."""
                try:
                    while True:
                        data = await websocket.receive_bytes()
                        await backend_ws.send(data)
                except WebSocketDisconnect:
                    pass
                except Exception as e:
                    logger.debug(f"Client->Backend forward ended: {e}")

            async def forward_to_client():
                """Forward messages from backend to client."""
                try:
                    async for message in backend_ws:
                        if isinstance(message, bytes):
                            await websocket.send_bytes(message)
                        else:
                            await websocket.send_text(message)
                except Exception as e:
                    logger.debug(f"Backend->Client forward ended: {e}")

            # Run both directions concurrently
            await asyncio.gather(forward_to_backend(), forward_to_client(), return_exceptions=True)

    except ImportError:
        # Fallback without websockets library
        logger.warning("websockets library not available, cannot proxy to backend")
        await websocket.send_json({"error": "Backend streaming not available"})
    except Exception as e:
        logger.error(f"WebSocket proxy error: {e}")
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
        logger.info(f"WebSocket disconnected: user={user_id}")


logger.info("✅ WebSocket Router initialized with streaming proxy")
