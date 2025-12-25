import asyncio
import logging
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

# Note: Authentication for streaming can be added later if needed
# from src.auth import get_current_user  # Commented out - would need absolute import

logger = logging.getLogger(__name__)

router = APIRouter()


class ConnectionManager:
    """
    Manages WebSocket connections for real-time Salesforce updates.
    """

    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket client connected. Total clients: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"WebSocket client disconnected. Total clients: {len(self.active_connections)}")

    async def broadcast(self, message: dict[str, Any]):
        """Broadcasts a JSON message to all connected clients."""
        if not self.active_connections:
            return

        disconnected_clients = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                disconnected_clients.append(connection)

        # Clean up disconnected clients
        for connection in disconnected_clients:
            self.disconnect(connection)


manager = ConnectionManager()


@router.websocket("/stream")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for streaming Salesforce CDC events.
    Clients connect to this endpoint to receive real-time updates.
    """
    # Note: Authentication for WebSockets is complex; for MVP we might skip strict auth
    # or implement token in query param. For now, trusting the connection or
    # relying on gateway protections if applicable.
    # Ideally: await get_current_user(token=websocket.query_params.get("token"))

    await manager.connect(websocket)
    try:
        while True:
            # Keep the connection alive
            # We can basically just wait for messages (heartbeats) or do nothing
            # If the client sends data, we might just ignore it for this one-way stream
            data = await websocket.receive_text()
            # Respond to ping/heartbeat if needed
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
        manager.disconnect(websocket)


async def broadcast_salesforce_event(event_type: str, data: dict[str, Any]):
    """
    Public utility to broadcast salesforce events from other parts of the system.
    """
    message = {"type": event_type, "data": data, "timestamp": asyncio.get_event_loop().time()}
    await manager.broadcast(message)
