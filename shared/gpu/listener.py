"""
GPU Command Listener

Redis pub/sub listener for services that need to respond to GPU commands
(pause/resume) from the coordinator. Used by Transcription service.

Author: Enterprise Analytics Team
Version: 1.0.0
"""

import asyncio
import json
import logging
import os
from collections.abc import Callable
from datetime import datetime
from typing import Any

try:
    import redis.asyncio as redis
except ImportError:
    import redis  # type: ignore

from shared.gpu.protocol import (
    GPUAck,
    GPUAckType,
    GPUCommand,
    GPUCommandType,
    REDIS_CHANNEL_COMMAND,
    REDIS_CHANNEL_ACK_PREFIX,
    # Legacy channels for backwards compatibility
    REDIS_CHANNEL_PAUSE,
    REDIS_CHANNEL_PAUSED,
    REDIS_CHANNEL_RESUME,
    REDIS_CHANNEL_VRAM_FREED,
)

logger = logging.getLogger(__name__)


class GPUCommandListener:
    """
    Redis pub/sub listener for GPU commands.
    
    Listens for pause/resume commands from the GPU coordinator and
    invokes registered callbacks. Sends acknowledgments back.
    
    Usage:
        listener = GPUCommandListener(service_name="transcription")
        listener.on_pause(my_pause_handler)
        listener.on_resume(my_resume_handler)
        await listener.start()
        
        # When done:
        await listener.stop()
    
    Callbacks:
        pause_callback: async def(session_id: str, requester: str) -> None
        resume_callback: async def(session_id: str) -> None
    """
    
    def __init__(
        self,
        service_name: str,
        redis_url: str | None = None,
    ) -> None:
        """
        Initialize GPU command listener.
        
        Args:
            service_name: Name of this service (for ACK channel)
            redis_url: Redis URL (default: from REDIS_URL env)
        """
        self.service_name = service_name
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://redis:6379")
        
        self.redis_client: redis.Redis | None = None
        self.pubsub: Any = None
        
        self._pause_callback: Callable | None = None
        self._resume_callback: Callable | None = None
        self._running = False
        self._listener_task: asyncio.Task | None = None
        
        # State tracking
        self.paused = False
        self.current_processing = False
        
        logger.info(f"[GPU-LISTENER] Initialized for service: {service_name}")
    
    async def connect(self) -> None:
        """Connect to Redis with authentication support."""
        # Load password from secret file or env
        password = None
        password_file = os.getenv("REDIS_PASSWORD_FILE")
        if password_file and os.path.exists(password_file):
            with open(password_file) as f:
                password = f.read().strip()
            logger.info("[GPU-LISTENER] Redis password loaded from secret file")
        else:
            password = os.getenv("REDIS_PASSWORD")
        
        # Build URL with password
        redis_url = self.redis_url
        if password and "@" not in redis_url and "://" in redis_url:
            scheme, rest = redis_url.split("://", 1)
            redis_url = f"{scheme}://:{password}@{rest}"
        
        self.redis_client = redis.from_url(
            redis_url,
            encoding="utf-8",
            decode_responses=True,
        )
        self.pubsub = self.redis_client.pubsub()
        
        # Subscribe to both new and legacy channels for compatibility
        channels = [
            REDIS_CHANNEL_COMMAND,
            REDIS_CHANNEL_PAUSE,
            REDIS_CHANNEL_RESUME,
        ]
        await self.pubsub.subscribe(*channels)
        
        logger.info(
            f"[GPU-LISTENER] Connected to Redis (auth={'yes' if password else 'no'}), "
            f"subscribed to: {channels}"
        )
    
    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self.pubsub:
            await self.pubsub.unsubscribe()
            await self.pubsub.close()
        if self.redis_client:
            await self.redis_client.close()
        logger.info("[GPU-LISTENER] Disconnected from Redis")
    
    def on_pause(self, callback: Callable) -> None:
        """
        Register pause callback.
        
        The callback should:
        1. Finish current processing
        2. Offload model from GPU to CPU
        3. Return when VRAM is freed
        
        Args:
            callback: async def(session_id: str, requester: str) -> None
        """
        self._pause_callback = callback
        logger.debug("[GPU-LISTENER] Pause callback registered")
    
    def on_resume(self, callback: Callable) -> None:
        """
        Register resume callback.
        
        The callback should:
        1. Load model back to GPU
        2. Resume processing queued work
        
        Args:
            callback: async def(session_id: str) -> None
        """
        self._resume_callback = callback
        logger.debug("[GPU-LISTENER] Resume callback registered")
    
    async def start(self) -> None:
        """Start listening for GPU commands."""
        if self._running:
            logger.warning("[GPU-LISTENER] Already running")
            return
        
        await self.connect()
        self._running = True
        self._listener_task = asyncio.create_task(self._listen_loop())
        logger.info("[GPU-LISTENER] Started listening for GPU commands")
    
    async def stop(self) -> None:
        """Stop listening and cleanup."""
        self._running = False
        if self._listener_task:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass
        await self.disconnect()
        logger.info("[GPU-LISTENER] Stopped")
    
    async def _listen_loop(self) -> None:
        """Background task to listen for messages."""
        try:
            while self._running:
                try:
                    message = await self.pubsub.get_message(
                        ignore_subscribe_messages=True
                    )
                    
                    if message and message["type"] == "message":
                        channel = message["channel"]
                        data = message["data"]
                        
                        await self._handle_message(channel, data)
                    
                    await asyncio.sleep(0.01)  # Prevent tight loop
                    
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.error(f"[GPU-LISTENER] Error processing message: {e}")
                    await asyncio.sleep(0.1)
                    
        except asyncio.CancelledError:
            logger.info("[GPU-LISTENER] Listener cancelled")
    
    async def _handle_message(self, channel: str, data: str) -> None:
        """Route message to appropriate handler."""
        try:
            if channel == REDIS_CHANNEL_COMMAND:
                # New protocol
                command = GPUCommand.from_redis(data)
                if command.command == GPUCommandType.PAUSE:
                    await self._handle_pause(command.session_id, command.requester)
                elif command.command == GPUCommandType.RESUME:
                    await self._handle_resume(command.session_id)
                    
            elif channel == REDIS_CHANNEL_PAUSE:
                # Legacy protocol
                parsed = json.loads(data)
                session_id = parsed.get("task_id", "unknown")
                requester = parsed.get("reason", "gemma")
                await self._handle_pause(session_id, requester)
                
            elif channel == REDIS_CHANNEL_RESUME:
                # Legacy protocol
                parsed = json.loads(data)
                session_id = parsed.get("task_id", "unknown")
                await self._handle_resume(session_id)
                
        except Exception as e:
            logger.error(f"[GPU-LISTENER] Failed to handle message: {e}")
    
    async def _handle_pause(self, session_id: str, requester: str) -> None:
        """Handle pause command."""
        logger.info(f"[GPU-LISTENER] PAUSE received: session={session_id}, requester={requester}")
        
        # Wait for current processing to complete
        if self.current_processing:
            logger.info("[GPU-LISTENER] Waiting for current processing to complete...")
            for _ in range(20):  # Max 2 seconds
                if not self.current_processing:
                    break
                await asyncio.sleep(0.1)
        
        self.paused = True
        
        # Send immediate ACK (before model offload)
        await self._send_ack(
            GPUAckType.PAUSED,
            session_id,
            metadata={"offloading": True},
        )
        
        # Call pause callback (model offload)
        if self._pause_callback:
            try:
                await self._pause_callback(session_id, requester)
                logger.info("[GPU-LISTENER] Pause callback completed")
                
                # Send VRAM freed signal (legacy protocol compatibility)
                await self._send_vram_freed(session_id)
                
            except Exception as e:
                logger.error(f"[GPU-LISTENER] Pause callback error: {e}")
                await self._send_ack(
                    GPUAckType.ERROR,
                    session_id,
                    error=str(e),
                )
        
        logger.info(f"[GPU-LISTENER] PAUSED: session={session_id}")
    
    async def _handle_resume(self, session_id: str) -> None:
        """Handle resume command."""
        logger.info(f"[GPU-LISTENER] RESUME received: session={session_id}")
        
        self.paused = False
        
        # Call resume callback
        if self._resume_callback:
            try:
                await self._resume_callback(session_id)
                logger.info("[GPU-LISTENER] Resume callback completed")
            except Exception as e:
                logger.error(f"[GPU-LISTENER] Resume callback error: {e}")
        
        # Send ACK
        await self._send_ack(GPUAckType.RESUMED, session_id)
        
        logger.info(f"[GPU-LISTENER] RESUMED: session={session_id}")
    
    async def _send_ack(
        self,
        ack_type: GPUAckType,
        session_id: str,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Send acknowledgment to coordinator."""
        if not self.redis_client:
            logger.warning("[GPU-LISTENER] Cannot send ACK: not connected")
            return
        
        ack = GPUAck(
            ack_type=ack_type,
            service=self.service_name,
            session_id=session_id,
            error=error,
            metadata=metadata or {},
        )
        
        # Send to new channel
        channel = f"{REDIS_CHANNEL_ACK_PREFIX}{self.service_name}"
        await self.redis_client.publish(channel, ack.to_redis())
        
        # Also send to legacy channel for compatibility
        if ack_type == GPUAckType.PAUSED:
            legacy_data = json.dumps({
                "status": "paused",
                "offloading": metadata.get("offloading") if metadata else False,
                "timestamp": datetime.now().isoformat(),
            })
            await self.redis_client.publish(REDIS_CHANNEL_PAUSED, legacy_data)
        
        logger.debug(f"[GPU-LISTENER] ACK sent: type={ack_type.value}, session={session_id}")
    
    async def _send_vram_freed(self, session_id: str) -> None:
        """Send VRAM freed signal (legacy protocol)."""
        if not self.redis_client:
            return
        
        data = json.dumps({
            "status": "freed",
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
        })
        await self.redis_client.publish(REDIS_CHANNEL_VRAM_FREED, data)
        logger.info("[GPU-LISTENER] VRAM freed signal sent")
    
    def is_paused(self) -> bool:
        """Check if currently paused."""
        return self.paused
    
    def set_processing(self, processing: bool) -> None:
        """Set current processing status."""
        self.current_processing = processing
    
    def get_status(self) -> dict[str, Any]:
        """Get current listener status."""
        return {
            "service": self.service_name,
            "paused": self.paused,
            "processing": self.current_processing,
            "running": self._running,
            "timestamp": datetime.now().isoformat(),
        }


# Singleton pattern for service use
_listener: GPUCommandListener | None = None


def get_gpu_listener(service_name: str = "transcription") -> GPUCommandListener:
    """
    Get or create GPU command listener singleton.
    
    Args:
        service_name: Service name for this listener
        
    Returns:
        GPUCommandListener instance
    """
    global _listener
    if _listener is None:
        _listener = GPUCommandListener(service_name=service_name)
    return _listener


async def init_gpu_listener(
    service_name: str,
    pause_callback: Callable | None = None,
    resume_callback: Callable | None = None,
) -> GPUCommandListener:
    """
    Initialize and start GPU command listener.
    
    Args:
        service_name: Service name
        pause_callback: Callback for pause commands
        resume_callback: Callback for resume commands
        
    Returns:
        Started GPUCommandListener
    """
    listener = get_gpu_listener(service_name)
    
    if pause_callback:
        listener.on_pause(pause_callback)
    if resume_callback:
        listener.on_resume(resume_callback)
    
    await listener.start()
    return listener


async def shutdown_gpu_listener() -> None:
    """Shutdown GPU command listener."""
    global _listener
    if _listener:
        await _listener.stop()
        _listener = None
        logger.info("[GPU-LISTENER] Shutdown complete")
