"""
Pause Manager for Transcription Service
Handles GPU pause/resume signals from coordinator
"""

import asyncio
import json
import logging
from collections.abc import Callable
from datetime import datetime
from typing import Any

try:
    import redis.asyncio as redis
except ImportError:
    import redis

logger = logging.getLogger(__name__)


class PauseManager:
    """
    Manages pause/resume for transcription service

    Listens for Redis pub/sub signals from GPU coordinator:
    - transcription_pause: Pause and queue new requests
    - transcription_resume: Resume and process queued requests
    """

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        """
        Initialize pause manager

        Args:
            redis_url: Redis connection URL
        """
        self.redis_url = redis_url
        self.redis_client: redis.Redis | None = None
        self.pubsub = None
        self.paused = False
        self.chunk_queue: list[dict[str, Any]] = []
        self.current_processing = False
        self.pause_callback: Callable | None = None
        self.resume_callback: Callable | None = None

        logger.info("Pause Manager initialized")

    async def connect(self):
        """Connect to Redis and subscribe to pause/resume signals"""
        import os

        # Phase 4 Security: Support Redis password authentication
        password = None
        password_file = os.getenv("REDIS_PASSWORD_FILE")
        if password_file and os.path.exists(password_file):
            with open(password_file) as f:
                password = f.read().strip()
            logger.info("Redis password loaded from secret file")
        else:
            password = os.getenv("REDIS_PASSWORD")

        # Build URL with password if provided
        redis_url = self.redis_url
        if password and "@" not in redis_url and "://" in redis_url:
            scheme, rest = redis_url.split("://", 1)
            redis_url = f"{scheme}://:{password}@{rest}"

        self.redis_client = redis.from_url(redis_url, encoding="utf-8", decode_responses=True)
        self.pubsub = self.redis_client.pubsub()

        # Subscribe to pause/resume channels
        await self.pubsub.subscribe("transcription_pause", "transcription_resume")

        logger.info("Connected to Redis (auth=%s), listening for pause/resume signals", "yes" if password else "no")

        # Start background listener
        asyncio.create_task(self._listen_for_signals())

    async def disconnect(self):
        """Disconnect from Redis"""
        if self.pubsub:
            await self.pubsub.unsubscribe("transcription_pause", "transcription_resume")
            await self.pubsub.close()
        if self.redis_client:
            await self.redis_client.close()

    async def _listen_for_signals(self):
        """Background task to listen for pause/resume signals"""
        try:
            while True:
                message = await self.pubsub.get_message(ignore_subscribe_messages=True)

                if message and message["type"] == "message":
                    channel = message["channel"]
                    data = json.loads(message["data"]) if message["data"] else {}

                    if channel == "transcription_pause":
                        await self._handle_pause_signal(data)
                    elif channel == "transcription_resume":
                        await self._handle_resume_signal(data)

                await asyncio.sleep(0.01)  # Small delay to prevent tight loop

        except asyncio.CancelledError:
            logger.info("Signal listener cancelled")
        except Exception as e:
            logger.error(f"Error in signal listener: {e}")

    async def _handle_pause_signal(self, data: dict[str, Any]):
        """
        Handle pause signal from coordinator

        Process:
        1. Wait for current chunk to complete
        2. Set paused flag
        3. Acknowledge to coordinator
        """
        logger.info(f"Pause signal received: {data}")

        # Wait for current processing to complete (if any)
        if self.current_processing:
            logger.info("Waiting for current chunk to complete...")
            # Give it up to 2 seconds to complete
            for _ in range(20):
                if not self.current_processing:
                    break
                await asyncio.sleep(0.1)

        # Set paused flag
        self.paused = True

        # Call pause callback if set
        if self.pause_callback:
            try:
                await self.pause_callback()
            except Exception as e:
                logger.error(f"Error in pause callback: {e}")

        # Acknowledge pause to coordinator
        await self.redis_client.publish(
            "transcription_paused",
            json.dumps(
                {"status": "paused", "timestamp": datetime.now().isoformat(), "queued_chunks": len(self.chunk_queue)}
            ),
        )

        logger.info(f"Transcription PAUSED - queuing new chunks (queue size: {len(self.chunk_queue)})")

    async def _handle_resume_signal(self, data: dict[str, Any]):
        """
        Handle resume signal from coordinator

        Process:
        1. Clear paused flag
        2. Process queued chunks
        """
        logger.info(f"Resume signal received: {data}")

        self.paused = False

        # Call resume callback if set
        if self.resume_callback:
            try:
                await self.resume_callback()
            except Exception as e:
                logger.error(f"Error in resume callback: {e}")

        logger.info(f"Transcription RESUMED - processing {len(self.chunk_queue)} queued chunks")

    def set_pause_callback(self, callback: Callable):
        """Set callback to be called when pause signal received"""
        self.pause_callback = callback

    def set_resume_callback(self, callback: Callable):
        """Set callback to be called when resume signal received"""
        self.resume_callback = callback

    def is_paused(self) -> bool:
        """Check if currently paused"""
        return self.paused

    def add_to_queue(self, chunk_data: dict[str, Any]):
        """
        Add chunk to queue during pause

        Args:
            chunk_data: Chunk data to queue
        """
        self.chunk_queue.append(chunk_data)
        logger.info(f"Added chunk to queue (queue size: {len(self.chunk_queue)})")

    def get_queued_chunks(self) -> list[dict[str, Any]]:
        """
        Get all queued chunks and clear queue

        Returns:
            List of queued chunks
        """
        chunks = self.chunk_queue.copy()
        self.chunk_queue.clear()
        return chunks

    def set_processing(self, processing: bool):
        """
        Set current processing status

        Args:
            processing: True if currently processing a chunk
        """
        self.current_processing = processing

    def get_status(self) -> dict[str, Any]:
        """
        Get current pause manager status

        Returns:
            Status dictionary
        """
        return {
            "paused": self.paused,
            "current_processing": self.current_processing,
            "queued_chunks": len(self.chunk_queue),
            "timestamp": datetime.now().isoformat(),
        }


# Singleton instance
_pause_manager: PauseManager | None = None


def get_pause_manager() -> PauseManager:
    """Get or create pause manager singleton"""
    global _pause_manager
    if _pause_manager is None:
        _pause_manager = PauseManager()
    return _pause_manager
