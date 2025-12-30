"""
GPU Lock Manager
Coordinates exclusive GPU access between Transcription (owner) and Gemma (requester)
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from enum import Enum
from typing import Any

try:
    import redis.asyncio as redis
except ImportError:
    import redis

logger = logging.getLogger(__name__)


class GPUState(Enum):
    """GPU ownership states"""

    TRANSCRIPTION_ACTIVE = "transcription"  # Default: Transcription owns GPU
    PAUSING = "pausing"  # Waiting for transcription to finish chunk
    GEMMA_EXCLUSIVE = "gemma"  # Gemma has exclusive GPU access


class GPULockManager:
    """
    Manages GPU ownership with pause/resume coordination

    Strategy:
    - Transcription owns GPU by default (runs 24/7)
    - Gemma requests GPU when needed (~50x/day)
    - Only ONE service uses GPU at a time
    """

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        """
        Initialize GPU lock manager

        Args:
            redis_url: Redis connection URL
        """
        self.redis_url = redis_url
        self.redis_client: redis.Redis | None = None
        self.pubsub = None
        self.state = GPUState.TRANSCRIPTION_ACTIVE
        self.current_gemma_task: dict[str, Any] | None = None
        self.pause_ack_timeout = float(os.getenv("GPU_PAUSE_TIMEOUT", "10.0"))

        logger.info("GPU Lock Manager initialized")

    async def connect(self):
        """Connect to Redis with authentication support"""
        # Phase 4 Security: Support Redis password authentication
        password = None

        # Try reading from secret file first (Docker secrets)
        password_file = os.getenv("REDIS_PASSWORD_FILE")
        if password_file and os.path.exists(password_file):
            with open(password_file) as f:
                password = f.read().strip()
            logger.info("Redis password loaded from secret file")
        else:
            # Fallback to environment variable
            password = os.getenv("REDIS_PASSWORD")

        # Build connection with auth if password is provided
        if password:
            # Parse URL and add password
            if "@" not in self.redis_url:
                # Insert password into URL: redis://host:port -> redis://:password@host:port
                if "://" in self.redis_url:
                    scheme, rest = self.redis_url.split("://", 1)
                    self.redis_url = f"{scheme}://:{password}@{rest}"

        self.redis_client = redis.from_url(self.redis_url, encoding="utf-8", decode_responses=True)
        self.pubsub = self.redis_client.pubsub()

        # Subscribe to transcription acknowledgments
        await self.pubsub.subscribe("transcription_paused", "vram_freed")

        # Set initial state
        await self.redis_client.set("gpu_state", self.state.value)
        await self.redis_client.publish("gpu_owner", "transcription")

        logger.info("Connected to Redis (auth=%s), GPU state: TRANSCRIPTION_ACTIVE", "yes" if password else "no")

    async def disconnect(self):
        """Disconnect from Redis"""
        if self.pubsub:
            await self.pubsub.unsubscribe("transcription_paused", "vram_freed")
            await self.pubsub.close()
        if self.redis_client:
            await self.redis_client.close()

    async def request_gpu_for_gemma(self, task_id: str, task_data: dict[str, Any]) -> bool:
        """
        Request GPU for Gemma execution

        Process:
        1. Signal transcription to pause
        2. Wait for acknowledgment (default timeout 10s)
        3. Grant GPU to Gemma

        Args:
            task_id: Unique task identifier
            task_data: Task payload

        Returns:
            True if GPU acquired successfully
        """
        logger.info(f"[GPU] Gemma task {task_id} requesting GPU (immediate priority)")

        # Store current task
        self.current_gemma_task = {"task_id": task_id, "data": task_data, "requested_at": time.time()}

        # Change state to PAUSING
        self.state = GPUState.PAUSING
        await self.redis_client.set("gpu_state", self.state.value)

        # Signal transcription to pause
        pause_message = json.dumps({"reason": "gemma", "task_id": task_id, "timestamp": datetime.now().isoformat()})

        await self.redis_client.publish("transcription_pause", pause_message)
        logger.info(f"[GPU] Pause signal sent to transcription for task {task_id}")

        # Wait for transcription to acknowledge pause (max 2s timeout)
        paused = await self._wait_for_transcription_pause(timeout=self.pause_ack_timeout)

        if not paused:
            logger.warning(f"[GPU] Transcription didn't pause in time for task {task_id}, forcing GPU takeover")
        else:
            logger.info(f"[GPU] Transcription paused successfully for task {task_id}")

        # CRITICAL: Wait for VRAM to be actually available before granting
        # Longer timeout (60s) since model offload can take time
        vram_ready = await self._wait_for_vram_available(min_free_mb=2500, timeout=60.0)

        if not vram_ready:
            logger.error("[GPU] VRAM not available after 60s - ABORTING GPU grant to prevent OOM")
            self.state = GPUState.TRANSCRIPTION_ACTIVE
            await self.redis_client.set("gpu_state", self.state.value)
            return False  # Don't grant GPU if no VRAM
        else:
            logger.info(f"[GPU] ✅ VRAM verified available for task {task_id}")

        # Grant GPU to Gemma
        self.state = GPUState.GEMMA_EXCLUSIVE
        await self.redis_client.set("gpu_state", self.state.value)
        await self.redis_client.publish("gpu_owner", "gemma")

        # Store task start time for metrics
        await self.redis_client.set(f"gemma_task:{task_id}:start", time.time())

        logger.info(f"[GPU] Gemma has exclusive GPU access for task {task_id}")
        return True

    async def release_gpu_from_gemma(self, task_id: str, result: dict[str, Any] | None = None):
        """
        Release GPU after Gemma completes

        Process:
        1. Update state to TRANSCRIPTION_ACTIVE
        2. Signal transcription to resume
        3. Clear current task

        Args:
            task_id: Task identifier
            result: Optional task result for logging
        """
        logger.info(f"[GPU] Gemma task {task_id} releasing GPU")

        # Calculate task duration
        start_time = await self.redis_client.get(f"gemma_task:{task_id}:start")
        if start_time:
            duration = time.time() - float(start_time)
            logger.info(f"[GPU] Gemma task {task_id} duration: {duration:.2f}s")
            await self.redis_client.delete(f"gemma_task:{task_id}:start")

        # Return GPU to transcription
        self.state = GPUState.TRANSCRIPTION_ACTIVE
        await self.redis_client.set("gpu_state", self.state.value)

        # Signal transcription to resume
        resume_message = json.dumps({"task_id": task_id, "timestamp": datetime.now().isoformat()})
        await self.redis_client.publish("transcription_resume", resume_message)

        # Clear current task
        self.current_gemma_task = None

        logger.info(f"[GPU] GPU returned to transcription after task {task_id}")

    async def _wait_for_transcription_pause(self, timeout: float = 2.0) -> bool:
        """
        Wait for transcription pause acknowledgment

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if transcription paused, False if timeout
        """
        start_time = time.time()

        try:
            while (time.time() - start_time) < timeout:
                # Check for pause acknowledgment
                message = await asyncio.wait_for(self.pubsub.get_message(ignore_subscribe_messages=True), timeout=0.1)

                if message and message["type"] == "message":
                    if message["channel"] == "transcription_paused":
                        data = json.loads(message["data"])
                        logger.info(f"[GPU] Transcription pause acknowledged: {data}")
                        return True

                await asyncio.sleep(0.05)  # Small delay between checks

            return False

        except TimeoutError:
            return False

    async def _wait_for_vram_available(self, min_free_mb: int = 2500, timeout: float = 60.0) -> bool:
        """
        Wait for VRAM to be available before granting GPU to Gemma

        Listens for 'vram_freed' Redis signal from transcription service.
        This approach works without needing nvidia-smi access in the container.

        Args:
            min_free_mb: Unused (kept for API compatibility)
            timeout: Maximum time to wait in seconds

        Returns:
            True if VRAM freed signal received, False if timeout
        """
        start_time = time.time()
        request_timestamp = datetime.now()

        logger.info(f"[GPU] Waiting for VRAM freed signal (timeout: {timeout}s)...")

        # CRITICAL FIX: Flush any stale messages from pubsub buffer before waiting
        # This prevents race conditions where old vram_freed signals are processed
        flushed_count = 0
        while True:
            try:
                stale_msg = await asyncio.wait_for(
                    self.pubsub.get_message(ignore_subscribe_messages=True), 
                    timeout=0.05
                )
                if stale_msg and stale_msg["type"] == "message":
                    flushed_count += 1
                    logger.debug(f"[GPU] Flushed stale pubsub message: {stale_msg['channel']}")
                else:
                    break  # No more messages
            except asyncio.TimeoutError:
                break  # Buffer is empty
        
        if flushed_count > 0:
            logger.info(f"[GPU] Flushed {flushed_count} stale pubsub messages before waiting")

        while (time.time() - start_time) < timeout:
            try:
                # Check for vram_freed message
                message = await asyncio.wait_for(
                    self.pubsub.get_message(ignore_subscribe_messages=True), 
                    timeout=0.5
                )

                if message and message["type"] == "message":
                    if message["channel"] == "vram_freed":
                        data = json.loads(message["data"])
                        
                        # Validate timestamp is recent (within 30 seconds of our request)
                        msg_timestamp_str = data.get("timestamp", "")
                        try:
                            msg_timestamp = datetime.fromisoformat(msg_timestamp_str)
                            age_seconds = (request_timestamp - msg_timestamp).total_seconds()
                            if age_seconds > 30:
                                logger.warning(f"[GPU] Ignoring stale vram_freed signal (age: {age_seconds:.1f}s)")
                                continue  # Skip stale message, keep waiting
                        except (ValueError, TypeError):
                            pass  # No valid timestamp, accept the message
                        
                        logger.info(f"[GPU] VRAM freed signal received: {data}")
                        return True

            except asyncio.TimeoutError:
                elapsed = time.time() - start_time
                if int(elapsed) % 10 == 0 and elapsed > 1:
                    logger.info(f"[GPU] Still waiting for VRAM freed signal... [{elapsed:.0f}s]")

            await asyncio.sleep(0.1)

        logger.warning(f"[GPU] VRAM freed signal not received after {timeout}s")
        return False

    async def get_status(self) -> dict[str, Any]:
        """
        Get current GPU lock status

        Returns:
            Status dictionary
        """
        status = {
            "state": self.state.value,
            "current_owner": "gemma" if self.state == GPUState.GEMMA_EXCLUSIVE else "transcription",
            "current_task": self.current_gemma_task.get("task_id") if self.current_gemma_task else None,
            "timestamp": datetime.now().isoformat(),
        }

        return status

    async def force_reset(self):
        """
        Force reset GPU to transcription (emergency recovery)
        Use with caution!
        """
        logger.warning("[GPU] FORCE RESET: Returning GPU to transcription")

        self.state = GPUState.TRANSCRIPTION_ACTIVE
        await self.redis_client.set("gpu_state", self.state.value)
        await self.redis_client.publish("transcription_resume", json.dumps({"force": True}))
        self.current_gemma_task = None

        logger.info("[GPU] Force reset complete")


# Singleton instance
_lock_manager: GPULockManager | None = None


def get_lock_manager() -> GPULockManager:
    """Get or create GPU lock manager singleton"""
    global _lock_manager
    if _lock_manager is None:
        _lock_manager = GPULockManager()
    return _lock_manager
