"""
VoiceMonkey Command Queue

Manages a queue of commands to VoiceMonkey with rate limiting.
Ensures commands are spaced at least COOLDOWN_SECONDS apart to prevent
VoiceMonkey rate limit issues.
"""

import asyncio
import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)

# Minimum seconds between VoiceMonkey commands
COOLDOWN_SECONDS = 10.0


@dataclass
class QueuedCommand:
    """A command waiting in the queue"""

    device: str
    preset: str
    queued_at: datetime


class CommandQueue:
    """
    Thread-safe command queue that ensures VoiceMonkey commands
    are spaced at least COOLDOWN_SECONDS apart.

    Usage:
        queue = get_command_queue()
        queue.set_voice_monkey_client(voice_monkey_client)

        # First command executes immediately
        result = await queue.enqueue("kitchenlights", "kitchenlights")

        # Commands within cooldown are queued
        result = await queue.enqueue("livingroomlight", "livingroomlight")
        # result["queued"] == True, result["queue_position"] == 1
    """

    def __init__(self, cooldown_seconds: float = COOLDOWN_SECONDS):
        self._queue: deque[QueuedCommand] = deque()
        self._last_command_time: datetime | None = None
        self._cooldown = timedelta(seconds=cooldown_seconds)
        self._lock = asyncio.Lock()
        self._processing = False
        self._voice_monkey_client = None

    def set_voice_monkey_client(self, client) -> None:
        """Set the VoiceMonkey client for executing commands"""
        self._voice_monkey_client = client
        logger.info("📋 CommandQueue: VoiceMonkey client configured")

    async def enqueue(self, device: str, preset: str) -> dict[str, Any]:
        """
        Add a command to the queue.

        If no cooldown is active, execute immediately.
        Otherwise, queue for later execution.

        Args:
            device: VoiceMonkey device name
            preset: Preset/action name (usually same as device)

        Returns:
            Dict with execution status:
            - queued: bool - True if command was queued
            - executed_immediately: bool - True if sent right away
            - queue_position: int - Position in queue (if queued)
            - estimated_wait_seconds: float - Estimated wait time (if queued)
            - result: dict - VoiceMonkey response (if executed)
        """
        async with self._lock:
            now = datetime.now()

            # Check if we can send immediately (no cooldown or cooldown expired)
            if self._last_command_time is None or now - self._last_command_time >= self._cooldown:
                # Send immediately
                self._last_command_time = now

        # Execute outside lock to not block queue
        if self._last_command_time == now:
            result = await self._execute(device, preset)

            # Start processing queue in background if commands are waiting
            async with self._lock:
                if not self._processing and self._queue:
                    asyncio.create_task(self._process_queue())

            return {"queued": False, "executed_immediately": True, "result": result}

        # Add to queue
        async with self._lock:
            cmd = QueuedCommand(device=device, preset=preset, queued_at=now)
            self._queue.append(cmd)

            time_since_last = now - self._last_command_time if self._last_command_time else timedelta(0)
            remaining_cooldown = max(0, (self._cooldown - time_since_last).total_seconds())
            position = len(self._queue)

            # Estimated wait = remaining cooldown + (position-1) * full cooldown
            estimated_wait = remaining_cooldown + (position - 1) * self._cooldown.total_seconds()

            logger.info(f"📋 Command queued: {device} (position {position}, ~{estimated_wait:.1f}s wait)")

            # Start queue processor if not running
            if not self._processing:
                asyncio.create_task(self._process_queue())

            return {
                "queued": True,
                "executed_immediately": False,
                "queue_position": position,
                "estimated_wait_seconds": estimated_wait,
            }

    async def _process_queue(self) -> None:
        """Background task to process queued commands with proper spacing"""
        self._processing = True
        logger.info("📋 Queue processor started")

        try:
            while True:
                # Check if queue is empty
                async with self._lock:
                    if not self._queue:
                        logger.info("📋 Queue empty, processor stopping")
                        break

                    # Calculate wait time
                    now = datetime.now()
                    if self._last_command_time:
                        time_since_last = now - self._last_command_time
                        if time_since_last < self._cooldown:
                            wait_time = (self._cooldown - time_since_last).total_seconds()
                        else:
                            wait_time = 0
                    else:
                        wait_time = 0

                # Wait for cooldown (outside lock)
                if wait_time > 0:
                    logger.info(f"⏳ Queue: Waiting {wait_time:.1f}s before next command")
                    await asyncio.sleep(wait_time)

                # Pop and execute next command
                async with self._lock:
                    if not self._queue:
                        break

                    cmd = self._queue.popleft()
                    self._last_command_time = datetime.now()
                    remaining = len(self._queue)

                logger.info(f"📋 Queue: Executing {cmd.device} ({remaining} remaining)")
                await self._execute(cmd.device, cmd.preset)

        except Exception as e:
            logger.error(f"📋 Queue processor error: {e}")
        finally:
            self._processing = False
            logger.info("📋 Queue processor stopped")

    async def _execute(self, device: str, preset: str) -> dict[str, Any]:
        """Execute command via VoiceMonkey client"""
        if not self._voice_monkey_client:
            logger.error("📋 VoiceMonkey client not configured!")
            return {"success": False, "error": "VoiceMonkey client not configured"}

        logger.info(f"🔊 Queue: Sending to VoiceMonkey -> {device}")
        try:
            return await self._voice_monkey_client.trigger_preset(preset)
        except Exception as e:
            logger.error(f"🔊 Queue: VoiceMonkey error: {e}")
            return {"success": False, "error": str(e)}

    def get_status(self) -> dict[str, Any]:
        """Get current queue status"""
        now = datetime.now()

        # Calculate time until next command can be sent
        if self._last_command_time:
            time_since_last = now - self._last_command_time
            cooldown_remaining = max(0, (self._cooldown - time_since_last).total_seconds())
        else:
            cooldown_remaining = 0

        return {
            "queue_length": len(self._queue),
            "processing": self._processing,
            "cooldown_seconds": self._cooldown.total_seconds(),
            "cooldown_remaining": round(cooldown_remaining, 1),
            "last_command_time": self._last_command_time.isoformat() if self._last_command_time else None,
            "queued_devices": [cmd.device for cmd in self._queue],
        }


# Singleton instance
_queue: CommandQueue | None = None


def get_command_queue() -> CommandQueue:
    """Get the global command queue singleton"""
    global _queue
    if _queue is None:
        _queue = CommandQueue()
    return _queue
