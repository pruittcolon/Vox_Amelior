"""
GPU Coordination Protocol

Pydantic models for GPU acquire/release requests and Redis pub/sub messages.
Provides type-safe serialization for coordinator communication.

Author: Enterprise Analytics Team
Version: 1.0.0
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class GPUPriority(Enum):
    """Priority levels for GPU requests."""
    
    IMMEDIATE = 1  # User-facing: chat, predictions (preempts background)
    BACKGROUND = 2  # Batch jobs: training, large synthesis


class GPUCommandType(Enum):
    """Types of commands sent by coordinator to services."""
    
    PAUSE = "pause"    # Request service to pause and release GPU
    RESUME = "resume"  # Signal service to resume GPU operations


class GPUAckType(Enum):
    """Types of acknowledgments sent by services to coordinator."""
    
    PAUSED = "paused"   # Service has paused and released VRAM
    RESUMED = "resumed" # Service has resumed on GPU
    ERROR = "error"     # Service encountered an error


# ============================================================================
# HTTP Request/Response Models (Coordinator API)
# ============================================================================


class GPUAcquireRequest(BaseModel):
    """Request to acquire GPU from coordinator."""
    
    session_id: str = Field(..., description="Unique session identifier")
    requester: str = Field(..., description="Service requesting GPU (gemma, ml-service)")
    priority: GPUPriority = Field(
        default=GPUPriority.IMMEDIATE,
        description="Request priority level"
    )
    timeout_ms: int = Field(
        default=30000,
        ge=1000,
        le=120000,
        description="Timeout in milliseconds (1-120 seconds)"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional metadata for logging/debugging"
    )


class GPUAcquireResponse(BaseModel):
    """Response from GPU acquire request."""
    
    success: bool = Field(..., description="Whether GPU was acquired")
    session_id: str = Field(..., description="Session identifier")
    acquired_at: datetime | None = Field(
        default=None,
        description="Timestamp when GPU was acquired"
    )
    error: str | None = Field(
        default=None,
        description="Error message if acquisition failed"
    )
    wait_time_ms: float = Field(
        default=0.0,
        description="Time spent waiting for GPU (milliseconds)"
    )


class GPUReleaseRequest(BaseModel):
    """Request to release GPU back to coordinator."""
    
    session_id: str = Field(..., description="Session identifier to release")
    requester: str = Field(..., description="Service releasing GPU")
    result: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional result/status from the session"
    )


class GPUReleaseResponse(BaseModel):
    """Response from GPU release request."""
    
    success: bool = Field(..., description="Whether GPU was released")
    session_id: str = Field(..., description="Session identifier")
    released_at: datetime | None = Field(
        default=None,
        description="Timestamp when GPU was released"
    )
    session_duration_ms: float = Field(
        default=0.0,
        description="Total session duration (milliseconds)"
    )


class GPUStatusResponse(BaseModel):
    """Current GPU coordinator status."""
    
    owner: str = Field(..., description="Current GPU owner service")
    session_id: str | None = Field(default=None, description="Active session ID")
    requester: str | None = Field(default=None, description="Service using GPU")
    acquired_at: datetime | None = Field(default=None, description="Session start time")
    state: str = Field(..., description="Current state machine state")
    redis_connected: bool = Field(default=True, description="Redis connection status")
    postgres_connected: bool = Field(default=True, description="Postgres connection status")


# ============================================================================
# Redis Pub/Sub Message Models
# ============================================================================


class GPUCommand(BaseModel):
    """
    Command message sent by coordinator to services via Redis pub/sub.
    
    Channel: gpu:command
    """
    
    command: GPUCommandType = Field(..., description="Command type")
    session_id: str = Field(..., description="Associated session ID")
    requester: str = Field(..., description="Service that caused this command")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Command timestamp for ordering"
    )
    timeout_ms: int = Field(
        default=10000,
        description="Expected response timeout"
    )
    
    def to_redis(self) -> str:
        """Serialize to Redis-compatible JSON string."""
        return self.model_dump_json()
    
    @classmethod
    def from_redis(cls, data: str) -> "GPUCommand":
        """Deserialize from Redis JSON string."""
        return cls.model_validate_json(data)


class GPUAck(BaseModel):
    """
    Acknowledgment message sent by services to coordinator via Redis pub/sub.
    
    Channel: gpu:ack:{service_name}
    """
    
    ack_type: GPUAckType = Field(..., description="Acknowledgment type")
    service: str = Field(..., description="Service sending acknowledgment")
    session_id: str = Field(..., description="Associated session ID")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Ack timestamp"
    )
    error: str | None = Field(
        default=None,
        description="Error message if ack_type is ERROR"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional metadata (e.g., vram freed, model unloaded)"
    )
    
    def to_redis(self) -> str:
        """Serialize to Redis-compatible JSON string."""
        return self.model_dump_json()
    
    @classmethod
    def from_redis(cls, data: str) -> "GPUAck":
        """Deserialize from Redis JSON string."""
        return cls.model_validate_json(data)


# Redis channel constants
REDIS_CHANNEL_COMMAND = "gpu:command"
REDIS_CHANNEL_ACK_PREFIX = "gpu:ack:"  # Append service name

# Legacy channel names for backwards compatibility during migration
REDIS_CHANNEL_PAUSE = "transcription_pause"
REDIS_CHANNEL_PAUSED = "transcription_paused"
REDIS_CHANNEL_RESUME = "transcription_resume"
REDIS_CHANNEL_VRAM_FREED = "vram_freed"
