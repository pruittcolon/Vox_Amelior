"""
GPU State Management

Defines GPU ownership states, state dataclass, and transition validation.
Ensures atomic and valid state transitions for the GPU coordinator.

Author: Enterprise Analytics Team
Version: 1.0.0
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class GPUOwner(Enum):
    """
    GPU ownership states.
    
    State machine transitions:
    - TRANSCRIPTION -> ACQUIRING (when Gemma requests GPU)
    - ACQUIRING -> GEMMA (when pause acknowledged)
    - GEMMA -> RELEASING (when session ends)
    - RELEASING -> TRANSCRIPTION (when resume acknowledged)
    - Any state -> TRANSCRIPTION (on force reset)
    """
    
    TRANSCRIPTION = "transcription"  # Default: Transcription owns GPU (24/7 background)
    GEMMA = "gemma"                  # Gemma owns GPU (during user session)
    ML_SERVICE = "ml_service"        # ML service owns GPU (for training/inference)
    ACQUIRING = "acquiring"          # Transitional: waiting for pause ACK
    RELEASING = "releasing"          # Transitional: waiting for resume ACK


class StateTransitionError(Exception):
    """Raised when an invalid state transition is attempted."""
    
    def __init__(
        self, 
        current_state: GPUOwner, 
        target_state: GPUOwner, 
        reason: str = ""
    ) -> None:
        """
        Initialize state transition error.
        
        Args:
            current_state: Current GPU owner
            target_state: Attempted target state
            reason: Human-readable explanation
        """
        self.current_state = current_state
        self.target_state = target_state
        self.reason = reason
        message = (
            f"Invalid GPU state transition: {current_state.value} -> {target_state.value}"
        )
        if reason:
            message += f" ({reason})"
        super().__init__(message)


# Valid state transitions
VALID_TRANSITIONS: dict[GPUOwner, set[GPUOwner]] = {
    GPUOwner.TRANSCRIPTION: {GPUOwner.ACQUIRING},
    GPUOwner.ACQUIRING: {GPUOwner.GEMMA, GPUOwner.ML_SERVICE, GPUOwner.TRANSCRIPTION},
    GPUOwner.GEMMA: {GPUOwner.RELEASING},
    GPUOwner.ML_SERVICE: {GPUOwner.RELEASING},
    GPUOwner.RELEASING: {GPUOwner.TRANSCRIPTION},
}


@dataclass
class GPUState:
    """
    Current GPU state with metadata.
    
    Thread-safe state container for GPU ownership tracking.
    """
    
    owner: GPUOwner = GPUOwner.TRANSCRIPTION
    session_id: str | None = None
    requester: str | None = None  # Service that requested GPU
    acquired_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def can_transition_to(self, target: GPUOwner) -> bool:
        """
        Check if transition to target state is valid.
        
        Args:
            target: Target GPU owner state
            
        Returns:
            True if transition is valid
        """
        # Force reset always allowed
        if target == GPUOwner.TRANSCRIPTION:
            return True
        
        valid_targets = VALID_TRANSITIONS.get(self.owner, set())
        return target in valid_targets
    
    def transition_to(
        self, 
        target: GPUOwner, 
        session_id: str | None = None,
        requester: str | None = None,
        metadata: dict[str, Any] | None = None
    ) -> "GPUState":
        """
        Transition to a new state.
        
        Args:
            target: Target GPU owner state
            session_id: Optional session identifier
            requester: Service requesting the transition
            metadata: Optional metadata to store
            
        Returns:
            New GPUState instance
            
        Raises:
            StateTransitionError: If transition is invalid
        """
        if not self.can_transition_to(target):
            raise StateTransitionError(
                self.owner, 
                target,
                f"Not a valid transition from {self.owner.value}"
            )
        
        # Determine acquired_at timestamp
        if target in (GPUOwner.GEMMA, GPUOwner.ML_SERVICE):
            acquired_at = datetime.now()
        elif target == GPUOwner.TRANSCRIPTION:
            acquired_at = None  # Transcription doesn't track session
        else:
            acquired_at = self.acquired_at  # Keep existing during transitional states
        
        return GPUState(
            owner=target,
            session_id=session_id or self.session_id,
            requester=requester or self.requester,
            acquired_at=acquired_at,
            metadata=metadata or self.metadata,
        )
    
    def to_dict(self) -> dict[str, Any]:
        """
        Serialize state to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "owner": self.owner.value,
            "session_id": self.session_id,
            "requester": self.requester,
            "acquired_at": self.acquired_at.isoformat() if self.acquired_at else None,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GPUState":
        """
        Deserialize state from dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            GPUState instance
        """
        acquired_at = None
        if data.get("acquired_at"):
            acquired_at = datetime.fromisoformat(data["acquired_at"])
        
        return cls(
            owner=GPUOwner(data.get("owner", "transcription")),
            session_id=data.get("session_id"),
            requester=data.get("requester"),
            acquired_at=acquired_at,
            metadata=data.get("metadata", {}),
        )
    
    def __repr__(self) -> str:
        """String representation for logging."""
        parts = [f"GPUState(owner={self.owner.value}"]
        if self.session_id:
            parts.append(f", session={self.session_id[:8]}...")
        if self.requester:
            parts.append(f", requester={self.requester}")
        parts.append(")")
        return "".join(parts)
