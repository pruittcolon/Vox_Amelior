"""
Unit Tests for GPU State Module

Tests for shared.gpu.state module including:
- GPUOwner enum values
- GPUState transitions
- Invalid transition handling
- Serialization/deserialization

Author: Enterprise Analytics Team
"""

import pytest
from datetime import datetime

from shared.gpu.state import (
    GPUOwner,
    GPUState,
    StateTransitionError,
    VALID_TRANSITIONS,
)


class TestGPUOwner:
    """Tests for GPUOwner enum."""
    
    def test_enum_values(self) -> None:
        """Verify all expected enum values exist."""
        assert GPUOwner.TRANSCRIPTION.value == "transcription"
        assert GPUOwner.GEMMA.value == "gemma"
        assert GPUOwner.ML_SERVICE.value == "ml_service"
        assert GPUOwner.ACQUIRING.value == "acquiring"
        assert GPUOwner.RELEASING.value == "releasing"
    
    def test_enum_from_string(self) -> None:
        """Verify enum can be constructed from string."""
        assert GPUOwner("transcription") == GPUOwner.TRANSCRIPTION
        assert GPUOwner("gemma") == GPUOwner.GEMMA


class TestGPUState:
    """Tests for GPUState dataclass."""
    
    def test_default_state(self) -> None:
        """Verify default state is TRANSCRIPTION."""
        state = GPUState()
        assert state.owner == GPUOwner.TRANSCRIPTION
        assert state.session_id is None
        assert state.requester is None
        assert state.acquired_at is None
        assert state.metadata == {}
    
    def test_custom_state(self) -> None:
        """Verify custom state creation."""
        now = datetime.now()
        state = GPUState(
            owner=GPUOwner.GEMMA,
            session_id="session-123",
            requester="gemma-service",
            acquired_at=now,
            metadata={"test": True},
        )
        assert state.owner == GPUOwner.GEMMA
        assert state.session_id == "session-123"
        assert state.requester == "gemma-service"
        assert state.acquired_at == now
        assert state.metadata == {"test": True}


class TestStateTransitions:
    """Tests for state transition validation."""
    
    def test_valid_transition_transcription_to_acquiring(self) -> None:
        """TRANSCRIPTION -> ACQUIRING is valid."""
        state = GPUState(owner=GPUOwner.TRANSCRIPTION)
        assert state.can_transition_to(GPUOwner.ACQUIRING) is True
        
        new_state = state.transition_to(GPUOwner.ACQUIRING, session_id="test-123")
        assert new_state.owner == GPUOwner.ACQUIRING
        assert new_state.session_id == "test-123"
    
    def test_valid_transition_acquiring_to_gemma(self) -> None:
        """ACQUIRING -> GEMMA is valid."""
        state = GPUState(owner=GPUOwner.ACQUIRING, session_id="test-123")
        assert state.can_transition_to(GPUOwner.GEMMA) is True
        
        new_state = state.transition_to(GPUOwner.GEMMA)
        assert new_state.owner == GPUOwner.GEMMA
        assert new_state.acquired_at is not None  # Should set timestamp
    
    def test_valid_transition_gemma_to_releasing(self) -> None:
        """GEMMA -> RELEASING is valid."""
        state = GPUState(owner=GPUOwner.GEMMA, session_id="test-123")
        assert state.can_transition_to(GPUOwner.RELEASING) is True
        
        new_state = state.transition_to(GPUOwner.RELEASING)
        assert new_state.owner == GPUOwner.RELEASING
    
    def test_valid_transition_releasing_to_transcription(self) -> None:
        """RELEASING -> TRANSCRIPTION is valid."""
        state = GPUState(owner=GPUOwner.RELEASING, session_id="test-123")
        assert state.can_transition_to(GPUOwner.TRANSCRIPTION) is True
        
        new_state = state.transition_to(GPUOwner.TRANSCRIPTION)
        assert new_state.owner == GPUOwner.TRANSCRIPTION
        assert new_state.acquired_at is None  # Should clear timestamp
    
    def test_force_reset_always_allowed(self) -> None:
        """Force reset to TRANSCRIPTION is always valid."""
        for initial_owner in GPUOwner:
            state = GPUState(owner=initial_owner)
            assert state.can_transition_to(GPUOwner.TRANSCRIPTION) is True
    
    def test_invalid_transition_transcription_to_gemma(self) -> None:
        """TRANSCRIPTION -> GEMMA directly is invalid."""
        state = GPUState(owner=GPUOwner.TRANSCRIPTION)
        assert state.can_transition_to(GPUOwner.GEMMA) is False
        
        with pytest.raises(StateTransitionError) as exc_info:
            state.transition_to(GPUOwner.GEMMA)
        
        assert exc_info.value.current_state == GPUOwner.TRANSCRIPTION
        assert exc_info.value.target_state == GPUOwner.GEMMA
    
    def test_invalid_transition_gemma_to_acquiring(self) -> None:
        """GEMMA -> ACQUIRING is invalid."""
        state = GPUState(owner=GPUOwner.GEMMA)
        assert state.can_transition_to(GPUOwner.ACQUIRING) is False
        
        with pytest.raises(StateTransitionError):
            state.transition_to(GPUOwner.ACQUIRING)


class TestSerialization:
    """Tests for state serialization/deserialization."""
    
    def test_to_dict(self) -> None:
        """Verify to_dict produces correct dictionary."""
        now = datetime.now()
        state = GPUState(
            owner=GPUOwner.GEMMA,
            session_id="session-123",
            requester="gemma-service",
            acquired_at=now,
            metadata={"key": "value"},
        )
        
        data = state.to_dict()
        
        assert data["owner"] == "gemma"
        assert data["session_id"] == "session-123"
        assert data["requester"] == "gemma-service"
        assert data["acquired_at"] == now.isoformat()
        assert data["metadata"] == {"key": "value"}
    
    def test_from_dict(self) -> None:
        """Verify from_dict reconstructs state correctly."""
        now = datetime.now()
        data = {
            "owner": "gemma",
            "session_id": "session-123",
            "requester": "gemma-service",
            "acquired_at": now.isoformat(),
            "metadata": {"key": "value"},
        }
        
        state = GPUState.from_dict(data)
        
        assert state.owner == GPUOwner.GEMMA
        assert state.session_id == "session-123"
        assert state.requester == "gemma-service"
        assert state.acquired_at is not None
        assert state.metadata == {"key": "value"}
    
    def test_roundtrip(self) -> None:
        """Verify to_dict -> from_dict preserves data."""
        original = GPUState(
            owner=GPUOwner.ML_SERVICE,
            session_id="ml-task-456",
            requester="ml-service",
            acquired_at=datetime.now(),
            metadata={"engine": "titan"},
        )
        
        data = original.to_dict()
        restored = GPUState.from_dict(data)
        
        assert restored.owner == original.owner
        assert restored.session_id == original.session_id
        assert restored.requester == original.requester
        assert restored.metadata == original.metadata


class TestStateTransitionError:
    """Tests for StateTransitionError exception."""
    
    def test_error_message(self) -> None:
        """Verify error message format."""
        error = StateTransitionError(
            GPUOwner.TRANSCRIPTION,
            GPUOwner.GEMMA,
            "must go through ACQUIRING first",
        )
        
        message = str(error)
        assert "transcription" in message
        assert "gemma" in message
        assert "must go through ACQUIRING first" in message
    
    def test_error_without_reason(self) -> None:
        """Verify error works without reason."""
        error = StateTransitionError(
            GPUOwner.GEMMA,
            GPUOwner.ACQUIRING,
        )
        
        message = str(error)
        assert "gemma" in message
        assert "acquiring" in message


class TestValidTransitionsMap:
    """Tests for VALID_TRANSITIONS constant."""
    
    def test_all_states_have_transitions(self) -> None:
        """Verify all states have defined transitions."""
        for state in GPUOwner:
            assert state in VALID_TRANSITIONS, f"Missing transitions for {state}"
    
    def test_no_self_transitions(self) -> None:
        """Verify no state transitions to itself (except via TRANSCRIPTION reset)."""
        for state, targets in VALID_TRANSITIONS.items():
            if state != GPUOwner.TRANSCRIPTION:
                # TRANSCRIPTION can reset to itself
                assert state not in targets, f"{state} should not transition to itself"
