"""
Key Rotation Manager.

Provides automated key rotation with zero-downtime:
- Primary/previous key model for graceful rotation
- Scheduled rotation based on policy
- Key versioning and lifecycle management
- Integration with HSM for secure key generation

Usage:
    manager = get_key_rotation_manager()
    manager.rotate_key("jwt-signing-key")
    
    # Check if rotation needed
    if manager.should_rotate("jwt-signing-key"):
        manager.rotate_key("jwt-signing-key")
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class RotationReason(Enum):
    """Reasons for key rotation."""
    
    SCHEDULED = "scheduled"       # Regular scheduled rotation
    COMPROMISE = "compromise"     # Key compromise detected
    MANUAL = "manual"             # Manual rotation request
    POLICY = "policy"             # Policy requirement
    UPGRADE = "upgrade"           # Algorithm upgrade


class KeyState(Enum):
    """Key lifecycle states."""
    
    ACTIVE = "active"             # Primary, used for new operations
    PREVIOUS = "previous"         # Previous key, valid for verification
    RETIRED = "retired"           # No longer valid
    DESTROYED = "destroyed"       # Key material destroyed


@dataclass
class RotationEvent:
    """Record of a key rotation event."""
    
    key_id: str
    old_version: int
    new_version: int
    reason: RotationReason
    rotated_at: datetime
    rotated_by: str = "system"
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "key_id": self.key_id,
            "old_version": self.old_version,
            "new_version": self.new_version,
            "reason": self.reason.value,
            "rotated_at": self.rotated_at.isoformat(),
            "rotated_by": self.rotated_by,
            "metadata": self.metadata,
        }


@dataclass
class KeyRotationPolicy:
    """Policy for key rotation.
    
    Defines when and how keys should be rotated.
    """
    
    # Rotation interval
    rotation_interval_days: int = 90  # Rotate every 90 days
    
    # Grace period for previous key
    grace_period_days: int = 7  # Previous key valid for 7 days
    
    # Maximum key age before forced rotation
    max_key_age_days: int = 365
    
    # Minimum time between rotations
    min_rotation_interval_hours: int = 1
    
    # Enable automatic rotation
    auto_rotate: bool = True
    
    # Notify before rotation
    notify_days_before: int = 7
    
    # Key-specific overrides
    key_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    
    def get_rotation_interval(self, key_id: str) -> timedelta:
        """Get rotation interval for a specific key."""
        if key_id in self.key_overrides:
            days = self.key_overrides[key_id].get(
                "rotation_interval_days",
                self.rotation_interval_days,
            )
            return timedelta(days=days)
        return timedelta(days=self.rotation_interval_days)
    
    def get_grace_period(self, key_id: str) -> timedelta:
        """Get grace period for a specific key."""
        if key_id in self.key_overrides:
            days = self.key_overrides[key_id].get(
                "grace_period_days",
                self.grace_period_days,
            )
            return timedelta(days=days)
        return timedelta(days=self.grace_period_days)


@dataclass
class KeyVersion:
    """A version of a key with its state."""
    
    key_id: str
    version: int
    state: KeyState
    created_at: datetime
    rotated_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    algorithm: str = "AES-256"
    
    def is_valid_for_verification(self) -> bool:
        """Check if key can be used for verification (not encryption)."""
        return self.state in (KeyState.ACTIVE, KeyState.PREVIOUS)
    
    def is_valid_for_encryption(self) -> bool:
        """Check if key can be used for encryption (only active)."""
        return self.state == KeyState.ACTIVE


class KeyRotationManager:
    """Manages key rotation lifecycle.
    
    Provides:
    - Zero-downtime key rotation
    - Policy-based automatic rotation
    - Key versioning
    - Rotation event logging
    
    Key Model:
    - Primary (Active): Used for new signatures/encryption
    - Previous: Valid for verification during grace period
    - Retired: No longer valid
    """
    
    def __init__(
        self,
        policy: Optional[KeyRotationPolicy] = None,
        state_file: Optional[str] = None,
    ):
        """Initialize the rotation manager.
        
        Args:
            policy: Rotation policy (uses defaults if None)
            state_file: Path to persist rotation state
        """
        self.policy = policy or KeyRotationPolicy()
        self.state_file = state_file or os.getenv(
            "KEY_ROTATION_STATE",
            "/tmp/key_rotation_state.json",
        )
        
        # Key versions: key_id -> list of KeyVersion
        self._key_versions: dict[str, list[KeyVersion]] = {}
        
        # Rotation history
        self._rotation_events: list[RotationEvent] = []
        
        # Callbacks for rotation events
        self._rotation_callbacks: list[Callable[[RotationEvent], None]] = []
        
        # Load state if exists
        self._load_state()
        
        logger.info("KeyRotationManager initialized")
    
    def _load_state(self) -> None:
        """Load rotation state from file."""
        state_path = Path(self.state_file)
        if state_path.exists():
            try:
                data = json.loads(state_path.read_text())
                
                for key_id, versions in data.get("key_versions", {}).items():
                    self._key_versions[key_id] = [
                        KeyVersion(
                            key_id=v["key_id"],
                            version=v["version"],
                            state=KeyState(v["state"]),
                            created_at=datetime.fromisoformat(v["created_at"]),
                            rotated_at=datetime.fromisoformat(v["rotated_at"]) if v.get("rotated_at") else None,
                            expires_at=datetime.fromisoformat(v["expires_at"]) if v.get("expires_at") else None,
                            algorithm=v.get("algorithm", "AES-256"),
                        )
                        for v in versions
                    ]
                
                logger.info("Loaded rotation state for %d keys", len(self._key_versions))
            except Exception as e:
                logger.warning("Failed to load rotation state: %s", e)
    
    def _save_state(self) -> None:
        """Save rotation state to file."""
        try:
            data = {
                "key_versions": {
                    key_id: [
                        {
                            "key_id": v.key_id,
                            "version": v.version,
                            "state": v.state.value,
                            "created_at": v.created_at.isoformat(),
                            "rotated_at": v.rotated_at.isoformat() if v.rotated_at else None,
                            "expires_at": v.expires_at.isoformat() if v.expires_at else None,
                            "algorithm": v.algorithm,
                        }
                        for v in versions
                    ]
                    for key_id, versions in self._key_versions.items()
                },
                "saved_at": datetime.utcnow().isoformat(),
            }
            
            Path(self.state_file).write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.error("Failed to save rotation state: %s", e)
    
    def register_key(
        self,
        key_id: str,
        algorithm: str = "AES-256",
        created_at: Optional[datetime] = None,
    ) -> KeyVersion:
        """Register an existing key for rotation management.
        
        Args:
            key_id: Key identifier
            algorithm: Key algorithm
            created_at: When key was created (default: now)
            
        Returns:
            KeyVersion for the registered key
        """
        if key_id in self._key_versions:
            logger.warning("Key %s already registered", key_id)
            return self.get_active_version(key_id)
        
        version = KeyVersion(
            key_id=key_id,
            version=1,
            state=KeyState.ACTIVE,
            created_at=created_at or datetime.utcnow(),
            algorithm=algorithm,
        )
        
        self._key_versions[key_id] = [version]
        self._save_state()
        
        logger.info("Registered key: %s (version=1)", key_id)
        return version
    
    def get_active_version(self, key_id: str) -> Optional[KeyVersion]:
        """Get the active (primary) version of a key."""
        if key_id not in self._key_versions:
            return None
        
        for v in self._key_versions[key_id]:
            if v.state == KeyState.ACTIVE:
                return v
        return None
    
    def get_previous_version(self, key_id: str) -> Optional[KeyVersion]:
        """Get the previous version of a key (for verification grace period)."""
        if key_id not in self._key_versions:
            return None
        
        for v in self._key_versions[key_id]:
            if v.state == KeyState.PREVIOUS:
                return v
        return None
    
    def get_all_valid_versions(self, key_id: str) -> list[KeyVersion]:
        """Get all versions valid for verification."""
        if key_id not in self._key_versions:
            return []
        
        return [v for v in self._key_versions[key_id] if v.is_valid_for_verification()]
    
    def should_rotate(self, key_id: str) -> tuple[bool, Optional[str]]:
        """Check if a key should be rotated.
        
        Args:
            key_id: Key to check
            
        Returns:
            Tuple of (should_rotate, reason)
        """
        active = self.get_active_version(key_id)
        if not active:
            return False, None
        
        now = datetime.utcnow()
        age = now - active.created_at
        rotation_interval = self.policy.get_rotation_interval(key_id)
        max_age = timedelta(days=self.policy.max_key_age_days)
        
        # Check if past rotation interval
        if age >= rotation_interval:
            return True, f"Age ({age.days} days) exceeds rotation interval"
        
        # Check if approaching max age
        if age >= max_age:
            return True, f"Age ({age.days} days) exceeds max age"
        
        # Check if key expires soon
        if active.expires_at and now >= active.expires_at:
            return True, "Key has expired"
        
        return False, None
    
    def rotate_key(
        self,
        key_id: str,
        reason: RotationReason = RotationReason.SCHEDULED,
        rotated_by: str = "system",
        new_key_callback: Optional[Callable[[str], None]] = None,
    ) -> RotationEvent:
        """Rotate a key.
        
        Creates new active version, demotes current active to previous,
        and retires old previous version.
        
        Args:
            key_id: Key to rotate
            reason: Reason for rotation
            rotated_by: Who initiated the rotation
            new_key_callback: Called with new key ID to generate actual key material
            
        Returns:
            RotationEvent record
        """
        now = datetime.utcnow()
        
        # Get current active version
        active = self.get_active_version(key_id)
        old_version = active.version if active else 0
        new_version = old_version + 1
        
        # Register key if not exists
        if key_id not in self._key_versions:
            self._key_versions[key_id] = []
        
        # Retire old previous version
        for v in self._key_versions[key_id]:
            if v.state == KeyState.PREVIOUS:
                v.state = KeyState.RETIRED
                logger.info("Retired key version: %s v%d", key_id, v.version)
        
        # Demote current active to previous
        if active:
            active.state = KeyState.PREVIOUS
            active.rotated_at = now
            active.expires_at = now + self.policy.get_grace_period(key_id)
            logger.info("Demoted key to previous: %s v%d", key_id, active.version)
        
        # Create new active version
        new_active = KeyVersion(
            key_id=key_id,
            version=new_version,
            state=KeyState.ACTIVE,
            created_at=now,
            algorithm=active.algorithm if active else "AES-256",
        )
        self._key_versions[key_id].append(new_active)
        
        # Generate actual key material via callback
        if new_key_callback:
            try:
                new_key_callback(f"{key_id}_v{new_version}")
            except Exception as e:
                logger.error("Failed to generate new key material: %s", e)
        
        # Create rotation event
        event = RotationEvent(
            key_id=key_id,
            old_version=old_version,
            new_version=new_version,
            reason=reason,
            rotated_at=now,
            rotated_by=rotated_by,
        )
        self._rotation_events.append(event)
        
        # Notify callbacks
        for callback in self._rotation_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error("Rotation callback failed: %s", e)
        
        # Persist state
        self._save_state()
        
        logger.info(
            "Rotated key %s: v%d -> v%d (reason=%s)",
            key_id, old_version, new_version, reason.value,
        )
        
        return event
    
    def emergency_rotate(
        self,
        key_id: str,
        rotated_by: str = "security-team",
    ) -> RotationEvent:
        """Emergency rotation due to suspected compromise.
        
        Immediately retires all old versions (no grace period).
        """
        # Retire all existing versions immediately
        if key_id in self._key_versions:
            for v in self._key_versions[key_id]:
                v.state = KeyState.RETIRED
                v.expires_at = datetime.utcnow()
        
        # Create new version
        return self.rotate_key(
            key_id,
            reason=RotationReason.COMPROMISE,
            rotated_by=rotated_by,
        )
    
    def add_rotation_callback(
        self,
        callback: Callable[[RotationEvent], None],
    ) -> None:
        """Add a callback to be notified on rotation events."""
        self._rotation_callbacks.append(callback)
    
    def get_rotation_history(
        self,
        key_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[RotationEvent]:
        """Get rotation history, optionally filtered by key."""
        events = self._rotation_events
        if key_id:
            events = [e for e in events if e.key_id == key_id]
        return events[-limit:]
    
    def cleanup_retired_keys(self, older_than_days: int = 30) -> int:
        """Remove old retired key versions from memory.
        
        Note: Does not destroy key material in HSM, just removes tracking.
        """
        cutoff = datetime.utcnow() - timedelta(days=older_than_days)
        removed = 0
        
        for key_id in self._key_versions:
            original_count = len(self._key_versions[key_id])
            self._key_versions[key_id] = [
                v for v in self._key_versions[key_id]
                if v.state != KeyState.RETIRED or (v.rotated_at and v.rotated_at > cutoff)
            ]
            removed += original_count - len(self._key_versions[key_id])
        
        if removed > 0:
            self._save_state()
            logger.info("Cleaned up %d retired key versions", removed)
        
        return removed
    
    def check_and_rotate_all(self) -> list[RotationEvent]:
        """Check all keys and rotate those that need it.
        
        Typically called by a scheduled job.
        """
        if not self.policy.auto_rotate:
            return []
        
        events = []
        for key_id in list(self._key_versions.keys()):
            should_rotate, reason = self.should_rotate(key_id)
            if should_rotate:
                event = self.rotate_key(
                    key_id,
                    reason=RotationReason.SCHEDULED,
                )
                events.append(event)
        
        return events


# Singleton instance
_rotation_manager: Optional[KeyRotationManager] = None


def get_key_rotation_manager(
    policy: Optional[KeyRotationPolicy] = None,
) -> KeyRotationManager:
    """Get or create the global key rotation manager."""
    global _rotation_manager
    
    if _rotation_manager is None:
        _rotation_manager = KeyRotationManager(policy=policy)
    
    return _rotation_manager
