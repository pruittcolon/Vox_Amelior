"""
Account Lockout Module - Redis-backed failed attempt tracking.

Provides brute-force protection by:
- Tracking failed login attempts per username/IP
- Locking accounts after threshold exceeded
- Exponential backoff for repeat offenders
- Automatic unlock after timeout

ISO 27002 Control: A.9.4.3 - Password Management
"""

import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Configuration
MAX_FAILED_ATTEMPTS = int(os.getenv("LOCKOUT_MAX_ATTEMPTS", "5"))
BASE_LOCKOUT_SECONDS = int(os.getenv("LOCKOUT_BASE_SECONDS", "900"))  # 15 minutes
MAX_LOCKOUT_SECONDS = int(os.getenv("LOCKOUT_MAX_SECONDS", "86400"))  # 24 hours
ATTEMPT_WINDOW_SECONDS = int(os.getenv("LOCKOUT_WINDOW_SECONDS", "3600"))  # 1 hour

# Redis key prefixes
ATTEMPTS_KEY_PREFIX = "lockout:attempts:"
LOCKOUT_KEY_PREFIX = "lockout:locked:"
BACKOFF_KEY_PREFIX = "lockout:backoff:"


@dataclass
class LockoutStatus:
    """Result of lockout check."""
    is_locked: bool
    remaining_seconds: int = 0
    failed_attempts: int = 0
    message: str = ""


# Optional Redis import (fallback to in-memory if unavailable)
try:
    import redis
    _redis_available = True
except ImportError:
    redis = None
    _redis_available = False


class _InMemoryLockoutStore:
    """Process-local lockout store fallback."""
    
    def __init__(self):
        self._attempts: dict[str, list[float]] = {}
        self._lockouts: dict[str, float] = {}
        self._backoff_multipliers: dict[str, int] = {}
    
    def get_attempts(self, key: str) -> list[float]:
        """Get failed attempt timestamps."""
        now = time.time()
        # Clean expired attempts
        if key in self._attempts:
            self._attempts[key] = [
                ts for ts in self._attempts[key] 
                if now - ts < ATTEMPT_WINDOW_SECONDS
            ]
        return self._attempts.get(key, [])
    
    def add_attempt(self, key: str) -> int:
        """Add a failed attempt, return total count."""
        now = time.time()
        if key not in self._attempts:
            self._attempts[key] = []
        self._attempts[key].append(now)
        # Clean old attempts
        self._attempts[key] = [
            ts for ts in self._attempts[key] 
            if now - ts < ATTEMPT_WINDOW_SECONDS
        ]
        return len(self._attempts[key])
    
    def clear_attempts(self, key: str) -> None:
        """Clear failed attempts on successful login."""
        self._attempts.pop(key, None)
        self._backoff_multipliers.pop(key, None)
    
    def set_lockout(self, key: str, duration_seconds: int) -> None:
        """Lock the account."""
        self._lockouts[key] = time.time() + duration_seconds
        # Increment backoff multiplier
        self._backoff_multipliers[key] = self._backoff_multipliers.get(key, 0) + 1
    
    def get_lockout(self, key: str) -> Optional[float]:
        """Get lockout expiry time, None if not locked."""
        if key not in self._lockouts:
            return None
        if time.time() > self._lockouts[key]:
            del self._lockouts[key]
            return None
        return self._lockouts[key]
    
    def get_backoff_multiplier(self, key: str) -> int:
        """Get exponential backoff multiplier."""
        return self._backoff_multipliers.get(key, 0)


class AccountLockout:
    """
    Account lockout manager with Redis backend.
    
    Features:
    - Track failed login attempts per user
    - Lock account after MAX_FAILED_ATTEMPTS
    - Exponential backoff for repeat offenders
    - Automatic unlock after timeout
    
    Usage:
        lockout = AccountLockout()
        
        # Check before login
        status = lockout.check_lockout(username)
        if status.is_locked:
            raise HTTPException(423, status.message)
        
        # Record failed attempt
        lockout.record_failed_attempt(username)
        
        # Clear on success
        lockout.record_success(username)
    """
    
    def __init__(self, redis_url: Optional[str] = None):
        """
        Initialize lockout manager.
        
        Args:
            redis_url: Redis connection URL, falls back to REDIS_URL env var
        """
        self._redis: Optional["redis.Redis"] = None
        self._fallback = _InMemoryLockoutStore()
        
        if _redis_available:
            url = redis_url or os.getenv("REDIS_URL", "redis://redis:6379/0")
            try:
                self._redis = redis.from_url(url, decode_responses=True)
                self._redis.ping()
                logger.info("AccountLockout: Redis connected")
            except Exception as e:
                logger.warning(f"AccountLockout: Redis unavailable ({e}), using in-memory fallback")
                self._redis = None
        else:
            logger.warning("AccountLockout: redis package not installed, using in-memory fallback")
    
    def _get_attempts_key(self, identifier: str) -> str:
        """Get Redis key for failed attempts."""
        return f"{ATTEMPTS_KEY_PREFIX}{identifier}"
    
    def _get_lockout_key(self, identifier: str) -> str:
        """Get Redis key for lockout status."""
        return f"{LOCKOUT_KEY_PREFIX}{identifier}"
    
    def _get_backoff_key(self, identifier: str) -> str:
        """Get Redis key for backoff multiplier."""
        return f"{BACKOFF_KEY_PREFIX}{identifier}"
    
    def check_lockout(self, identifier: str) -> LockoutStatus:
        """
        Check if account is locked.
        
        Args:
            identifier: Username or IP address
            
        Returns:
            LockoutStatus with lock state and details
        """
        if self._redis:
            try:
                ttl = self._redis.ttl(self._get_lockout_key(identifier))
                if ttl > 0:
                    return LockoutStatus(
                        is_locked=True,
                        remaining_seconds=ttl,
                        message=f"Account locked. Try again in {ttl // 60} minutes."
                    )
                
                # Get current attempt count
                attempts = self._redis.llen(self._get_attempts_key(identifier))
                return LockoutStatus(
                    is_locked=False,
                    failed_attempts=attempts
                )
            except Exception as e:
                logger.error(f"Redis error in check_lockout: {e}")
                # Fall through to fallback
        
        # Fallback
        lockout_expiry = self._fallback.get_lockout(identifier)
        if lockout_expiry:
            remaining = int(lockout_expiry - time.time())
            return LockoutStatus(
                is_locked=True,
                remaining_seconds=remaining,
                message=f"Account locked. Try again in {remaining // 60} minutes."
            )
        
        attempts = self._fallback.get_attempts(identifier)
        return LockoutStatus(
            is_locked=False,
            failed_attempts=len(attempts)
        )
    
    def record_failed_attempt(self, identifier: str) -> LockoutStatus:
        """
        Record a failed login attempt.
        
        Args:
            identifier: Username or IP address
            
        Returns:
            LockoutStatus (may indicate account is now locked)
        """
        if self._redis:
            try:
                key = self._get_attempts_key(identifier)
                # Add attempt timestamp
                self._redis.rpush(key, str(time.time()))
                self._redis.expire(key, ATTEMPT_WINDOW_SECONDS)
                
                # Check if threshold exceeded
                attempts = self._redis.llen(key)
                if attempts >= MAX_FAILED_ATTEMPTS:
                    return self._lock_account(identifier)
                
                remaining = MAX_FAILED_ATTEMPTS - attempts
                return LockoutStatus(
                    is_locked=False,
                    failed_attempts=attempts,
                    message=f"{remaining} attempts remaining before lockout"
                )
            except Exception as e:
                logger.error(f"Redis error in record_failed_attempt: {e}")
                # Fall through to fallback
        
        # Fallback
        attempts = self._fallback.add_attempt(identifier)
        if attempts >= MAX_FAILED_ATTEMPTS:
            return self._lock_account(identifier)
        
        remaining = MAX_FAILED_ATTEMPTS - attempts
        return LockoutStatus(
            is_locked=False,
            failed_attempts=attempts,
            message=f"{remaining} attempts remaining before lockout"
        )
    
    def _lock_account(self, identifier: str) -> LockoutStatus:
        """
        Lock an account with exponential backoff.
        
        Args:
            identifier: Username or IP address
            
        Returns:
            LockoutStatus indicating locked state
        """
        # Calculate lockout duration with exponential backoff
        if self._redis:
            try:
                backoff_key = self._get_backoff_key(identifier)
                multiplier = int(self._redis.get(backoff_key) or 0)
                duration = min(
                    BASE_LOCKOUT_SECONDS * (2 ** multiplier),
                    MAX_LOCKOUT_SECONDS
                )
                
                # Set lockout
                self._redis.setex(
                    self._get_lockout_key(identifier),
                    duration,
                    "1"
                )
                
                # Increment backoff multiplier
                self._redis.incr(backoff_key)
                self._redis.expire(backoff_key, MAX_LOCKOUT_SECONDS * 2)
                
                # Clear attempts
                self._redis.delete(self._get_attempts_key(identifier))
                
                logger.warning(
                    f"Account locked: {identifier} for {duration}s "
                    f"(backoff level {multiplier + 1})"
                )
                
                return LockoutStatus(
                    is_locked=True,
                    remaining_seconds=duration,
                    message=f"Account locked for {duration // 60} minutes due to too many failed attempts."
                )
            except Exception as e:
                logger.error(f"Redis error in _lock_account: {e}")
                # Fall through to fallback
        
        # Fallback
        multiplier = self._fallback.get_backoff_multiplier(identifier)
        duration = min(
            BASE_LOCKOUT_SECONDS * (2 ** multiplier),
            MAX_LOCKOUT_SECONDS
        )
        self._fallback.set_lockout(identifier, duration)
        self._fallback.clear_attempts(identifier)
        
        logger.warning(
            f"Account locked (fallback): {identifier} for {duration}s "
            f"(backoff level {multiplier + 1})"
        )
        
        return LockoutStatus(
            is_locked=True,
            remaining_seconds=duration,
            message=f"Account locked for {duration // 60} minutes due to too many failed attempts."
        )
    
    def record_success(self, identifier: str) -> None:
        """
        Record successful login, clear failed attempts.
        
        Args:
            identifier: Username or IP address
        """
        if self._redis:
            try:
                self._redis.delete(self._get_attempts_key(identifier))
                self._redis.delete(self._get_backoff_key(identifier))
                logger.debug(f"Cleared failed attempts for: {identifier}")
                return
            except Exception as e:
                logger.error(f"Redis error in record_success: {e}")
        
        # Fallback
        self._fallback.clear_attempts(identifier)


# Singleton instance
_lockout_manager: Optional[AccountLockout] = None


def get_lockout_manager() -> AccountLockout:
    """Get or create lockout manager singleton."""
    global _lockout_manager
    if _lockout_manager is None:
        _lockout_manager = AccountLockout()
    return _lockout_manager
