"""
Adaptive Rate Limiting.

Intelligent rate limiting that adjusts based on:
- Client behavior patterns
- Threat scores from WAF
- Time of day / load conditions
- Historical violation patterns

Provides more aggressive limiting for suspicious clients
while allowing legitimate users more flexibility.
"""

import hashlib
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ClientRisk(Enum):
    """Client risk classification."""
    
    TRUSTED = "trusted"       # Known good client
    NORMAL = "normal"         # Default for new clients
    SUSPICIOUS = "suspicious" # Some warning signs
    HIGH_RISK = "high_risk"   # Multiple violations
    BLOCKED = "blocked"       # Temporary block


@dataclass
class RateLimitConfig:
    """Rate limit configuration."""
    
    # Default limits (requests per window)
    default_requests_per_minute: int = 100
    default_requests_per_hour: int = 1000
    
    # Limits by risk level
    trusted_multiplier: float = 2.0      # Trusted get 2x limit
    suspicious_multiplier: float = 0.5    # Suspicious get 0.5x
    high_risk_multiplier: float = 0.1     # High risk get 0.1x
    
    # Threat score thresholds
    suspicious_threshold: float = 20.0
    high_risk_threshold: float = 50.0
    block_threshold: float = 80.0
    
    # Auto-escalation settings
    violations_to_suspicious: int = 3
    violations_to_high_risk: int = 10
    violations_to_block: int = 20
    
    # Decay settings
    violation_decay_hours: int = 24  # Violations decay after 24 hours
    
    # Block duration
    block_duration_minutes: int = 30


@dataclass
class ClientState:
    """Tracks state for a single client."""
    
    client_id: str
    risk_level: ClientRisk = ClientRisk.NORMAL
    
    # Request tracking
    requests_minute: list[float] = field(default_factory=list)
    requests_hour: list[float] = field(default_factory=list)
    
    # Violation tracking
    violations: list[float] = field(default_factory=list)
    total_threat_score: float = 0.0
    
    # Block info
    blocked_until: Optional[float] = None
    
    # First/last seen
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    
    def is_blocked(self) -> bool:
        """Check if client is currently blocked."""
        if self.blocked_until is None:
            return False
        return time.time() < self.blocked_until


@dataclass
class RateLimitResult:
    """Result of rate limit check."""
    
    allowed: bool
    client_risk: ClientRisk
    current_rate: float
    limit: int
    retry_after_seconds: Optional[int] = None
    reason: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "allowed": self.allowed,
            "risk": self.client_risk.value,
            "rate": self.current_rate,
            "limit": self.limit,
            "retry_after": self.retry_after_seconds,
            "reason": self.reason,
        }


class AdaptiveRateLimiter:
    """Adaptive rate limiter with threat-based adjustment.
    
    Features:
    - Per-client rate tracking
    - Risk-based limit adjustment
    - WAF integration for threat scoring
    - Automatic escalation/de-escalation
    """
    
    def __init__(self, config: Optional[RateLimitConfig] = None):
        """Initialize the rate limiter.
        
        Args:
            config: Rate limit configuration
        """
        self.config = config or RateLimitConfig()
        
        # Client states: client_id -> ClientState
        self._clients: dict[str, ClientState] = {}
        
        # Cleanup tracking
        self._last_cleanup = time.time()
        self._cleanup_interval = 300  # 5 minutes
        
        logger.info("AdaptiveRateLimiter initialized")
    
    def _get_client_id(
        self,
        ip: str,
        user_id: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> str:
        """Generate a consistent client identifier."""
        # Prefer user_id or api_key over IP
        if user_id:
            return f"user:{user_id}"
        if api_key:
            return f"key:{hashlib.sha256(api_key.encode()).hexdigest()[:16]}"
        return f"ip:{ip}"
    
    def _get_or_create_client(self, client_id: str) -> ClientState:
        """Get or create client state."""
        if client_id not in self._clients:
            self._clients[client_id] = ClientState(client_id=client_id)
        return self._clients[client_id]
    
    def _get_limit_for_risk(self, risk: ClientRisk, base_limit: int) -> int:
        """Get adjusted limit based on risk level."""
        multipliers = {
            ClientRisk.TRUSTED: self.config.trusted_multiplier,
            ClientRisk.NORMAL: 1.0,
            ClientRisk.SUSPICIOUS: self.config.suspicious_multiplier,
            ClientRisk.HIGH_RISK: self.config.high_risk_multiplier,
            ClientRisk.BLOCKED: 0,
        }
        return int(base_limit * multipliers.get(risk, 1.0))
    
    def _cleanup_old_requests(self, state: ClientState, now: float) -> None:
        """Remove old request timestamps."""
        minute_ago = now - 60
        hour_ago = now - 3600
        
        state.requests_minute = [t for t in state.requests_minute if t > minute_ago]
        state.requests_hour = [t for t in state.requests_hour if t > hour_ago]
        
        # Decay violations
        decay_time = now - (self.config.violation_decay_hours * 3600)
        state.violations = [t for t in state.violations if t > decay_time]
    
    def _update_risk_level(self, state: ClientState) -> None:
        """Update client risk level based on violations."""
        violation_count = len(state.violations)
        
        if state.blocked_until and time.time() < state.blocked_until:
            state.risk_level = ClientRisk.BLOCKED
        elif violation_count >= self.config.violations_to_block:
            state.risk_level = ClientRisk.BLOCKED
            state.blocked_until = time.time() + (self.config.block_duration_minutes * 60)
            logger.warning(
                "Client %s blocked for %d minutes",
                state.client_id, self.config.block_duration_minutes,
            )
        elif violation_count >= self.config.violations_to_high_risk:
            state.risk_level = ClientRisk.HIGH_RISK
        elif violation_count >= self.config.violations_to_suspicious:
            state.risk_level = ClientRisk.SUSPICIOUS
        elif state.total_threat_score >= self.config.high_risk_threshold:
            state.risk_level = ClientRisk.HIGH_RISK
        elif state.total_threat_score >= self.config.suspicious_threshold:
            state.risk_level = ClientRisk.SUSPICIOUS
        else:
            # Allow recovery to normal
            if state.risk_level in (ClientRisk.SUSPICIOUS, ClientRisk.HIGH_RISK):
                if violation_count == 0 and state.total_threat_score < self.config.suspicious_threshold:
                    state.risk_level = ClientRisk.NORMAL
    
    def check(
        self,
        ip: str,
        user_id: Optional[str] = None,
        api_key: Optional[str] = None,
        threat_score: float = 0.0,
        endpoint: str = "",
    ) -> RateLimitResult:
        """Check if request should be allowed.
        
        Args:
            ip: Client IP address
            user_id: Optional authenticated user ID
            api_key: Optional API key
            threat_score: WAF threat score for this request
            endpoint: Request endpoint (for per-endpoint limits)
            
        Returns:
            RateLimitResult indicating if request is allowed
        """
        now = time.time()
        client_id = self._get_client_id(ip, user_id, api_key)
        state = self._get_or_create_client(client_id)
        
        # Periodic cleanup
        if now - self._last_cleanup > self._cleanup_interval:
            self._cleanup_all()
            self._last_cleanup = now
        
        # Cleanup old requests for this client
        self._cleanup_old_requests(state, now)
        
        # Update threat score
        if threat_score > 0:
            state.total_threat_score = max(
                state.total_threat_score,
                state.total_threat_score * 0.9 + threat_score * 0.1,
            )
        else:
            # Decay threat score over time
            state.total_threat_score *= 0.99
        
        # Update risk level
        self._update_risk_level(state)
        
        # Check if blocked
        if state.is_blocked():
            retry_after = int(state.blocked_until - now)
            return RateLimitResult(
                allowed=False,
                client_risk=state.risk_level,
                current_rate=len(state.requests_minute),
                limit=0,
                retry_after_seconds=retry_after,
                reason="Client temporarily blocked",
            )
        
        # Calculate limit based on risk
        limit = self._get_limit_for_risk(
            state.risk_level,
            self.config.default_requests_per_minute,
        )
        
        # Check rate
        current_rate = len(state.requests_minute)
        
        if current_rate >= limit:
            # Rate limit exceeded
            state.violations.append(now)
            self._update_risk_level(state)
            
            logger.info(
                "Rate limit exceeded: client=%s rate=%d limit=%d risk=%s",
                client_id, current_rate, limit, state.risk_level.value,
            )
            
            return RateLimitResult(
                allowed=False,
                client_risk=state.risk_level,
                current_rate=current_rate,
                limit=limit,
                retry_after_seconds=60,
                reason="Rate limit exceeded",
            )
        
        # Allow request
        state.requests_minute.append(now)
        state.requests_hour.append(now)
        state.last_seen = now
        
        return RateLimitResult(
            allowed=True,
            client_risk=state.risk_level,
            current_rate=current_rate + 1,
            limit=limit,
        )
    
    def record_violation(
        self,
        ip: str,
        user_id: Optional[str] = None,
        api_key: Optional[str] = None,
        threat_score: float = 0.0,
    ) -> None:
        """Record a violation (e.g., WAF block) for a client."""
        client_id = self._get_client_id(ip, user_id, api_key)
        state = self._get_or_create_client(client_id)
        
        state.violations.append(time.time())
        state.total_threat_score += threat_score
        self._update_risk_level(state)
    
    def mark_trusted(
        self,
        ip: str,
        user_id: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        """Mark a client as trusted (increased limits)."""
        client_id = self._get_client_id(ip, user_id, api_key)
        state = self._get_or_create_client(client_id)
        state.risk_level = ClientRisk.TRUSTED
    
    def unblock(
        self,
        ip: str,
        user_id: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        """Manually unblock a client."""
        client_id = self._get_client_id(ip, user_id, api_key)
        if client_id in self._clients:
            self._clients[client_id].blocked_until = None
            self._clients[client_id].risk_level = ClientRisk.SUSPICIOUS
            self._clients[client_id].violations = []
    
    def _cleanup_all(self) -> None:
        """Remove stale client entries."""
        now = time.time()
        stale_threshold = now - 7200  # 2 hours
        
        stale_clients = [
            cid for cid, state in self._clients.items()
            if state.last_seen < stale_threshold
            and state.risk_level not in (ClientRisk.HIGH_RISK, ClientRisk.BLOCKED)
        ]
        
        for cid in stale_clients:
            del self._clients[cid]
        
        if stale_clients:
            logger.debug("Cleaned up %d stale client entries", len(stale_clients))
    
    def get_stats(self) -> dict[str, Any]:
        """Get rate limiter statistics."""
        risk_counts = defaultdict(int)
        for state in self._clients.values():
            risk_counts[state.risk_level.value] += 1
        
        return {
            "total_clients": len(self._clients),
            "risk_distribution": dict(risk_counts),
            "blocked_count": sum(
                1 for s in self._clients.values() if s.is_blocked()
            ),
        }


# Singleton
_rate_limiter: Optional[AdaptiveRateLimiter] = None


def get_rate_limiter(config: Optional[RateLimitConfig] = None) -> AdaptiveRateLimiter:
    """Get or create global rate limiter."""
    global _rate_limiter
    
    if _rate_limiter is None:
        _rate_limiter = AdaptiveRateLimiter(config=config)
    
    return _rate_limiter
