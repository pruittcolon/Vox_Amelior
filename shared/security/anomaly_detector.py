"""
Anomaly Detection for Security.

Statistical and ML-lite anomaly detection for:
- Request pattern anomalies
- Geographic anomalies
- Time-based anomalies
- Behavior deviation

Uses simple statistical methods (no heavy ML dependencies)
to detect unusual patterns that may indicate attacks.
"""

import logging
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


class AnomalyType(Enum):
    """Types of detected anomalies."""
    
    RATE_SPIKE = "rate_spike"           # Sudden traffic increase
    GEO_ANOMALY = "geo_anomaly"         # Unusual location
    TIME_ANOMALY = "time_anomaly"       # Unusual time of access
    BEHAVIOR_CHANGE = "behavior_change" # Changed patterns
    ENDPOINT_SPRAY = "endpoint_spray"   # Hitting many endpoints
    ERROR_SPIKE = "error_spike"         # Many errors in short time
    NEW_USER_AGENT = "new_user_agent"   # First-time user agent
    CREDENTIAL_STUFFING = "cred_stuff"  # Failed login attempts


@dataclass
class AnomalyResult:
    """Result of anomaly detection."""
    
    is_anomalous: bool
    anomaly_score: float  # 0-100
    anomaly_types: list[AnomalyType] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_anomalous": self.is_anomalous,
            "score": self.anomaly_score,
            "types": [t.value for t in self.anomaly_types],
            "details": self.details,
        }


@dataclass 
class AnomalyConfig:
    """Configuration for anomaly detection."""
    
    # Enable/disable detection
    enabled: bool = True
    
    # Thresholds
    anomaly_threshold: float = 70.0  # Score above this is anomalous
    
    # Rate spike detection
    rate_window_seconds: int = 60
    rate_spike_multiplier: float = 3.0  # 3x normal = anomaly
    
    # Error spike detection
    error_threshold: int = 5  # Errors in window
    error_window_seconds: int = 60
    
    # Endpoint spray detection
    endpoint_count_threshold: int = 20  # Unique endpoints in window
    endpoint_window_seconds: int = 60
    
    # Time-based detection
    unusual_hours_start: int = 2   # 2 AM
    unusual_hours_end: int = 5     # 5 AM (local time)


class StatisticalBaseline:
    """Maintains statistical baselines for anomaly detection."""
    
    def __init__(self, window_size: int = 100):
        """Initialize with window size for rolling stats."""
        self.window_size = window_size
        self._values: deque[float] = deque(maxlen=window_size)
        self._sum = 0.0
        self._sum_sq = 0.0
    
    def add(self, value: float) -> None:
        """Add a value to the baseline."""
        if len(self._values) >= self.window_size:
            old = self._values[0]
            self._sum -= old
            self._sum_sq -= old * old
        
        self._values.append(value)
        self._sum += value
        self._sum_sq += value * value
    
    @property
    def mean(self) -> float:
        """Calculate mean of values."""
        if not self._values:
            return 0.0
        return self._sum / len(self._values)
    
    @property
    def std(self) -> float:
        """Calculate standard deviation."""
        if len(self._values) < 2:
            return 0.0
        n = len(self._values)
        variance = (self._sum_sq - (self._sum ** 2) / n) / (n - 1)
        return math.sqrt(max(0, variance))
    
    def z_score(self, value: float) -> float:
        """Calculate z-score for a value."""
        std = self.std
        if std == 0:
            return 0.0
        return (value - self.mean) / std
    
    def is_outlier(self, value: float, threshold: float = 3.0) -> bool:
        """Check if value is an outlier (z-score > threshold)."""
        return abs(self.z_score(value)) > threshold


class AnomalyDetector:
    """Detects anomalous behavior patterns.
    
    Uses statistical methods to identify:
    - Unusual request patterns
    - Suspicious timing
    - Endpoint enumeration
    - Error spikes
    """
    
    def __init__(self, config: Optional[AnomalyConfig] = None):
        """Initialize the detector."""
        self.config = config or AnomalyConfig()
        
        # Per-client tracking
        self._client_requests: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=1000)
        )
        self._client_endpoints: dict[str, deque[tuple[float, str]]] = defaultdict(
            lambda: deque(maxlen=500)
        )
        self._client_errors: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=100)
        )
        self._client_user_agents: dict[str, set[str]] = defaultdict(set)
        
        # Global baselines
        self._rate_baseline = StatisticalBaseline(window_size=100)
        self._error_rate_baseline = StatisticalBaseline(window_size=100)
        
        # Failed login tracking
        self._failed_logins: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=50)
        )
        
        logger.info("AnomalyDetector initialized")
    
    def analyze(
        self,
        client_id: str,
        endpoint: str = "",
        is_error: bool = False,
        user_agent: str = "",
        is_failed_login: bool = False,
    ) -> AnomalyResult:
        """Analyze a request for anomalies.
        
        Args:
            client_id: Client identifier (IP or user ID)
            endpoint: Request endpoint
            is_error: Whether request resulted in error
            user_agent: Client user agent
            is_failed_login: Whether this was a failed login attempt
            
        Returns:
            AnomalyResult with detection results
        """
        if not self.config.enabled:
            return AnomalyResult(is_anomalous=False, anomaly_score=0.0)
        
        now = time.time()
        anomaly_types = []
        total_score = 0.0
        details = {}
        
        # Record request
        self._client_requests[client_id].append(now)
        
        # Check rate spike
        rate_score, is_rate_spike = self._check_rate_spike(client_id, now)
        if is_rate_spike:
            anomaly_types.append(AnomalyType.RATE_SPIKE)
            total_score += rate_score
            details["rate_spike"] = rate_score
        
        # Check endpoint spray
        if endpoint:
            self._client_endpoints[client_id].append((now, endpoint))
            spray_score, is_spray = self._check_endpoint_spray(client_id, now)
            if is_spray:
                anomaly_types.append(AnomalyType.ENDPOINT_SPRAY)
                total_score += spray_score
                details["endpoint_spray"] = spray_score
        
        # Check error spike
        if is_error:
            self._client_errors[client_id].append(now)
            error_score, is_error_spike = self._check_error_spike(client_id, now)
            if is_error_spike:
                anomaly_types.append(AnomalyType.ERROR_SPIKE)
                total_score += error_score
                details["error_spike"] = error_score
        
        # Check time anomaly
        time_score, is_time_anomaly = self._check_time_anomaly(now)
        if is_time_anomaly:
            anomaly_types.append(AnomalyType.TIME_ANOMALY)
            total_score += time_score
            details["time_anomaly"] = time_score
        
        # Check user agent
        if user_agent:
            ua_score, is_new_ua = self._check_user_agent(client_id, user_agent)
            if is_new_ua:
                anomaly_types.append(AnomalyType.NEW_USER_AGENT)
                total_score += ua_score
                details["new_user_agent"] = ua_score
        
        # Check credential stuffing
        if is_failed_login:
            self._failed_logins[client_id].append(now)
            cred_score, is_cred_stuffing = self._check_credential_stuffing(client_id, now)
            if is_cred_stuffing:
                anomaly_types.append(AnomalyType.CREDENTIAL_STUFFING)
                total_score += cred_score
                details["credential_stuffing"] = cred_score
        
        # Normalize score
        final_score = min(100, total_score)
        is_anomalous = final_score >= self.config.anomaly_threshold
        
        if is_anomalous:
            logger.warning(
                "Anomaly detected: client=%s score=%.1f types=%s",
                client_id, final_score, [t.value for t in anomaly_types],
            )
        
        return AnomalyResult(
            is_anomalous=is_anomalous,
            anomaly_score=final_score,
            anomaly_types=anomaly_types,
            details=details,
        )
    
    def _check_rate_spike(self, client_id: str, now: float) -> tuple[float, bool]:
        """Check for sudden rate increase."""
        requests = self._client_requests[client_id]
        window_start = now - self.config.rate_window_seconds
        
        recent_count = sum(1 for t in requests if t > window_start)
        
        # Add to baseline
        self._rate_baseline.add(recent_count)
        
        # Calculate if spike
        if self._rate_baseline.mean > 0:
            ratio = recent_count / self._rate_baseline.mean
            if ratio > self.config.rate_spike_multiplier:
                score = min(50, (ratio - 1) * 15)
                return score, True
        
        return 0.0, False
    
    def _check_endpoint_spray(self, client_id: str, now: float) -> tuple[float, bool]:
        """Check for hitting many different endpoints."""
        endpoints = self._client_endpoints[client_id]
        window_start = now - self.config.endpoint_window_seconds
        
        recent_endpoints = set(
            ep for t, ep in endpoints if t > window_start
        )
        
        if len(recent_endpoints) >= self.config.endpoint_count_threshold:
            score = min(40, (len(recent_endpoints) - self.config.endpoint_count_threshold) * 2 + 20)
            return score, True
        
        return 0.0, False
    
    def _check_error_spike(self, client_id: str, now: float) -> tuple[float, bool]:
        """Check for many errors in short time."""
        errors = self._client_errors[client_id]
        window_start = now - self.config.error_window_seconds
        
        recent_errors = sum(1 for t in errors if t > window_start)
        
        if recent_errors >= self.config.error_threshold:
            score = min(40, recent_errors * 5)
            return score, True
        
        return 0.0, False
    
    def _check_time_anomaly(self, now: float) -> tuple[float, bool]:
        """Check if request is at unusual time."""
        hour = datetime.fromtimestamp(now).hour
        
        if self.config.unusual_hours_start <= hour < self.config.unusual_hours_end:
            return 15.0, True
        
        return 0.0, False
    
    def _check_user_agent(self, client_id: str, user_agent: str) -> tuple[float, bool]:
        """Check if user agent is new for this client."""
        known = self._client_user_agents[client_id]
        
        if user_agent not in known:
            known.add(user_agent)
            # First few are expected
            if len(known) > 3:
                return 20.0, True
        
        return 0.0, False
    
    def _check_credential_stuffing(self, client_id: str, now: float) -> tuple[float, bool]:
        """Check for credential stuffing attack."""
        failed = self._failed_logins[client_id]
        window_start = now - 300  # 5 minute window
        
        recent_failures = sum(1 for t in failed if t > window_start)
        
        if recent_failures >= 5:
            score = min(60, recent_failures * 8)
            return score, True
        
        return 0.0, False
    
    def get_client_profile(self, client_id: str) -> dict[str, Any]:
        """Get behavioral profile for a client."""
        return {
            "request_count": len(self._client_requests.get(client_id, [])),
            "unique_endpoints": len(set(
                ep for _, ep in self._client_endpoints.get(client_id, [])
            )),
            "error_count": len(self._client_errors.get(client_id, [])),
            "user_agents": list(self._client_user_agents.get(client_id, set())),
            "failed_logins": len(self._failed_logins.get(client_id, [])),
        }
    
    def get_stats(self) -> dict[str, Any]:
        """Get detector statistics."""
        return {
            "tracked_clients": len(self._client_requests),
            "rate_baseline_mean": self._rate_baseline.mean,
            "rate_baseline_std": self._rate_baseline.std,
        }


# Singleton
_anomaly_detector: Optional[AnomalyDetector] = None


def get_anomaly_detector(config: Optional[AnomalyConfig] = None) -> AnomalyDetector:
    """Get or create global anomaly detector."""
    global _anomaly_detector
    
    if _anomaly_detector is None:
        _anomaly_detector = AnomalyDetector(config=config)
    
    return _anomaly_detector
