"""
Redis-backed Rate Limiting (ISO 27002 8.4 / OWASP API4:2023)

Enterprise-grade rate limiting with:
- Sliding window algorithm for accurate limiting
- Redis backend for distributed state
- Trusted proxy IP extraction via X-Forwarded-For
- Separate limits per endpoint type (auth, upload, generation)
- Fallback to in-memory when Redis unavailable
"""

import logging
import os
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class RateLimitConfig:
    """Rate limit configuration per endpoint type."""

    window_seconds: int = 60
    max_requests: int = 120

    # Separate limits per endpoint type
    LOGIN_LIMIT: int = 10  # /api/auth/login
    REGISTER_LIMIT: int = 5  # /api/auth/register
    UPLOAD_LIMIT: int = 20  # File uploads
    GENERATE_LIMIT: int = 30  # Gemma generation
    WEBSOCKET_LIMIT: int = 100  # WebSocket connections
    DEFAULT_LIMIT: int = 120  # Everything else


def get_rate_limit_config() -> RateLimitConfig:
    """Load rate limit config from environment."""
    return RateLimitConfig(
        window_seconds=int(os.getenv("RATE_LIMIT_WINDOW_SEC", "60")),
        max_requests=int(os.getenv("RATE_LIMIT_DEFAULT", "120")),
        LOGIN_LIMIT=int(os.getenv("RATE_LIMIT_LOGIN", "10")),
        REGISTER_LIMIT=int(os.getenv("RATE_LIMIT_REGISTER", "5")),
        UPLOAD_LIMIT=int(os.getenv("RATE_LIMIT_UPLOAD", "20")),
        GENERATE_LIMIT=int(os.getenv("RATE_LIMIT_GENERATE", "30")),
        WEBSOCKET_LIMIT=int(os.getenv("RATE_LIMIT_WEBSOCKET", "100")),
        DEFAULT_LIMIT=int(os.getenv("RATE_LIMIT_DEFAULT", "120")),
    )


# =============================================================================
# TRUSTED PROXY IP EXTRACTION
# =============================================================================

# Trusted proxy IPs - only trust X-Forwarded-For from these sources
TRUSTED_PROXIES: list[str] = [
    ip.strip()
    for ip in os.getenv("TRUSTED_PROXY_IPS", "127.0.0.1,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,::1").split(",")
]


def _ip_in_network(ip: str, network: str) -> bool:
    """Check if IP is in a CIDR network (simple check for common cases)."""
    try:
        import ipaddress

        return ipaddress.ip_address(ip) in ipaddress.ip_network(network, strict=False)
    except Exception:
        return ip == network


def is_trusted_proxy(ip: str) -> bool:
    """Check if an IP is a trusted proxy."""
    if not ip:
        return False
    for trusted in TRUSTED_PROXIES:
        if "/" in trusted:
            if _ip_in_network(ip, trusted):
                return True
        elif ip == trusted:
            return True
    return False


def extract_client_ip(request) -> str:
    """
    Extract real client IP, trusting X-Forwarded-For only from known proxies.

    Per OWASP guidance:
    - Only trust X-Forwarded-For if the immediate connection is from a known proxy
    - Take the rightmost non-trusted IP as the real client
    """
    direct_ip = request.client.host if request.client else "unknown"

    # Only trust forwarded headers if request came from trusted proxy
    if not is_trusted_proxy(direct_ip):
        return direct_ip

    # Parse X-Forwarded-For (format: client, proxy1, proxy2)
    forwarded_for = request.headers.get("x-forwarded-for", "")
    if not forwarded_for:
        return direct_ip

    # Get list of IPs (rightmost is closest to us)
    ips = [ip.strip() for ip in forwarded_for.split(",")]

    # Find rightmost non-trusted IP (the real client)
    for ip in reversed(ips):
        if ip and not is_trusted_proxy(ip):
            return ip

    return direct_ip


# =============================================================================
# REDIS RATE LIMITER
# =============================================================================


class RedisRateLimiter:
    """
    Redis-backed sliding window rate limiter.

    Uses sorted sets with timestamps for accurate sliding window limit.
    Falls back to in-memory limiting if Redis unavailable.
    """

    def __init__(self, redis_url: str, config: RateLimitConfig | None = None):
        self.config = config or get_rate_limit_config()
        self.redis_client = None
        self._redis_available = False
        self._fallback_buckets: dict[str, dict[str, any]] = {}

        try:
            import redis

            self.redis_client = redis.Redis.from_url(
                redis_url,
                decode_responses=True,
                socket_timeout=1.0,
                socket_connect_timeout=1.0,
            )
            # Test connection
            self.redis_client.ping()
            self._redis_available = True
            logger.info("✅ Redis rate limiting initialized")
        except Exception as e:
            logger.warning(f"⚠️ Redis unavailable for rate limiting, using in-memory fallback: {e}")
            self._redis_available = False

    def get_limit_for_path(self, path: str) -> int:
        """Get rate limit for a specific path."""
        if "/auth/login" in path:
            return self.config.LOGIN_LIMIT
        elif "/auth/register" in path:
            return self.config.REGISTER_LIMIT
        elif "/upload" in path or "/ingest" in path:
            return self.config.UPLOAD_LIMIT
        elif "/gemma/" in path:
            return self.config.GENERATE_LIMIT
        elif path.startswith("/ws"):
            return self.config.WEBSOCKET_LIMIT
        return self.config.DEFAULT_LIMIT

    def check_rate_limit(self, key: str, path: str) -> tuple[bool, int, int]:
        """
        Check if request is within rate limit.

        Returns: (allowed: bool, remaining: int, retry_after: int)
        """
        limit = self.get_limit_for_path(path)
        window = self.config.window_seconds

        if self._redis_available:
            return self._check_redis(key, limit, window)
        else:
            return self._check_fallback(key, limit, window)

    def _check_redis(self, key: str, limit: int, window: int) -> tuple[bool, int, int]:
        """Check rate limit using Redis sorted set (sliding window)."""
        try:
            now = time.time()
            window_start = now - window
            redis_key = f"rate_limit:{key}"

            pipe = self.redis_client.pipeline()
            # Remove expired entries
            pipe.zremrangebyscore(redis_key, 0, window_start)
            # Count current requests
            pipe.zcard(redis_key)
            # Add current request with timestamp
            pipe.zadd(redis_key, {f"{now}": now})
            # Set TTL
            pipe.expire(redis_key, window + 60)

            results = pipe.execute()
            current_count = results[1]

            if current_count >= limit:
                # Get oldest entry to calculate retry_after
                oldest = self.redis_client.zrange(redis_key, 0, 0, withscores=True)
                if oldest:
                    retry_after = int(oldest[0][1] + window - now) + 1
                else:
                    retry_after = window
                return False, 0, retry_after

            return True, limit - current_count - 1, 0

        except Exception as e:
            logger.error(f"Redis rate limit error: {e}")
            return self._check_fallback(key, limit, window)

    def _check_fallback(self, key: str, limit: int, window: int) -> tuple[bool, int, int]:
        """In-memory fallback rate limiting."""
        now = int(time.time())

        bucket = self._fallback_buckets.get(key, {"count": 0, "window_start": now})

        # Reset window if expired
        if now - bucket["window_start"] >= window:
            bucket = {"count": 0, "window_start": now}

        bucket["count"] += 1
        self._fallback_buckets[key] = bucket

        if bucket["count"] > limit:
            retry_after = window - (now - bucket["window_start"]) or 1
            return False, 0, retry_after

        return True, limit - bucket["count"], 0


# =============================================================================
# SINGLETON ACCESS
# =============================================================================

_rate_limiter: RedisRateLimiter | None = None


def get_rate_limiter(redis_url: str | None = None) -> RedisRateLimiter:
    """Get or create the global rate limiter instance."""
    global _rate_limiter
    if _rate_limiter is None:
        if redis_url is None:
            redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
        _rate_limiter = RedisRateLimiter(redis_url)
    return _rate_limiter


def reset_rate_limiter():
    """Reset the rate limiter (for testing)."""
    global _rate_limiter
    _rate_limiter = None
