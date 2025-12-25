"""
Cache Layer - Redis-backed caching for performance optimization.

Provides TTL-based caching with invalidation patterns.
"""

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import timedelta
from functools import wraps
from typing import Any, Callable, Optional


logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """Cache configuration."""

    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    default_ttl_seconds: int = 300  # 5 minutes
    prefix: str = "nemo:"


# Try to import Redis
_redis_available = False
_redis_client = None

try:
    import redis
    _redis_available = True
except ImportError:
    pass


class CacheLayer:
    """
    Redis-backed caching layer.

    Usage:
        cache = CacheLayer()
        cache.set("key", "value", ttl=60)
        value = cache.get("key")

        # Or use as decorator
        @cache.cached(ttl=300)
        def expensive_query():
            ...
    """

    def __init__(self, config: Optional[CacheConfig] = None):
        """
        Initialize cache layer.

        Args:
            config: Optional cache configuration
        """
        self.config = config or CacheConfig()
        self._client = None
        self._fallback: dict[str, tuple[Any, float]] = {}  # In-memory fallback

        if _redis_available:
            try:
                self._client = redis.Redis(
                    host=self.config.host,
                    port=self.config.port,
                    db=self.config.db,
                    password=self.config.password,
                    decode_responses=True,
                    socket_timeout=5,
                )
                # Test connection
                self._client.ping()
                logger.info(f"Redis cache connected: {self.config.host}:{self.config.port}")
            except Exception as e:
                logger.warning(f"Redis unavailable, using in-memory fallback: {e}")
                self._client = None

    def _make_key(self, key: str) -> str:
        """Create prefixed cache key."""
        return f"{self.config.prefix}{key}"

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None
        """
        cache_key = self._make_key(key)

        if self._client:
            try:
                value = self._client.get(cache_key)
                if value:
                    return json.loads(value)
            except Exception as e:
                logger.debug(f"Cache get error: {e}")

        # Fallback check
        if cache_key in self._fallback:
            value, expires = self._fallback[cache_key]
            import time
            if time.time() < expires:
                return value
            else:
                del self._fallback[cache_key]

        return None

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> bool:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache (must be JSON serializable)
            ttl: Time-to-live in seconds

        Returns:
            True if successful
        """
        cache_key = self._make_key(key)
        ttl = ttl or self.config.default_ttl_seconds

        try:
            serialized = json.dumps(value)
        except TypeError:
            logger.warning(f"Value not JSON serializable: {key}")
            return False

        if self._client:
            try:
                self._client.setex(cache_key, ttl, serialized)
                return True
            except Exception as e:
                logger.debug(f"Cache set error: {e}")

        # Fallback
        import time
        self._fallback[cache_key] = (value, time.time() + ttl)
        return True

    def delete(self, key: str) -> bool:
        """
        Delete value from cache.

        Args:
            key: Cache key

        Returns:
            True if deleted
        """
        cache_key = self._make_key(key)

        if self._client:
            try:
                self._client.delete(cache_key)
            except Exception as e:
                logger.debug(f"Cache delete error: {e}")

        if cache_key in self._fallback:
            del self._fallback[cache_key]

        return True

    def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidate all keys matching pattern.

        Args:
            pattern: Glob pattern (e.g., "user:*")

        Returns:
            Number of keys deleted
        """
        full_pattern = self._make_key(pattern)
        count = 0

        if self._client:
            try:
                keys = self._client.keys(full_pattern)
                if keys:
                    count = self._client.delete(*keys)
            except Exception as e:
                logger.debug(f"Cache invalidate error: {e}")

        # Fallback cleanup
        import fnmatch
        to_delete = [k for k in self._fallback if fnmatch.fnmatch(k, full_pattern)]
        for k in to_delete:
            del self._fallback[k]
            count += 1

        return count

    def cached(
        self,
        ttl: Optional[int] = None,
        key_prefix: str = "",
    ) -> Callable:
        """
        Decorator for caching function results.

        Args:
            ttl: Cache TTL in seconds
            key_prefix: Prefix for cache key

        Returns:
            Decorated function
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key from function name and arguments
                key_parts = [key_prefix or func.__name__]
                key_parts.extend(str(a) for a in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = ":".join(key_parts)

                # Hash if too long
                if len(cache_key) > 200:
                    cache_key = hashlib.md5(cache_key.encode()).hexdigest()

                # Check cache
                cached = self.get(cache_key)
                if cached is not None:
                    return cached

                # Call function and cache result
                result = func(*args, **kwargs)
                self.set(cache_key, result, ttl)
                return result

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                key_parts = [key_prefix or func.__name__]
                key_parts.extend(str(a) for a in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = ":".join(key_parts)

                if len(cache_key) > 200:
                    cache_key = hashlib.md5(cache_key.encode()).hexdigest()

                cached = self.get(cache_key)
                if cached is not None:
                    return cached

                result = await func(*args, **kwargs)
                self.set(cache_key, result, ttl)
                return result

            import asyncio
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            return wrapper

        return decorator

    def get_stats(self) -> dict:
        """Get cache statistics."""
        stats = {
            "backend": "redis" if self._client else "in-memory",
            "connected": self._client is not None,
            "fallback_size": len(self._fallback),
        }

        if self._client:
            try:
                info = self._client.info("stats")
                stats.update({
                    "hits": info.get("keyspace_hits", 0),
                    "misses": info.get("keyspace_misses", 0),
                })
            except Exception:
                pass

        return stats


# Singleton instance
_cache: Optional[CacheLayer] = None


def get_cache() -> CacheLayer:
    """Get or create singleton cache layer."""
    global _cache
    if _cache is None:
        _cache = CacheLayer()
    return _cache
