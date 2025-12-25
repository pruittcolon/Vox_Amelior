"""
Base HTTP client with retry, circuit breaker, and S2S JWT authentication.

Phase 2: Enterprise patterns for inter-service communication.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum

import httpx

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreaker:
    """
    Circuit breaker pattern for fault tolerance.

    - CLOSED: All requests go through
    - OPEN: All requests fail immediately (after failure_threshold)
    - HALF_OPEN: Allow one request to test recovery
    """

    failure_threshold: int = 5
    recovery_timeout: float = 30.0  # seconds

    failure_count: int = field(default=0, init=False)
    last_failure_time: float = field(default=0.0, init=False)
    state: CircuitState = field(default=CircuitState.CLOSED, init=False)

    def record_success(self) -> None:
        """Record successful call."""
        self.failure_count = 0
        self.state = CircuitState.CLOSED

    def record_failure(self) -> None:
        """Record failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning("Circuit breaker OPEN after %d failures", self.failure_count)

    def should_allow_request(self) -> bool:
        """Check if request should be allowed."""
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                logger.info("Circuit breaker HALF_OPEN, testing recovery")
                return True
            return False

        # HALF_OPEN: allow one request
        return True


class BaseServiceClient:
    """
    Base HTTP client for inter-service communication.

    Features:
    - Automatic retry with exponential backoff
    - Circuit breaker for fault tolerance
    - S2S JWT authentication headers
    - Request timeout handling
    """

    def __init__(
        self,
        base_url: str,
        service_name: str,
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 0.5,
    ):
        self.base_url = base_url.rstrip("/")
        self.service_name = service_name
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.circuit_breaker = CircuitBreaker()
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
            )
        return self._client

    def _get_auth_headers(self) -> dict[str, str]:
        """Get S2S JWT authentication headers."""
        try:
            from shared.security.service_auth import get_service_auth, load_service_jwt_keys

            keys = load_service_jwt_keys("gateway")
            auth = get_service_auth("gateway", keys)
            return auth.get_auth_header()
        except Exception as e:
            logger.warning("Failed to get S2S auth headers: %s", e)
            return {}

    async def request(
        self,
        method: str,
        path: str,
        **kwargs,
    ) -> httpx.Response:
        """
        Make HTTP request with retry and circuit breaker.

        Args:
            method: HTTP method (GET, POST, etc.)
            path: Request path
            **kwargs: Additional arguments for httpx

        Returns:
            httpx.Response

        Raises:
            httpx.HTTPError: On request failure
            RuntimeError: If circuit breaker is open
        """
        if not self.circuit_breaker.should_allow_request():
            raise RuntimeError(f"Circuit breaker OPEN for {self.service_name}")

        client = await self._get_client()

        # Add auth headers
        headers = kwargs.pop("headers", {})
        headers.update(self._get_auth_headers())

        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                response = await client.request(
                    method,
                    path,
                    headers=headers,
                    **kwargs,
                )

                # Success: record and return
                self.circuit_breaker.record_success()
                return response

            except (TimeoutError, httpx.HTTPError) as e:
                last_error = e
                self.circuit_breaker.record_failure()

                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2**attempt)  # Exponential backoff
                    logger.warning(
                        "%s request failed (attempt %d/%d): %s. Retrying in %.1fs",
                        self.service_name,
                        attempt + 1,
                        self.max_retries,
                        e,
                        delay,
                    )
                    await asyncio.sleep(delay)

        # All retries failed
        logger.error("%s request failed after %d attempts: %s", self.service_name, self.max_retries, last_error)
        raise last_error or RuntimeError("Request failed")

    async def get(self, path: str, **kwargs) -> httpx.Response:
        """Make GET request."""
        return await self.request("GET", path, **kwargs)

    async def post(self, path: str, **kwargs) -> httpx.Response:
        """Make POST request."""
        return await self.request("POST", path, **kwargs)

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
