"""
GPU Coordinator Client

Async HTTP client for GPU coordination used by Gemma, ML, and other services.
Provides acquire/release methods and a context manager for session management.

Author: Enterprise Analytics Team
Version: 1.0.0
"""

import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import httpx

from shared.gpu.protocol import (
    GPUAcquireRequest,
    GPUAcquireResponse,
    GPUPriority,
    GPUReleaseRequest,
    GPUReleaseResponse,
    GPUStatusResponse,
)

logger = logging.getLogger(__name__)


class GPUClientError(Exception):
    """Base exception for GPU client errors."""
    pass


class GPUAcquisitionTimeout(GPUClientError):
    """GPU acquisition timed out."""
    pass


class GPUCoordinatorUnavailable(GPUClientError):
    """GPU Coordinator service is unavailable."""
    pass


class GPUCoordinatorClient:
    """
    Async client for GPU coordination.
    
    Provides:
    - acquire_gpu(): Request GPU from coordinator
    - release_gpu(): Return GPU to transcription
    - gpu_session(): Context manager for automatic acquire/release
    - get_status(): Get current GPU state
    
    Usage:
        client = GPUCoordinatorClient(service_auth=my_auth)
        
        # Option 1: Manual acquire/release
        result = await client.acquire_gpu("session-123", "gemma-service")
        if result.success:
            # Use GPU
            await client.release_gpu("session-123", "gemma-service")
        
        # Option 2: Context manager (recommended)
        async with client.gpu_session("session-123", "gemma-service") as acquired:
            if acquired:
                # GPU is available
                pass
    
    Thread Safety: NOT thread-safe. Create one instance per async context.
    """
    
    def __init__(
        self,
        coordinator_url: str | None = None,
        timeout_seconds: float = 30.0,
        retry_attempts: int = 2,
        fallback_to_cpu: bool = True,
        service_auth: Any = None,
    ) -> None:
        """
        Initialize GPU coordinator client.
        
        Args:
            coordinator_url: Coordinator URL (default: from GPU_COORDINATOR_URL env)
            timeout_seconds: Request timeout in seconds
            retry_attempts: Number of retry attempts on failure
            fallback_to_cpu: If True, silently return False instead of raising
            service_auth: ServiceAuth instance for JWT token generation
        """
        self.coordinator_url = coordinator_url or os.getenv(
            "GPU_COORDINATOR_URL", "http://gpu-coordinator:8002"
        )
        self.timeout_seconds = timeout_seconds
        self.retry_attempts = retry_attempts
        self.fallback_to_cpu = fallback_to_cpu
        self.service_auth = service_auth
        
        # State tracking
        self._current_session_id: str | None = None
        self._current_requester: str | None = None
        self._acquired: bool = False
        self._acquisition_start: float | None = None
        
        # HTTP client (lazy initialization)
        self._http_client: httpx.AsyncClient | None = None
        
        logger.info(
            f"[GPU-CLIENT] Initialized: url={self.coordinator_url}, "
            f"timeout={timeout_seconds}s, fallback={fallback_to_cpu}"
        )
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client with connection pooling."""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout_seconds),
                limits=httpx.Limits(max_connections=5, max_keepalive_connections=2),
            )
        return self._http_client
    
    def _get_headers(self) -> dict[str, str]:
        """Get request headers including JWT token if available."""
        headers = {
            "Content-Type": "application/json",
            "X-Request-ID": str(uuid.uuid4()),
        }
        
        if self.service_auth:
            try:
                token = self.service_auth.create_token(expires_in=60, aud="internal")
                headers["X-Service-Token"] = token
            except Exception as e:
                logger.warning(f"[GPU-CLIENT] Failed to create JWT: {e}")
        
        return headers
    
    async def close(self) -> None:
        """Close HTTP client and release resources."""
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()
            self._http_client = None
        logger.debug("[GPU-CLIENT] Client closed")
    
    async def health_check(self) -> bool:
        """
        Check if coordinator is healthy.
        
        Returns:
            True if coordinator is healthy and reachable
        """
        try:
            client = await self._get_client()
            response = await client.get(
                f"{self.coordinator_url}/health",
                timeout=5.0,
            )
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"[GPU-CLIENT] Health check failed: {e}")
            return False
    
    async def get_status(self) -> GPUStatusResponse | None:
        """
        Get current GPU coordinator status.
        
        Returns:
            GPUStatusResponse or None if unavailable
        """
        try:
            client = await self._get_client()
            response = await client.get(
                f"{self.coordinator_url}/gpu/state",
                headers=self._get_headers(),
            )
            if response.status_code == 200:
                return GPUStatusResponse.model_validate(response.json())
            
            # Fallback to legacy endpoint
            response = await client.get(
                f"{self.coordinator_url}/status",
                headers=self._get_headers(),
            )
            if response.status_code == 200:
                data = response.json()
                lock_status = data.get("lock_status", {})
                return GPUStatusResponse(
                    owner=lock_status.get("current_owner", "transcription"),
                    session_id=lock_status.get("current_task"),
                    state=lock_status.get("state", "transcription"),
                    redis_connected=True,
                    postgres_connected=True,
                )
            return None
        except Exception as e:
            logger.warning(f"[GPU-CLIENT] Status check failed: {e}")
            return None
    
    async def acquire_gpu(
        self,
        session_id: str,
        requester: str,
        priority: GPUPriority = GPUPriority.IMMEDIATE,
        timeout_ms: int = 30000,
    ) -> GPUAcquireResponse:
        """
        Request GPU from coordinator.
        
        This will:
        1. Signal transcription to pause
        2. Wait for acknowledgment
        3. Grant exclusive GPU access to this service
        
        Args:
            session_id: Unique session identifier
            requester: Service name (e.g., "gemma-service")
            priority: Request priority level
            timeout_ms: Timeout in milliseconds
            
        Returns:
            GPUAcquireResponse with acquisition status
        """
        if self._acquired:
            logger.warning(
                f"[GPU-CLIENT] GPU already acquired for session {self._current_session_id}"
            )
            return GPUAcquireResponse(
                success=True,
                session_id=self._current_session_id or session_id,
                acquired_at=None,
            )
        
        start_time = time.time()
        request = GPUAcquireRequest(
            session_id=session_id,
            requester=requester,
            priority=priority,
            timeout_ms=timeout_ms,
        )
        
        logger.info(
            f"[GPU-CLIENT] Requesting GPU: session={session_id}, "
            f"requester={requester}, priority={priority.name}"
        )
        
        for attempt in range(self.retry_attempts + 1):
            try:
                client = await self._get_client()
                
                # Try new endpoint first, fall back to legacy
                response = await client.post(
                    f"{self.coordinator_url}/gpu/acquire",
                    json=request.model_dump(mode="json"),
                    headers=self._get_headers(),
                )
                
                # Fallback to legacy endpoint if new one not found
                if response.status_code == 404:
                    response = await client.post(
                        f"{self.coordinator_url}/gemma/request",
                        json={
                            "task_id": session_id,
                            "messages": [{"role": "system", "content": "gpu_acquire"}],
                            "max_tokens": 0,
                            "temperature": 0.0,
                        },
                        headers=self._get_headers(),
                    )
                
                if response.status_code == 200:
                    wait_time = (time.time() - start_time) * 1000
                    self._current_session_id = session_id
                    self._current_requester = requester
                    self._acquired = True
                    self._acquisition_start = time.time()
                    
                    logger.info(
                        f"[GPU-CLIENT] GPU acquired: session={session_id}, "
                        f"wait={wait_time:.1f}ms"
                    )
                    
                    return GPUAcquireResponse(
                        success=True,
                        session_id=session_id,
                        acquired_at=None,
                        wait_time_ms=wait_time,
                    )
                else:
                    error = response.text[:200]
                    logger.warning(
                        f"[GPU-CLIENT] Acquire failed (attempt {attempt + 1}): "
                        f"status={response.status_code}, error={error}"
                    )
                    
            except httpx.TimeoutException:
                logger.warning(f"[GPU-CLIENT] Acquire timeout (attempt {attempt + 1})")
            except httpx.ConnectError:
                logger.warning(f"[GPU-CLIENT] Coordinator unreachable (attempt {attempt + 1})")
            except Exception as e:
                logger.error(f"[GPU-CLIENT] Unexpected error: {e}")
            
            # Wait before retry with exponential backoff
            if attempt < self.retry_attempts:
                await asyncio.sleep(0.5 * (attempt + 1))
        
        # All attempts failed
        wait_time = (time.time() - start_time) * 1000
        error_msg = f"Failed to acquire GPU after {self.retry_attempts + 1} attempts"
        
        if self.fallback_to_cpu:
            logger.warning(f"[GPU-CLIENT] {error_msg}, falling back to CPU")
        else:
            logger.error(f"[GPU-CLIENT] {error_msg}")
        
        return GPUAcquireResponse(
            success=False,
            session_id=session_id,
            error=error_msg,
            wait_time_ms=wait_time,
        )
    
    async def release_gpu(
        self,
        session_id: str | None = None,
        requester: str | None = None,
        result: dict[str, Any] | None = None,
    ) -> GPUReleaseResponse:
        """
        Release GPU back to transcription.
        
        Args:
            session_id: Session to release (default: current session)
            requester: Service releasing GPU (default: current requester)
            result: Optional result/status from the session
            
        Returns:
            GPUReleaseResponse with release status
        """
        session_id = session_id or self._current_session_id
        requester = requester or self._current_requester
        
        if not self._acquired or not session_id:
            logger.debug("[GPU-CLIENT] No GPU to release (not acquired)")
            return GPUReleaseResponse(
                success=True,
                session_id=session_id or "unknown",
            )
        
        duration = time.time() - self._acquisition_start if self._acquisition_start else 0
        
        logger.info(
            f"[GPU-CLIENT] Releasing GPU: session={session_id}, duration={duration:.2f}s"
        )
        
        request = GPUReleaseRequest(
            session_id=session_id,
            requester=requester or "unknown",
            result=result or {},
        )
        
        try:
            client = await self._get_client()
            
            # Try new endpoint first, fall back to legacy
            response = await client.post(
                f"{self.coordinator_url}/gpu/release",
                json=request.model_dump(mode="json"),
                headers=self._get_headers(),
            )
            
            # Fallback to legacy endpoint
            if response.status_code == 404:
                response = await client.post(
                    f"{self.coordinator_url}/gemma/release/{session_id}",
                    json=result or {},
                    headers=self._get_headers(),
                )
            
            # Reset state regardless of response
            self._acquired = False
            self._current_session_id = None
            self._current_requester = None
            self._acquisition_start = None
            
            if response.status_code == 200:
                logger.info(f"[GPU-CLIENT] GPU released: session={session_id}")
                return GPUReleaseResponse(
                    success=True,
                    session_id=session_id,
                    session_duration_ms=duration * 1000,
                )
            else:
                logger.warning(
                    f"[GPU-CLIENT] Release returned status {response.status_code}"
                )
                return GPUReleaseResponse(
                    success=True,  # Consider released anyway
                    session_id=session_id,
                    session_duration_ms=duration * 1000,
                )
                
        except Exception as e:
            logger.error(f"[GPU-CLIENT] Failed to release GPU: {e}")
            # Reset state to prevent stuck state
            self._acquired = False
            self._current_session_id = None
            self._current_requester = None
            self._acquisition_start = None
            return GPUReleaseResponse(
                success=False,
                session_id=session_id,
            )
    
    @asynccontextmanager
    async def gpu_session(
        self,
        session_id: str,
        requester: str,
        priority: GPUPriority = GPUPriority.IMMEDIATE,
    ):
        """
        Context manager for GPU session management.
        
        Automatically releases GPU on exit, even if an exception occurs.
        
        Args:
            session_id: Unique session identifier
            requester: Service name
            priority: Request priority
            
        Yields:
            bool: True if GPU was acquired, False if fell back to CPU
            
        Example:
            async with client.gpu_session("session-123", "gemma") as acquired:
                if acquired:
                    model.to_gpu()
                else:
                    model.to_cpu()
        """
        result = await self.acquire_gpu(session_id, requester, priority)
        
        try:
            yield result.success
        finally:
            if result.success:
                await self.release_gpu(session_id, requester)
    
    @property
    def is_acquired(self) -> bool:
        """Check if GPU is currently acquired."""
        return self._acquired
    
    @property
    def current_session_id(self) -> str | None:
        """Get current session ID if GPU is acquired."""
        return self._current_session_id


# Singleton instance
_gpu_client: GPUCoordinatorClient | None = None


def get_gpu_client(service_auth: Any = None) -> GPUCoordinatorClient:
    """
    Get or create the GPU client singleton.
    
    Args:
        service_auth: Optional ServiceAuth for JWT generation
        
    Returns:
        GPUCoordinatorClient instance
    """
    global _gpu_client
    if _gpu_client is None:
        _gpu_client = GPUCoordinatorClient(service_auth=service_auth)
    elif service_auth is not None and _gpu_client.service_auth is None:
        _gpu_client.service_auth = service_auth
    return _gpu_client


async def init_gpu_client(service_auth: Any = None) -> GPUCoordinatorClient:
    """
    Initialize GPU client and verify coordinator health.
    
    Args:
        service_auth: Optional ServiceAuth for JWT generation
        
    Returns:
        Initialized GPUCoordinatorClient
    """
    client = get_gpu_client(service_auth)
    
    healthy = await client.health_check()
    if healthy:
        logger.info("[GPU-CLIENT] Coordinator is available")
    else:
        logger.warning("[GPU-CLIENT] Coordinator unavailable, will fall back to CPU")
    
    return client


async def shutdown_gpu_client() -> None:
    """Shutdown GPU client and release resources."""
    global _gpu_client
    if _gpu_client:
        if _gpu_client.is_acquired:
            await _gpu_client.release_gpu()
        await _gpu_client.close()
        _gpu_client = None
        logger.info("[GPU-CLIENT] Shutdown complete")
