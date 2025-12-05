"""
GPU Client for ML Service

Provides GPU coordination for ML engines that can benefit from GPU acceleration.
Communicates with the GPU Coordinator service using the same protocol as Gemma.

Engines that can use GPU:
- Mirror (CTGAN) - PyTorch GAN for synthetic data
- Galileo (GNN) - Graph Neural Networks
- Titan (future) - XGBoost/LightGBM GPU modes

Architecture:
- GPU Coordinator owns the lock (services/queue-service)
- Transcription owns GPU by default (runs 24/7)
- ML engines request GPU on-demand, like Gemma
- Uses Redis Pub/Sub for coordination signals

Author: Enterprise Analytics Team
Version: 1.0.0
"""

import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any

import httpx

logger = logging.getLogger(__name__)


class GPURequestPriority(Enum):
    """Priority levels for GPU requests"""
    IMMEDIATE = 1   # User-facing: predictions, analysis (same as Gemma)
    BACKGROUND = 2  # Batch jobs: training, large synthesis


@dataclass
class GPUAcquisitionResult:
    """Result of a GPU acquisition attempt"""
    acquired: bool
    task_id: Optional[str] = None
    error: Optional[str] = None
    acquisition_time_ms: float = 0.0
    
    @property
    def success(self) -> bool:
        return self.acquired


class GPUClientError(Exception):
    """Base exception for GPU client errors"""
    pass


class GPUAcquisitionTimeout(GPUClientError):
    """GPU acquisition timed out"""
    pass


class GPUCoordinatorUnavailable(GPUClientError):
    """GPU Coordinator service is unavailable"""
    pass


class GPUClient:
    """
    Async client for GPU coordination in ML Service.
    
    Provides:
    - request_gpu(): Acquire GPU from coordinator
    - release_gpu(): Return GPU to transcription
    - gpu_context(): Context manager for automatic acquire/release
    - Fallback to CPU if GPU unavailable
    
    Usage:
        gpu_client = GPUClient()
        
        # Manual acquire/release
        result = await gpu_client.request_gpu("mirror")
        if result.acquired:
            # Use GPU
            await gpu_client.release_gpu()
        
        # Or use context manager (recommended)
        async with gpu_client.gpu_context("mirror") as acquired:
            if acquired:
                # GPU is available, use it
            else:
                # Fallback to CPU
    
    Thread Safety: This client is NOT thread-safe. Create one per async context.
    """
    
    def __init__(
        self,
        coordinator_url: Optional[str] = None,
        timeout_seconds: float = 15.0,
        retry_attempts: int = 2,
        fallback_to_cpu: bool = True,
        service_auth: Any = None
    ):
        """
        Initialize GPU client.
        
        Args:
            coordinator_url: GPU coordinator URL (default: from env)
            timeout_seconds: Timeout for GPU acquisition
            retry_attempts: Number of retry attempts on failure
            fallback_to_cpu: If True, silently fall back to CPU on failure
            service_auth: ServiceAuth instance for token generation
        """
        self.coordinator_url = coordinator_url or os.getenv(
            "GPU_COORDINATOR_URL", 
            "http://gpu-coordinator:8002"
        )
        self.timeout_seconds = timeout_seconds
        self.retry_attempts = retry_attempts
        self.fallback_to_cpu = fallback_to_cpu
        self.service_auth = service_auth
        
        # State
        self._current_task_id: Optional[str] = None
        self._acquired: bool = False
        self._acquisition_start: Optional[float] = None
        
        # HTTP client with connection pooling
        self._http_client: Optional[httpx.AsyncClient] = None
        
        logger.info(
            f"GPUClient initialized: coordinator={self.coordinator_url}, "
            f"timeout={timeout_seconds}s, fallback_to_cpu={fallback_to_cpu}"
        )
    
    def _get_service_headers(self) -> Dict[str, str]:
        """Get service authentication headers"""
        headers = {
            "X-Service-ID": "ml-service",
            "X-Request-Source": "ml-engine"
        }
        
        if self.service_auth:
            try:
                token = self.service_auth.create_token(expires_in=60, aud="internal")
                headers["X-Service-Token"] = token
            except Exception as e:
                logger.error(f"Failed to create service JWT: {e}")
        
        return headers

    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client with connection pooling"""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout_seconds),
                limits=httpx.Limits(max_connections=5)
            )
        return self._http_client
    
    async def close(self):
        """Close HTTP client and release resources"""
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()
            self._http_client = None
    
    def _generate_task_id(self, engine_name: str) -> str:
        """Generate unique task ID for GPU request"""
        return f"ml-{engine_name}-{uuid.uuid4().hex[:8]}"
    
    async def check_coordinator_health(self) -> bool:
        """
        Check if GPU coordinator is available.
        
        Returns:
            True if coordinator is healthy
        """
        try:
            client = await self._get_http_client()
            response = await client.get(
                f"{self.coordinator_url}/health",
                timeout=5.0
            )
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"GPU Coordinator health check failed: {e}")
            return False
    
    async def get_gpu_status(self) -> Dict[str, Any]:
        """
        Get current GPU status from coordinator.
        
        Returns:
            Status dict with current owner, state, etc.
        """
        try:
            client = await self._get_http_client()
            response = await client.get(f"{self.coordinator_url}/status")
            if response.status_code == 200:
                return response.json()
            return {"error": f"Status code: {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}
    
    async def request_gpu(
        self,
        engine_name: str,
        priority: GPURequestPriority = GPURequestPriority.IMMEDIATE
    ) -> GPUAcquisitionResult:
        """
        Request GPU from coordinator.
        
        This will:
        1. Signal transcription to pause
        2. Wait for acknowledgment
        3. Grant exclusive GPU access to this service
        
        Args:
            engine_name: Name of the engine requesting GPU (for logging)
            priority: Request priority level
            
        Returns:
            GPUAcquisitionResult with acquisition status
        """
        if self._acquired:
            logger.warning(f"GPU already acquired for task {self._current_task_id}")
            return GPUAcquisitionResult(
                acquired=True,
                task_id=self._current_task_id
            )
        
        task_id = self._generate_task_id(engine_name)
        start_time = time.time()
        
        logger.info(f"[GPU] Requesting GPU for {engine_name} (task_id={task_id}, priority={priority.name})")
        
        for attempt in range(self.retry_attempts + 1):
            try:
                client = await self._get_http_client()
                
                # Use the same endpoint as Gemma
                response = await client.post(
                    f"{self.coordinator_url}/gemma/request",
                    json={
                        "task_id": task_id,
                        "messages": [],  # Not used for ML
                        "context": [],
                        "max_tokens": 0,
                        "temperature": 0.0
                    },
                    headers=self._get_service_headers()
                )
                
                if response.status_code == 200:
                    acquisition_time = (time.time() - start_time) * 1000
                    self._current_task_id = task_id
                    self._acquired = True
                    self._acquisition_start = time.time()
                    
                    logger.info(
                        f"[GPU] ✅ Acquired GPU for {engine_name} "
                        f"(task_id={task_id}, time={acquisition_time:.1f}ms)"
                    )
                    
                    return GPUAcquisitionResult(
                        acquired=True,
                        task_id=task_id,
                        acquisition_time_ms=acquisition_time
                    )
                else:
                    error_detail = response.text[:200]
                    logger.warning(
                        f"[GPU] Request failed (attempt {attempt+1}/{self.retry_attempts+1}): "
                        f"status={response.status_code}, detail={error_detail}"
                    )
                    
            except httpx.TimeoutException:
                logger.warning(f"[GPU] Request timeout (attempt {attempt+1})")
            except httpx.ConnectError:
                logger.warning(f"[GPU] Coordinator unreachable (attempt {attempt+1})")
            except Exception as e:
                logger.error(f"[GPU] Unexpected error: {e}")
            
            # Wait before retry
            if attempt < self.retry_attempts:
                await asyncio.sleep(0.5 * (attempt + 1))
        
        # All attempts failed
        acquisition_time = (time.time() - start_time) * 1000
        error_msg = f"Failed to acquire GPU after {self.retry_attempts + 1} attempts"
        
        if self.fallback_to_cpu:
            logger.warning(f"[GPU] {error_msg}, falling back to CPU")
        else:
            logger.error(f"[GPU] {error_msg}")
        
        return GPUAcquisitionResult(
            acquired=False,
            error=error_msg,
            acquisition_time_ms=acquisition_time
        )
    
    async def release_gpu(self) -> bool:
        """
        Release GPU back to transcription.
        
        Returns:
            True if released successfully
        """
        if not self._acquired or not self._current_task_id:
            logger.debug("[GPU] No GPU to release (not acquired)")
            return True
        
        task_id = self._current_task_id
        duration = time.time() - self._acquisition_start if self._acquisition_start else 0
        
        logger.info(f"[GPU] Releasing GPU (task_id={task_id}, duration={duration:.2f}s)")
        
        try:
            client = await self._get_http_client()
            
            response = await client.post(
                f"{self.coordinator_url}/gemma/release/{task_id}",
                json={},
                headers=self._get_service_headers()
            )
            
            self._acquired = False
            self._current_task_id = None
            self._acquisition_start = None
            
            if response.status_code == 200:
                logger.info(f"[GPU] ✅ Released GPU (task_id={task_id})")
                return True
            else:
                logger.warning(f"[GPU] Release returned status {response.status_code}")
                return True  # Consider released anyway
                
        except Exception as e:
            logger.error(f"[GPU] Failed to release GPU: {e}")
            # Reset state anyway to prevent stuck state
            self._acquired = False
            self._current_task_id = None
            self._acquisition_start = None
            return False
    
    def _get_service_headers(self) -> Dict[str, str]:
        """Get service authentication headers"""
        headers = {
            "X-Service-ID": "ml-service",
            "X-Request-Source": "ml-engine"
        }
        
        if self.service_auth:
            try:
                token = self.service_auth.create_token(expires_in=60, aud="internal")
                headers["X-Service-Token"] = token
            except Exception as e:
                logger.error(f"Failed to create service JWT: {e}")
        
        return headers
    
    @asynccontextmanager
    async def gpu_context(
        self,
        engine_name: str,
        priority: GPURequestPriority = GPURequestPriority.IMMEDIATE
    ):
        """
        Context manager for GPU acquisition and release.
        
        Automatically releases GPU on exit, even if an exception occurs.
        
        Args:
            engine_name: Name of the engine requesting GPU
            priority: Request priority level
            
        Yields:
            bool: True if GPU was acquired, False if fell back to CPU
            
        Example:
            async with gpu_client.gpu_context("mirror") as acquired:
                if acquired:
                    synthesizer = CTGANSynthesizer(cuda=True)
                else:
                    synthesizer = CTGANSynthesizer(cuda=False)
        """
        result = await self.request_gpu(engine_name, priority)
        
        try:
            yield result.acquired
        finally:
            if result.acquired:
                await self.release_gpu()
    
    @property
    def is_acquired(self) -> bool:
        """Check if GPU is currently acquired"""
        return self._acquired
    
    @property
    def current_task_id(self) -> Optional[str]:
        """Get current task ID if GPU is acquired"""
        return self._current_task_id


# Singleton instance for the service
_gpu_client: Optional[GPUClient] = None


def get_gpu_client(service_auth: Any = None) -> GPUClient:
    """
    Get or create the GPU client singleton.
    
    Returns:
        GPUClient instance
    """
    global _gpu_client
    if _gpu_client is None:
        _gpu_client = GPUClient(service_auth=service_auth)
    elif service_auth is not None and _gpu_client.service_auth is None:
        # Update auth if provided later
        _gpu_client.service_auth = service_auth
    return _gpu_client


async def init_gpu_client(service_auth: Any = None) -> GPUClient:
    """
    Initialize the GPU client and check coordinator health.
    
    Returns:
        Initialized GPUClient
    """
    client = get_gpu_client(service_auth)
    
    # Check coordinator health
    healthy = await client.check_coordinator_health()
    if healthy:
        logger.info("✅ GPU Coordinator is available")
    else:
        logger.warning("⚠️ GPU Coordinator unavailable, will fall back to CPU")
    
    return client


async def shutdown_gpu_client():
    """Shutdown the GPU client and release resources"""
    global _gpu_client
    if _gpu_client:
        # Release GPU if still acquired
        if _gpu_client.is_acquired:
            await _gpu_client.release_gpu()
        await _gpu_client.close()
        _gpu_client = None
        logger.info("GPU Client shutdown complete")
