"""
Unit Tests for GPU Client

Tests GPU coordination functionality with mocked coordinator responses.
Uses pytest-asyncio for async test support.

Author: Enterprise Analytics Team
"""

import asyncio
import os

# Import the module under test
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from core.gpu_client import GPUClient, get_gpu_client

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def gpu_client():
    """Create a fresh GPU client for each test"""
    client = GPUClient(
        coordinator_url="http://test-coordinator:8002", timeout_seconds=5.0, retry_attempts=1, fallback_to_cpu=True
    )
    return client


@pytest.fixture
def mock_http_response():
    """Factory for creating mock HTTP responses"""

    def _create_response(status_code: int, json_data: dict = None):
        response = MagicMock(spec=httpx.Response)
        response.status_code = status_code
        response.json.return_value = json_data or {}
        response.text = str(json_data) if json_data else ""
        return response

    return _create_response


# ============================================================================
# Unit Tests: GPUClient
# ============================================================================


class TestGPUClientInitialization:
    """Tests for GPU client initialization"""

    def test_default_initialization(self):
        """Test default initialization uses environment variables"""
        client = GPUClient()
        assert client.coordinator_url is not None
        assert client.timeout_seconds == 15.0
        assert client.retry_attempts == 2
        assert client.fallback_to_cpu is True

    def test_custom_initialization(self):
        """Test custom initialization parameters"""
        client = GPUClient(
            coordinator_url="http://custom:9000", timeout_seconds=30.0, retry_attempts=5, fallback_to_cpu=False
        )
        assert client.coordinator_url == "http://custom:9000"
        assert client.timeout_seconds == 30.0
        assert client.retry_attempts == 5
        assert client.fallback_to_cpu is False

    def test_initial_state(self, gpu_client):
        """Test initial state is not acquired"""
        assert gpu_client.is_acquired is False
        assert gpu_client.current_task_id is None


class TestGPUClientTaskIdGeneration:
    """Tests for task ID generation"""

    def test_task_id_format(self, gpu_client):
        """Test task ID follows expected format"""
        task_id = gpu_client._generate_task_id("mirror")
        assert task_id.startswith("ml-mirror-")
        assert len(task_id) == len("ml-mirror-") + 8  # 8 hex chars

    def test_task_id_unique(self, gpu_client):
        """Test each task ID is unique"""
        ids = [gpu_client._generate_task_id("test") for _ in range(100)]
        assert len(set(ids)) == 100


class TestGPUClientAcquisition:
    """Tests for GPU acquisition"""

    @pytest.mark.asyncio
    async def test_successful_acquisition(self, gpu_client, mock_http_response):
        """Test successful GPU acquisition"""
        mock_response = mock_http_response(200, {"status": "gpu_acquired"})

        with patch.object(gpu_client, "_get_http_client") as mock_client:
            mock_async_client = AsyncMock()
            mock_async_client.post.return_value = mock_response
            mock_client.return_value = mock_async_client

            result = await gpu_client.request_gpu("mirror")

            assert result.acquired is True
            assert result.task_id is not None
            assert result.task_id.startswith("ml-mirror-")
            assert result.error is None
            assert gpu_client.is_acquired is True

    @pytest.mark.asyncio
    async def test_failed_acquisition_with_fallback(self, gpu_client, mock_http_response):
        """Test failed acquisition falls back to CPU"""
        mock_response = mock_http_response(500, {"error": "Internal error"})

        with patch.object(gpu_client, "_get_http_client") as mock_client:
            mock_async_client = AsyncMock()
            mock_async_client.post.return_value = mock_response
            mock_client.return_value = mock_async_client

            result = await gpu_client.request_gpu("mirror")

            assert result.acquired is False
            assert result.error is not None
            assert gpu_client.is_acquired is False

    @pytest.mark.asyncio
    async def test_timeout_with_retry(self, gpu_client):
        """Test timeout triggers retry"""
        with patch.object(gpu_client, "_get_http_client") as mock_client:
            mock_async_client = AsyncMock()
            mock_async_client.post.side_effect = httpx.TimeoutException("Timeout")
            mock_client.return_value = mock_async_client

            result = await gpu_client.request_gpu("mirror")

            # Should have retried
            assert mock_async_client.post.call_count == gpu_client.retry_attempts + 1
            assert result.acquired is False

    @pytest.mark.asyncio
    async def test_double_acquisition_returns_existing(self, gpu_client, mock_http_response):
        """Test acquiring when already acquired returns existing task"""
        mock_response = mock_http_response(200, {"status": "gpu_acquired"})

        with patch.object(gpu_client, "_get_http_client") as mock_client:
            mock_async_client = AsyncMock()
            mock_async_client.post.return_value = mock_response
            mock_client.return_value = mock_async_client

            # First acquisition
            result1 = await gpu_client.request_gpu("mirror")
            task_id1 = result1.task_id

            # Second acquisition (should return existing)
            result2 = await gpu_client.request_gpu("mirror")

            assert result2.acquired is True
            assert result2.task_id == task_id1
            # POST should only be called once
            assert mock_async_client.post.call_count == 1


class TestGPUClientRelease:
    """Tests for GPU release"""

    @pytest.mark.asyncio
    async def test_successful_release(self, gpu_client, mock_http_response):
        """Test successful GPU release"""
        # First acquire
        acquire_response = mock_http_response(200, {"status": "gpu_acquired"})
        release_response = mock_http_response(200, {"status": "released"})

        with patch.object(gpu_client, "_get_http_client") as mock_client:
            mock_async_client = AsyncMock()
            mock_async_client.post.side_effect = [acquire_response, release_response]
            mock_client.return_value = mock_async_client

            await gpu_client.request_gpu("mirror")
            assert gpu_client.is_acquired is True

            success = await gpu_client.release_gpu()

            assert success is True
            assert gpu_client.is_acquired is False
            assert gpu_client.current_task_id is None

    @pytest.mark.asyncio
    async def test_release_when_not_acquired(self, gpu_client):
        """Test releasing when not acquired is a no-op"""
        success = await gpu_client.release_gpu()
        assert success is True

    @pytest.mark.asyncio
    async def test_release_clears_state_on_error(self, gpu_client, mock_http_response):
        """Test release clears state even on error"""
        acquire_response = mock_http_response(200, {"status": "gpu_acquired"})

        with patch.object(gpu_client, "_get_http_client") as mock_client:
            mock_async_client = AsyncMock()
            mock_async_client.post.side_effect = [acquire_response, Exception("Network error")]
            mock_client.return_value = mock_async_client

            await gpu_client.request_gpu("mirror")
            await gpu_client.release_gpu()

            # State should be cleared despite error
            assert gpu_client.is_acquired is False


class TestGPUClientContextManager:
    """Tests for GPU context manager"""

    @pytest.mark.asyncio
    async def test_context_manager_success(self, gpu_client, mock_http_response):
        """Test context manager acquires and releases"""
        acquire_response = mock_http_response(200, {"status": "gpu_acquired"})
        release_response = mock_http_response(200, {"status": "released"})

        with patch.object(gpu_client, "_get_http_client") as mock_client:
            mock_async_client = AsyncMock()
            mock_async_client.post.side_effect = [acquire_response, release_response]
            mock_client.return_value = mock_async_client

            async with gpu_client.gpu_context("mirror") as acquired:
                assert acquired is True
                assert gpu_client.is_acquired is True

            # After context, should be released
            assert gpu_client.is_acquired is False

    @pytest.mark.asyncio
    async def test_context_manager_releases_on_exception(self, gpu_client, mock_http_response):
        """Test context manager releases GPU on exception"""
        acquire_response = mock_http_response(200, {"status": "gpu_acquired"})
        release_response = mock_http_response(200, {"status": "released"})

        with patch.object(gpu_client, "_get_http_client") as mock_client:
            mock_async_client = AsyncMock()
            mock_async_client.post.side_effect = [acquire_response, release_response]
            mock_client.return_value = mock_async_client

            with pytest.raises(ValueError):
                async with gpu_client.gpu_context("mirror") as acquired:
                    assert acquired is True
                    raise ValueError("Test exception")

            # Should still release
            assert gpu_client.is_acquired is False

    @pytest.mark.asyncio
    async def test_context_manager_fallback(self, gpu_client, mock_http_response):
        """Test context manager with CPU fallback"""
        mock_response = mock_http_response(500, {"error": "unavailable"})

        with patch.object(gpu_client, "_get_http_client") as mock_client:
            mock_async_client = AsyncMock()
            mock_async_client.post.return_value = mock_response
            mock_client.return_value = mock_async_client

            async with gpu_client.gpu_context("mirror") as acquired:
                assert acquired is False  # Fell back to CPU


class TestGPUClientHealthCheck:
    """Tests for coordinator health check"""

    @pytest.mark.asyncio
    async def test_health_check_success(self, gpu_client, mock_http_response):
        """Test successful health check"""
        mock_response = mock_http_response(200, {"status": "healthy"})

        with patch.object(gpu_client, "_get_http_client") as mock_client:
            mock_async_client = AsyncMock()
            mock_async_client.get.return_value = mock_response
            mock_client.return_value = mock_async_client

            healthy = await gpu_client.check_coordinator_health()
            assert healthy is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self, gpu_client):
        """Test health check on unreachable coordinator"""
        with patch.object(gpu_client, "_get_http_client") as mock_client:
            mock_async_client = AsyncMock()
            mock_async_client.get.side_effect = httpx.ConnectError("Connection refused")
            mock_client.return_value = mock_async_client

            healthy = await gpu_client.check_coordinator_health()
            assert healthy is False


class TestGPUClientSingleton:
    """Tests for singleton pattern"""

    def test_get_gpu_client_returns_same_instance(self):
        """Test singleton returns same instance"""
        # Clear any existing singleton
        import core.gpu_client as gpu_module

        gpu_module._gpu_client = None

        client1 = get_gpu_client()
        client2 = get_gpu_client()

        assert client1 is client2


# ============================================================================
# Integration-style Tests (with mocked coordinator)
# ============================================================================


class TestGPUClientIntegration:
    """Integration-style tests simulating real coordinator interactions"""

    @pytest.mark.asyncio
    async def test_full_workflow_mirror_engine(self, gpu_client, mock_http_response):
        """Test full workflow for Mirror engine GPU usage"""
        acquire_response = mock_http_response(200, {"status": "gpu_acquired"})
        release_response = mock_http_response(200, {"status": "released"})

        with patch.object(gpu_client, "_get_http_client") as mock_client:
            mock_async_client = AsyncMock()
            mock_async_client.post.side_effect = [acquire_response, release_response]
            mock_client.return_value = mock_async_client

            # Simulate Mirror engine workflow
            async with gpu_client.gpu_context("mirror") as use_cuda:
                # Engine would use cuda=use_cuda
                assert use_cuda is True

                # Simulate work
                await asyncio.sleep(0.01)

            # Verify proper cleanup
            assert gpu_client.is_acquired is False

            # Verify correct API calls
            calls = mock_async_client.post.call_args_list
            assert len(calls) == 2

            # First call: acquire
            assert "/gemma/request" in str(calls[0])

            # Second call: release
            assert "/gemma/release/" in str(calls[1])


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
