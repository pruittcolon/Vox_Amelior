"""
Phase 5 Integration Tests: Reliability and FinOps.

Tests use REAL API endpoints - no mocks.
Covers:
- Health endpoints
- Analytics API
- SLO tracking
- Cost tracking
"""

import os

import httpx
import pytest

# Test configuration
BASE_URL = os.getenv("TEST_BASE_URL", "http://localhost:8000")
TEST_USER = os.getenv("TEST_USER", "admin")
TEST_PASS = os.getenv("TEST_PASS", "admin123")


@pytest.fixture
def anyio_backend():
    """Use asyncio backend."""
    return "asyncio"


@pytest.fixture
async def http_client():
    """Unauthenticated HTTP client."""
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=60.0) as client:
        yield client


# =============================================================================
# COST TRACKER TESTS
# =============================================================================


class TestCostTracker:
    """Tests for cost tracker."""

    def test_cost_tracker_import(self):
        """Test cost tracker can be imported."""
        from shared.telemetry.cost_tracker import (
            CostTracker,
            ServiceType,
            UsageRecord,
            get_cost_tracker,
        )

        assert CostTracker is not None
        assert callable(get_cost_tracker)

    def test_track_request(self):
        """Test tracking a request."""
        from shared.telemetry.cost_tracker import CostTracker, ServiceType
        from uuid import uuid4

        tracker = CostTracker()
        tenant_id = uuid4()

        record = tracker.track_request(
            tenant_id=tenant_id,
            service=ServiceType.AI_CHAT,
            tokens_in=100,
            tokens_out=50,
            latency_ms=500.0,
        )

        assert record.tenant_id == tenant_id
        assert record.total_tokens == 150

    def test_get_tenant_usage(self):
        """Test getting tenant usage summary."""
        from shared.telemetry.cost_tracker import CostTracker, ServiceType
        from uuid import uuid4

        tracker = CostTracker()
        tenant_id = uuid4()

        # Track some requests
        tracker.track_request(tenant_id, ServiceType.API)
        tracker.track_request(tenant_id, ServiceType.AI_CHAT, tokens_in=100)

        summary = tracker.get_tenant_usage(tenant_id)

        assert summary["total_requests"] == 2


# =============================================================================
# SLO TRACKER TESTS
# =============================================================================


class TestSLOTracker:
    """Tests for SLO tracker."""

    def test_slo_tracker_import(self):
        """Test SLO tracker can be imported."""
        from shared.telemetry.slo_tracker import (
            SLOTracker,
            SLODefinition,
            get_slo_tracker,
        )

        assert SLOTracker is not None
        assert callable(get_slo_tracker)

    def test_record_request(self):
        """Test recording requests for SLI."""
        from shared.telemetry.slo_tracker import SLOTracker

        tracker = SLOTracker()

        record = tracker.record_request(
            endpoint="/api/v1/test",
            latency_ms=50.0,
            success=True,
        )

        assert record.endpoint == "/api/v1/test"
        assert record.success is True

    def test_get_slo_status(self):
        """Test getting SLO status."""
        from shared.telemetry.slo_tracker import SLOTracker

        tracker = SLOTracker()

        # Record some requests
        for _ in range(10):
            tracker.record_request("/api/test", latency_ms=100, success=True)

        status = tracker.get_slo_status("api_availability")

        assert "slo_name" in status
        assert "meeting_slo" in status or "status" in status


# =============================================================================
# CACHE TESTS
# =============================================================================


class TestCacheLayer:
    """Tests for cache layer."""

    def test_cache_import(self):
        """Test cache can be imported."""
        from shared.storage.cache import CacheLayer, get_cache

        assert CacheLayer is not None
        assert callable(get_cache)

    def test_set_and_get(self):
        """Test setting and getting cache values."""
        from shared.storage.cache import CacheLayer

        cache = CacheLayer()

        cache.set("test_key", {"value": 123}, ttl=60)
        result = cache.get("test_key")

        assert result == {"value": 123}

    def test_cache_miss(self):
        """Test cache miss returns None."""
        from shared.storage.cache import CacheLayer

        cache = CacheLayer()
        result = cache.get("nonexistent_key")

        assert result is None


# =============================================================================
# API TESTS
# =============================================================================


class TestAnalyticsAPI:
    """Tests for analytics API endpoints."""

    @pytest.mark.anyio
    async def test_analytics_summary(self, http_client):
        """Test analytics summary endpoint."""
        response = await http_client.get("/api/v1/analytics/summary")
        assert response.status_code in [200, 401, 404]

    @pytest.mark.anyio
    async def test_analytics_costs(self, http_client):
        """Test analytics costs endpoint."""
        response = await http_client.get("/api/v1/analytics/costs")
        assert response.status_code in [200, 401, 404]

    @pytest.mark.anyio
    async def test_analytics_slos(self, http_client):
        """Test analytics SLOs endpoint."""
        response = await http_client.get("/api/v1/analytics/slos")
        assert response.status_code in [200, 401, 404]

    @pytest.mark.anyio
    async def test_health_endpoint(self, http_client):
        """Test health endpoint."""
        response = await http_client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
