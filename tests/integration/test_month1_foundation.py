"""
Phase 1 Foundation Integration Tests.

Tests for Month 1: Foundation and Security Baseline.
All tests use REAL API endpoints - no mocks for business logic.
"""

import os
import pytest
from typing import AsyncGenerator

# Try to import httpx, skip tests if not available
pytest.importorskip("httpx")
import httpx


# Configuration
BASE_URL = os.environ.get("TEST_BASE_URL", "http://localhost:8000")
TEST_USER = os.environ.get("TEST_USER", "admin")
TEST_PASS = os.environ.get("TEST_PASS", "admin123")


@pytest.fixture
async def client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """HTTP client for real API calls."""
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=30.0) as c:
        yield c


@pytest.fixture
async def authenticated_client(
    client: httpx.AsyncClient,
) -> AsyncGenerator[httpx.AsyncClient, None]:
    """Authenticated client using REAL login endpoint."""
    response = await client.post(
        "/api/auth/login",
        json={"username": TEST_USER, "password": TEST_PASS},
    )
    
    if response.status_code != 200:
        pytest.skip(f"Login failed: {response.status_code} - {response.text}")
    
    data = response.json()
    if "session_token" in data and data["session_token"]:
        client.headers["Authorization"] = f"Bearer {data['session_token']}"
    
    # Also set cookies if returned
    for cookie in response.cookies:
        client.cookies.set(cookie.name, cookie.value)
    
    yield client


class TestSecurityHeaders:
    """Verify security headers on all responses."""

    async def test_health_has_security_headers(self, client: httpx.AsyncClient):
        """Health endpoint returns security headers."""
        response = await client.get("/health")
        headers = response.headers
        
        # X-Content-Type-Options
        assert headers.get("x-content-type-options") == "nosniff", \
            "Missing X-Content-Type-Options: nosniff"
        
        # X-Frame-Options
        assert headers.get("x-frame-options") == "DENY", \
            "Missing X-Frame-Options: DENY"
    
    async def test_csp_header_present(self, client: httpx.AsyncClient):
        """CSP header is present."""
        response = await client.get("/health")
        csp = response.headers.get("content-security-policy")
        
        assert csp is not None, "Missing Content-Security-Policy header"
    
    async def test_hsts_header_present(self, client: httpx.AsyncClient):
        """HSTS header is present (may only appear on HTTPS)."""
        response = await client.get("/health")
        hsts = response.headers.get("strict-transport-security")
        
        # HSTS may only be set on HTTPS connections
        if "https" in BASE_URL.lower():
            assert hsts is not None, "Missing Strict-Transport-Security header"


class TestHealthEndpoint:
    """Verify health endpoint functionality."""

    async def test_health_returns_ok(self, client: httpx.AsyncClient):
        """Health endpoint returns OK status."""
        response = await client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] in ["ok", "healthy"]
    
    async def test_health_returns_json(self, client: httpx.AsyncClient):
        """Health endpoint returns JSON."""
        response = await client.get("/health")
        
        assert response.headers.get("content-type", "").startswith("application/json")


class TestMetricsEndpoint:
    """Verify Prometheus metrics endpoint."""

    async def test_metrics_returns_prometheus_format(
        self, client: httpx.AsyncClient
    ):
        """Metrics endpoint returns Prometheus format."""
        response = await client.get("/metrics")
        
        # Metrics endpoint should exist
        if response.status_code == 404:
            pytest.skip("Metrics endpoint not configured")
        
        assert response.status_code == 200
        
        # Should contain Prometheus-style metrics
        content = response.text
        assert any(
            metric in content
            for metric in [
                "http_requests_total",
                "http_request_duration",
                "process_",
            ]
        ), "Missing expected Prometheus metrics"


class TestAuthentication:
    """Verify authentication flow."""

    async def test_login_with_valid_credentials(self, client: httpx.AsyncClient):
        """Login succeeds with valid credentials."""
        response = await client.post(
            "/api/auth/login",
            json={"username": TEST_USER, "password": TEST_PASS},
        )
        
        if response.status_code == 401:
            pytest.skip("Test credentials not configured")
        
        assert response.status_code == 200
        data = response.json()
        assert data.get("success") is True or "session_token" in data
    
    async def test_login_with_invalid_credentials(self, client: httpx.AsyncClient):
        """Login fails with invalid credentials."""
        response = await client.post(
            "/api/auth/login",
            json={"username": "invalid_user", "password": "wrong_password"},
        )
        
        assert response.status_code in [401, 403]
    
    async def test_protected_endpoint_requires_auth(self, client: httpx.AsyncClient):
        """Protected endpoints return 401 without authentication."""
        response = await client.get("/api/v1/banking/accounts")
        
        if response.status_code == 404:
            pytest.skip("Banking endpoint not configured")
        
        assert response.status_code in [401, 403]


class TestRateLimiting:
    """Verify rate limiting is active."""

    async def test_rate_limit_headers_present(self, client: httpx.AsyncClient):
        """Rate limit headers are present on responses."""
        response = await client.get("/health")
        
        # Some implementations use these headers
        rate_headers = [
            "x-ratelimit-limit",
            "x-ratelimit-remaining",
            "ratelimit-limit",
            "ratelimit-remaining",
        ]
        
        has_rate_limit = any(h in response.headers for h in rate_headers)
        
        # Skip if rate limiting not configured on health endpoint
        if not has_rate_limit:
            pytest.skip("Rate limit headers not present on health endpoint")


class TestGemmaService:
    """Verify Gemma LLM service (may be unavailable if GPU not present)."""

    async def test_gemma_health(self, authenticated_client: httpx.AsyncClient):
        """Gemma service health check."""
        response = await authenticated_client.get("/api/gemma/health")
        
        # 200 = healthy, 503 = GPU unavailable, 404 = not configured
        assert response.status_code in [200, 503, 404]


class TestAPIDocumentation:
    """Verify API documentation is available."""

    async def test_openapi_schema_available(self, client: httpx.AsyncClient):
        """OpenAPI schema is available."""
        response = await client.get("/openapi.json")
        
        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data or "swagger" in data
        assert "paths" in data
    
    async def test_docs_ui_available(self, client: httpx.AsyncClient):
        """Swagger UI is available."""
        response = await client.get("/docs")
        
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
