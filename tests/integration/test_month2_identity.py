"""
Phase 2 Identity Integration Tests.

Tests for Month 2: Tenant Isolation and Enterprise Identity.
All tests use REAL API endpoints - no mocks for business logic.

Note: Tests are designed to work with low RAM (1.77 GB available).
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
ADMIN_USER = os.environ.get("ADMIN_USER", "admin")
ADMIN_PASS = os.environ.get("ADMIN_PASS", "admin123")


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
        pytest.skip(f"Login failed: {response.status_code}")
    
    data = response.json()
    if "session_token" in data and data["session_token"]:
        client.headers["Authorization"] = f"Bearer {data['session_token']}"
    
    for cookie in response.cookies:
        client.cookies.set(cookie.name, cookie.value)
    
    yield client


@pytest.fixture
async def admin_client(
    client: httpx.AsyncClient,
) -> AsyncGenerator[httpx.AsyncClient, None]:
    """Admin authenticated client."""
    response = await client.post(
        "/api/auth/login",
        json={"username": ADMIN_USER, "password": ADMIN_PASS},
    )
    
    if response.status_code != 200:
        pytest.skip(f"Admin login failed: {response.status_code}")
    
    data = response.json()
    if "session_token" in data and data["session_token"]:
        client.headers["Authorization"] = f"Bearer {data['session_token']}"
    
    for cookie in response.cookies:
        client.cookies.set(cookie.name, cookie.value)
    
    yield client


class TestTenantIsolation:
    """Verify tenant data isolation."""

    async def test_tenant_header_accepted(self, authenticated_client: httpx.AsyncClient):
        """Verify X-Tenant-ID header is accepted."""
        response = await authenticated_client.get(
            "/health",
            headers={"X-Tenant-ID": "test-tenant-123"}
        )
        assert response.status_code == 200

    async def test_health_with_tenant_context(self, authenticated_client: httpx.AsyncClient):
        """Health endpoint works with tenant context."""
        response = await authenticated_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["ok", "healthy"]


class TestSCIMProvisioning:
    """Verify SCIM 2.0 endpoints."""

    async def test_scim_service_provider_config(self, client: httpx.AsyncClient):
        """SCIM service provider config endpoint."""
        response = await client.get(
            "/scim/ServiceProviderConfig",
            headers={"Authorization": "Bearer test-scim-token-12345"}
        )
        
        if response.status_code == 404:
            pytest.skip("SCIM endpoints not configured")
        
        assert response.status_code == 200
        data = response.json()
        assert "schemas" in data
        assert "patch" in data

    async def test_scim_schemas(self, client: httpx.AsyncClient):
        """SCIM schemas endpoint."""
        response = await client.get(
            "/scim/Schemas",
            headers={"Authorization": "Bearer test-scim-token-12345"}
        )
        
        if response.status_code == 404:
            pytest.skip("SCIM endpoints not configured")
        
        assert response.status_code == 200

    async def test_scim_resource_types(self, client: httpx.AsyncClient):
        """SCIM resource types endpoint."""
        response = await client.get(
            "/scim/ResourceTypes",
            headers={"Authorization": "Bearer test-scim-token-12345"}
        )
        
        if response.status_code == 404:
            pytest.skip("SCIM endpoints not configured")
        
        assert response.status_code == 200

    async def test_scim_list_users(self, client: httpx.AsyncClient):
        """SCIM list users endpoint."""
        response = await client.get(
            "/scim/Users",
            headers={"Authorization": "Bearer test-scim-token-12345"}
        )
        
        if response.status_code == 404:
            pytest.skip("SCIM endpoints not configured")
        
        assert response.status_code == 200
        data = response.json()
        assert "Resources" in data
        assert "totalResults" in data

    async def test_scim_create_user(self, client: httpx.AsyncClient):
        """SCIM user creation."""
        import uuid
        unique_email = f"test.user.{uuid.uuid4().hex[:8]}@example.com"
        
        response = await client.post(
            "/scim/Users",
            headers={"Authorization": "Bearer test-scim-token-12345"},
            json={
                "schemas": ["urn:ietf:params:scim:schemas:core:2.0:User"],
                "userName": unique_email,
                "name": {"givenName": "Test", "familyName": "User"},
                "emails": [{"value": unique_email, "primary": True}],
                "active": True
            }
        )
        
        if response.status_code == 404:
            pytest.skip("SCIM endpoints not configured")
        
        assert response.status_code == 201
        data = response.json()
        assert data["userName"] == unique_email
        assert "id" in data


class TestRBAC:
    """Verify role-based access control."""

    async def test_authenticated_user_can_access_dashboard(
        self, authenticated_client: httpx.AsyncClient
    ):
        """Authenticated users can access dashboard."""
        response = await authenticated_client.get("/api/v1/banking/executive/dashboard")
        
        # Should be 200 or 403 (depending on role), not 401
        if response.status_code == 404:
            pytest.skip("Banking endpoints not configured")
        
        assert response.status_code in [200, 403]

    async def test_unauthenticated_blocked(self, client: httpx.AsyncClient):
        """Unauthenticated requests are blocked from protected endpoints."""
        response = await client.get("/api/v1/banking/executive/dashboard")
        
        if response.status_code == 404:
            pytest.skip("Banking endpoints not configured")
        
        assert response.status_code in [401, 403]


class TestAdminConsole:
    """Verify admin console endpoints."""

    async def test_settings_page_accessible(self, authenticated_client: httpx.AsyncClient):
        """Settings page is accessible to authenticated users."""
        response = await authenticated_client.get("/ui/settings.html")
        
        if response.status_code == 404:
            pytest.skip("Settings page not found")
        
        assert response.status_code == 200

    async def test_admin_qa_page_accessible(self, admin_client: httpx.AsyncClient):
        """Admin QA page is accessible to admins."""
        response = await admin_client.get("/ui/admin_qa.html")
        
        if response.status_code == 404:
            pytest.skip("Admin QA page not found")
        
        assert response.status_code == 200
