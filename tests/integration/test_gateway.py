"""
Integration Tests for Gateway to Service Communication
Requires running Docker Compose stack.
"""
import pytest
import uuid


@pytest.mark.integration
class TestGatewayProxy:
    """Test gateway correctly proxies to backend services."""
    
    @pytest.mark.asyncio
    async def test_health_endpoint_accessible(self, http_client):
        """Health endpoint should be accessible without auth."""
        response = await http_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data.get("status") in ["ok", "healthy"]
    
    @pytest.mark.asyncio
    async def test_gemma_stats_accessible(self, http_client, auth_headers):
        """Gemma stats should be accessible with auth."""
        response = await http_client.get(
            "/api/gemma/stats",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "model_on_gpu" in data or "status" in data
    
    @pytest.mark.asyncio
    async def test_search_endpoint_accessible(self, http_client, auth_headers):
        """Search endpoint should be accessible with auth."""
        response = await http_client.post(
            "/api/search/semantic",
            json={"query": "test", "top_k": 5},
            headers=auth_headers
        )
        
        assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_analytics_signals_accessible(self, http_client, auth_headers):
        """Analytics signals should be accessible with auth."""
        response = await http_client.get(
            "/api/analytics/signals",
            headers=auth_headers
        )
        
        assert response.status_code == 200


@pytest.mark.integration
class TestAuthenticationFlow:
    """Test authentication flow through gateway."""
    
    @pytest.mark.asyncio
    async def test_login_returns_session_token(self, http_client):
        """Login should return session token."""
        response = await http_client.post("/api/auth/login", json={
            "username": "admin",
            "password": "admin123"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data.get("success") is True
        assert "session_token" in data or "token" in data
    
    @pytest.mark.asyncio
    async def test_protected_endpoint_requires_auth(self, http_client):
        """Protected endpoints should reject unauthenticated requests."""
        response = await http_client.get("/api/transcripts/recent")
        
        # Should get 401 or 403 without auth
        assert response.status_code in [401, 403]
    
    @pytest.mark.asyncio
    async def test_invalid_login_rejected(self, http_client):
        """Invalid credentials should be rejected."""
        response = await http_client.post("/api/auth/login", json={
            "username": "nonexistent",
            "password": "wrongpassword"
        })
        
        assert response.status_code in [401, 403]

    @pytest.mark.asyncio
    async def test_register_creates_account_and_session(self, http_client):
        """Self-registration should create a user and return a valid session."""
        username = f"testuser_{uuid.uuid4().hex[:6]}"
        password = "TestPass123"

        res = await http_client.post("/api/auth/register", json={
            "username": username,
            "password": password,
            "email": f"{username}@example.com",
        })
        assert res.status_code == 200, res.text
        data = res.json()
        assert data.get("success") is True
        assert data.get("session_token")

        token = data["session_token"]
        check = await http_client.get("/api/auth/check", cookies={"ws_session": token})
        assert check.status_code == 200
        assert check.json().get("valid") is True

        # Duplicate registration should be rejected
        dup = await http_client.post("/api/auth/register", json={
            "username": username,
            "password": password,
        })
        assert dup.status_code == 409
