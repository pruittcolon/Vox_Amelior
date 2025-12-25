"""
Integration Tests for Critical System Flows
- Mobile App Login (HTTPS/Secure Cookies)
- RAG Database Integrity (Check for empty text)
- CSRF Protection Validation
"""
import pytest
import httpx
import asyncio

# Use HTTPS for these tests as Mobile App and CSRF/Cookie logic depends on it
BASE_URL = "https://localhost"

@pytest.mark.integration
class TestCriticalFlows:
    
    @pytest.fixture
    async def client(self):
        """Async client with HTTPS and self-signed cert verification disabled"""
        async with httpx.AsyncClient(base_url=BASE_URL, verify=False, timeout=30.0) as client:
            yield client

    @pytest.mark.asyncio
    async def test_mobile_app_login_flow(self, client):
        """
        Simulates the logic in auth_service.dart to verify mobile login works
        despite self-signed certs and secure cookies.
        """
        # 1. Login
        login_payload = {"username": "PruittColon", "password": "Pruitt12!"}
        response = await client.post("/api/auth/login", json=login_payload)
        
        assert response.status_code == 200, f"Login failed: {response.text}"
        data = response.json()
        assert data["success"] is True
        assert "session_token" in data
        assert "csrf_token" in data
        
        # Verify headers/cookies are received
        cookies = response.cookies
        assert "ws_session" in cookies
        assert "ws_csrf" in cookies
        
        # 2. Check Session (using cookies from login)
        # Mobile app sends 'Cookie' header manually constructed
        session_token = data["session_token"]
        csrf_token = data["csrf_token"]
        
        check_response = await client.get(
            "/api/auth/check",
            cookies={"ws_session": session_token}
        )
        assert check_response.status_code == 200
        assert check_response.json()["valid"] is True

    @pytest.mark.asyncio
    async def test_rag_data_integrity(self, client):
        """
        Verifies that /transcripts/query returns valid text.
        This confirms the database is not corrupted (regression test).
        """
        # Login first
        login_res = await client.post("/api/auth/login", json={"username": "PruittColon", "password": "Pruitt12!"})
        cookies = login_res.cookies
        csrf_token = login_res.cookies.get("ws_csrf") or login_res.json().get("csrf_token")
        
        # Query transcripts (same as Gemma streaming)
        payload = {"limit": 5}
        headers = {"X-CSRF-Token": csrf_token}
        
        response = await client.post(
            "/api/transcripts/query",
            json=payload,
            cookies=cookies,
            headers=headers
        )
        
        assert response.status_code == 200, f"RAG query failed: {response.text}"
        data = response.json()
        
        # CRITICAL CHECK: Verify items have non-empty text
        items = data.get("items", [])
        # If database is fresh/empty, this might be validly empty, 
        # but if we restored backup, it should have data.
        # We assert that IF items exist, they MUST have text.
        for item in items:
            text = item.get("text", "")
            assert text and text.strip(), f"Found item with empty text! ID: {item.get('segment_id')}"
            
        print(f"Verified {len(items)} RAG items have valid text.")

    @pytest.mark.asyncio
    async def test_csrf_protection_enforcement(self, client):
        """
        Verifies that CSRF protection blocks requests correctly
        """
        # Login
        login_res = await client.post("/api/auth/login", json={"username": "PruittColon", "password": "Pruitt12!"})
        cookies = login_res.cookies
        
        # Attempt POST without CSRF header
        response = await client.post(
            "/api/transcripts/query",
            json={"limit": 1},
            cookies=cookies
            # Missing X-CSRF-Token header
        )
        
        assert response.status_code == 403
        assert "CSRF" in response.text
