"""
Integration tests for Week 5: OIDC & SAML SSO Implementation.

Tests cover:
- OIDC provider initialization and PKCE flow
- SAML provider initialization and AuthnRequest generation
- SCIM store CRUD operations with tenant isolation
- SCIM lifecycle: create -> update -> deactivate -> reactivate
"""

import os
import sys
import tempfile
from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest

# Add paths for direct imports (avoiding services/__init__.py chain)
_api_gateway_auth = os.path.join(
    os.path.dirname(__file__), "..", "..", "services", "api-gateway", "src", "auth"
)
sys.path.insert(0, os.path.abspath(_api_gateway_auth))

# Set test environment before imports
os.environ.setdefault("OIDC_CLIENT_ID", "test-client-id")
os.environ.setdefault("OIDC_CLIENT_SECRET", "test-secret")
os.environ.setdefault("OIDC_REDIRECT_URI", "https://localhost/auth/callback")


class TestOIDCProvider:
    """Tests for OIDC provider functionality."""
    
    def test_pkce_generation(self) -> None:
        """PKCE verifier and challenge generation."""
        from oidc_provider import (
            _generate_code_verifier,
            _generate_code_challenge,
        )
        
        verifier = _generate_code_verifier()
        challenge = _generate_code_challenge(verifier)
        
        # Verifier should be 43-128 characters per RFC 7636
        assert 43 <= len(verifier) <= 128
        assert verifier.isascii()
        
        # Challenge is base64url encoded SHA256
        assert len(challenge) == 43  # 256 bits / 6 bits per char ≈ 43
        assert "=" not in challenge  # No padding
    
    def test_provider_from_environment(self) -> None:
        """Provider loads configuration from environment."""
        from oidc_provider import OIDCProvider
        
        provider = OIDCProvider.from_environment()
        
        assert provider.config.client_id == "test-client-id"
        assert provider.config.redirect_uri == "https://localhost/auth/callback"
        assert "openid" in provider.config.scopes
    
    def test_authorization_url_contains_pkce(self) -> None:
        """Authorization URL includes PKCE challenge."""
        from oidc_provider import OIDCProvider
        
        provider = OIDCProvider.from_environment()
        state = provider.generate_state()
        verifier, challenge = provider.generate_pkce()
        
        auth_url = provider.get_authorization_url(state=state, code_challenge=challenge)
        
        assert "code_challenge=" in auth_url
        assert "code_challenge_method=S256" in auth_url
        assert f"state={state}" in auth_url
        assert "response_type=code" in auth_url
    
    def test_state_is_cryptographically_random(self) -> None:
        """State parameter is sufficiently random."""
        from oidc_provider import OIDCProvider
        
        provider = OIDCProvider.from_environment()
        
        states = [provider.generate_state() for _ in range(100)]
        
        # All states should be unique
        assert len(set(states)) == 100
        
        # Each state should be at least 32 bytes when decoded
        for state in states:
            assert len(state) >= 32


class TestSAMLProvider:
    """Tests for SAML provider functionality."""
    
    @pytest.fixture
    def saml_env(self) -> dict[str, str]:
        """Set up SAML environment variables."""
        env = {
            "SAML_ENTITY_ID": "https://localhost/saml/metadata",
            "SAML_ACS_URL": "https://localhost/saml/acs",
            "SAML_IDP_ENTITY_ID": "https://idp.example.com",
            "SAML_IDP_SSO_URL": "https://idp.example.com/sso",
        }
        with patch.dict(os.environ, env):
            yield env
    
    def test_authn_request_generation(self, saml_env: dict[str, str]) -> None:
        """AuthnRequest is properly formatted XML."""
        from saml_provider import SAMLProvider
        
        provider = SAMLProvider.from_environment()
        request = provider.create_authn_request(relay_state="https://app.example.com")
        
        # Request ID is properly formatted
        assert request.id.startswith("_")
        assert len(request.id) >= 32
        
        # Issue instant is ISO format
        datetime.fromisoformat(request.issue_instant.replace("Z", "+00:00"))
        
        # XML contains required elements
        assert "AuthnRequest" in request.request_xml
        assert "Issuer" in request.request_xml
        assert saml_env["SAML_ENTITY_ID"] in request.request_xml
    
    def test_redirect_url_contains_request(self, saml_env: dict[str, str]) -> None:
        """SSO redirect URL contains encoded request."""
        from saml_provider import SAMLProvider
        
        provider = SAMLProvider.from_environment()
        request = provider.create_authn_request(relay_state="https://app.example.com")
        redirect_url = provider.get_sso_redirect_url(request)
        
        assert redirect_url.startswith("https://idp.example.com/sso?")
        assert "SAMLRequest=" in redirect_url
        assert "RelayState=" in redirect_url
    
    def test_pending_request_tracking(self, saml_env: dict[str, str]) -> None:
        """Pending AuthnRequests are tracked and cleaned up."""
        from saml_provider import SAMLProvider
        
        provider = SAMLProvider.from_environment()
        
        # Create multiple requests
        request1 = provider.create_authn_request()
        request2 = provider.create_authn_request()
        
        # Both should be tracked
        assert request1.id in provider._pending_requests
        assert request2.id in provider._pending_requests
        
        # IDs should be unique
        assert request1.id != request2.id


class TestSCIMStore:
    """Tests for SCIM database store."""
    
    @pytest.fixture
    def scim_store(self) -> "SCIMStore":
        """Create temporary SCIM store for testing."""
        from shared.auth.scim_store import SCIMStore
        
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        
        store = SCIMStore(db_path)
        yield store
        
        # Cleanup
        os.unlink(db_path)
    
    def test_user_creation(self, scim_store) -> None:
        """Users can be created with all fields."""
        user = scim_store.create_user(
            tenant_id="tenant-1",
            user_name="john.doe@example.com",
            given_name="John",
            family_name="Doe",
            email="john.doe@example.com",
            external_id="ext-123",
        )
        
        assert user["id"]
        assert user["userName"] == "john.doe@example.com"
        assert user["name"]["givenName"] == "John"
        assert user["name"]["familyName"] == "Doe"
        assert user["active"] is True
        assert "meta" in user
    
    def test_user_lookup_by_username(self, scim_store) -> None:
        """Users can be looked up by username."""
        created = scim_store.create_user(
            tenant_id="tenant-1",
            user_name="lookup@example.com",
        )
        
        found = scim_store.get_user_by_username("tenant-1", "lookup@example.com")
        
        assert found is not None
        assert found["id"] == created["id"]
    
    def test_tenant_isolation(self, scim_store) -> None:
        """Users are isolated between tenants."""
        scim_store.create_user(
            tenant_id="tenant-1",
            user_name="shared@example.com",
        )
        
        # Same username in different tenant
        scim_store.create_user(
            tenant_id="tenant-2",
            user_name="shared@example.com",
        )
        
        users_t1, count_t1 = scim_store.list_users("tenant-1")
        users_t2, count_t2 = scim_store.list_users("tenant-2")
        
        assert count_t1 == 1
        assert count_t2 == 1
        assert users_t1[0]["id"] != users_t2[0]["id"]
    
    def test_user_update(self, scim_store) -> None:
        """User fields can be updated."""
        user = scim_store.create_user(
            tenant_id="tenant-1",
            user_name="update@example.com",
            given_name="Original",
        )
        
        updated = scim_store.update_user(
            "tenant-1",
            user["id"],
            givenName="Updated",
            displayName="Updated User",
        )
        
        assert updated["name"]["givenName"] == "Updated"
        assert updated["displayName"] == "Updated User"
    
    def test_soft_delete_preserves_user(self, scim_store) -> None:
        """Soft delete marks user inactive but preserves record."""
        user = scim_store.create_user(
            tenant_id="tenant-1",
            user_name="softdelete@example.com",
        )
        
        deleted = scim_store.delete_user("tenant-1", user["id"], soft=True)
        assert deleted is True
        
        # User still exists but inactive
        found = scim_store.get_user("tenant-1", user["id"])
        assert found is not None
        assert found["active"] is False
    
    def test_user_reactivation(self, scim_store) -> None:
        """Soft-deleted users can be reactivated."""
        user = scim_store.create_user(
            tenant_id="tenant-1",
            user_name="reactivate@example.com",
        )
        
        scim_store.delete_user("tenant-1", user["id"], soft=True)
        
        reactivated = scim_store.update_user(
            "tenant-1",
            user["id"],
            active=True,
        )
        
        assert reactivated["active"] is True
    
    def test_list_with_filter(self, scim_store) -> None:
        """Users can be filtered by field."""
        for i in range(5):
            scim_store.create_user(
                tenant_id="tenant-1",
                user_name=f"user{i}@example.com",
                email=f"user{i}@example.com",
            )
        
        users, total = scim_store.list_users(
            "tenant-1",
            filter_field="userName",
            filter_value="user2@example.com",
        )
        
        assert total == 1
        assert users[0]["userName"] == "user2@example.com"
    
    def test_pagination(self, scim_store) -> None:
        """User list supports pagination."""
        for i in range(10):
            scim_store.create_user(
                tenant_id="tenant-1",
                user_name=f"page{i}@example.com",
            )
        
        page1, total = scim_store.list_users("tenant-1", start_index=1, count=3)
        page2, _ = scim_store.list_users("tenant-1", start_index=4, count=3)
        
        assert total == 10
        assert len(page1) == 3
        assert len(page2) == 3
        
        # Pages shouldn't overlap
        page1_ids = {u["id"] for u in page1}
        page2_ids = {u["id"] for u in page2}
        assert page1_ids.isdisjoint(page2_ids)


class TestSCIMLifecycle:
    """End-to-end SCIM provisioning lifecycle tests."""
    
    @pytest.fixture
    def scim_store(self):
        """Create temporary SCIM store."""
        from shared.auth.scim_store import SCIMStore
        
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        
        store = SCIMStore(db_path)
        yield store
        os.unlink(db_path)
    
    def test_full_user_lifecycle(self, scim_store) -> None:
        """Test complete user provisioning lifecycle."""
        tenant = "enterprise-tenant"
        
        # 1. Create user (new hire)
        user = scim_store.create_user(
            tenant_id=tenant,
            user_name="new.employee@company.com",
            given_name="New",
            family_name="Employee",
            email="new.employee@company.com",
            external_id="emp-12345",
        )
        assert user["active"] is True
        
        # 2. Update user (name change)
        user = scim_store.update_user(
            tenant,
            user["id"],
            familyName="UpdatedLastName",
        )
        assert user["name"]["familyName"] == "UpdatedLastName"
        
        # 3. Deactivate user (off-boarding)
        scim_store.delete_user(tenant, user["id"], soft=True)
        user = scim_store.get_user(tenant, user["id"])
        assert user["active"] is False
        
        # 4. Reactivate user (re-hire)
        user = scim_store.update_user(tenant, user["id"], active=True)
        assert user["active"] is True
        
        # 5. Hard delete (GDPR data deletion request)
        scim_store.delete_user(tenant, user["id"], soft=False)
        user = scim_store.get_user(tenant, user["id"])
        assert user is None
    
    def test_group_membership_lifecycle(self, scim_store) -> None:
        """Test group membership management."""
        tenant = "enterprise-tenant"
        
        # Create users
        user1 = scim_store.create_user(tenant_id=tenant, user_name="user1@example.com")
        user2 = scim_store.create_user(tenant_id=tenant, user_name="user2@example.com")
        
        # Create group
        group = scim_store.create_group(
            tenant_id=tenant,
            display_name="Developers",
            external_id="group-dev",
        )
        
        # Add members
        scim_store.add_user_to_group(group["id"], user1["id"])
        scim_store.add_user_to_group(group["id"], user2["id"])
        
        # Verify membership
        group = scim_store.get_group(tenant, group["id"])
        assert len(group["members"]) == 2
        
        # Remove member
        scim_store.remove_user_from_group(group["id"], user1["id"])
        
        group = scim_store.get_group(tenant, group["id"])
        assert len(group["members"]) == 1
        assert group["members"][0]["value"] == user2["id"]


class TestOIDCEnabled:
    """Tests for OIDC feature flag."""
    
    def test_oidc_enabled_when_configured(self) -> None:
        """OIDC is enabled when client ID is set."""
        from oidc_provider import is_oidc_enabled
        
        with patch.dict(os.environ, {"OIDC_CLIENT_ID": "test-id"}):
            assert is_oidc_enabled() is True
    
    def test_oidc_disabled_when_not_configured(self) -> None:
        """OIDC is disabled when client ID is not set."""
        from oidc_provider import is_oidc_enabled
        
        env = os.environ.copy()
        env.pop("OIDC_CLIENT_ID", None)
        
        with patch.dict(os.environ, env, clear=True):
            assert is_oidc_enabled() is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
