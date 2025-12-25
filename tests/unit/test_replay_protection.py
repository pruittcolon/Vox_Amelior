"""
Test replay protection for S2S JWT tokens.

Phase 1: Security Hardening - Replay Attack Prevention
"""
import pytest
import time
from unittest.mock import patch, MagicMock


class TestReplayProtection:
    """Test replay attack protection for S2S JWT."""
    
    def test_first_request_allowed(self):
        """First use of a request ID should succeed."""
        from shared.security.service_auth import _InMemoryReplayStore
        
        store = _InMemoryReplayStore()
        result = store.add_if_absent("test-request-123", ttl_seconds=60)
        
        assert result is True
    
    def test_replay_detected(self):
        """Same request ID should be rejected on second use."""
        from shared.security.service_auth import _InMemoryReplayStore
        
        store = _InMemoryReplayStore()
        
        # First request succeeds
        result1 = store.add_if_absent("test-request-456", ttl_seconds=60)
        assert result1 is True
        
        # Same request ID is rejected (replay)
        result2 = store.add_if_absent("test-request-456", ttl_seconds=60)
        assert result2 is False
    
    def test_unique_requests_allowed(self):
        """Different request IDs should all be allowed."""
        from shared.security.service_auth import _InMemoryReplayStore
        
        store = _InMemoryReplayStore()
        
        result_a = store.add_if_absent("request-a", ttl_seconds=60)
        result_b = store.add_if_absent("request-b", ttl_seconds=60)
        result_c = store.add_if_absent("request-c", ttl_seconds=60)
        
        assert result_a is True
        assert result_b is True
        assert result_c is True
    
    def test_expired_entries_cleaned(self):
        """Expired entries should be cleaned up and allow reuse."""
        from shared.security.service_auth import _InMemoryReplayStore
        
        store = _InMemoryReplayStore()
        
        # Add with very short TTL
        store.add_if_absent("expiring-request", ttl_seconds=1)
        
        # Wait for expiry
        time.sleep(1.1)
        
        # Should now be allowed again
        result = store.add_if_absent("expiring-request", ttl_seconds=60)
        assert result is True


class TestReplayProtectorIntegration:
    """Integration tests for ReplayProtector with Redis fallback."""
    
    def test_replay_protector_check_and_store(self):
        """ReplayProtector should detect replays."""
        from shared.security.service_auth import ReplayProtector
        
        # Use in-memory fallback (no Redis in unit tests)
        with patch.dict('os.environ', {'REDIS_URL': 'redis://nonexistent:6379'}):
            protector = ReplayProtector(url="redis://nonexistent:6379")
        
        # First request succeeds
        ok1, reason1 = protector.check_and_store("req-001", ttl_seconds=60)
        assert ok1 is True
        assert reason1 == "ok"
        
        # Replay is detected
        ok2, reason2 = protector.check_and_store("req-001", ttl_seconds=60)
        assert ok2 is False
        assert "replay" in reason2.lower()
    
    def test_empty_request_id_rejected(self):
        """Empty request ID should be rejected."""
        from shared.security.service_auth import ReplayProtector
        
        protector = ReplayProtector(url="redis://nonexistent:6379")
        
        ok, reason = protector.check_and_store("", ttl_seconds=60)
        assert ok is False
        assert "missing" in reason.lower()


class TestServiceAuthWithReplay:
    """End-to-end tests for ServiceAuth with replay protection."""
    
    @pytest.fixture
    def service_auth(self):
        """Create a ServiceAuth instance for testing."""
        from shared.security.service_auth import ServiceAuth
        import secrets
        import base64
        
        test_secret = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode()
        return ServiceAuth("test-service", test_secret)
    
    def test_token_replay_blocked(self, service_auth):
        """Replaying the same token should be blocked."""
        from shared.security.service_auth import get_replay_protector
        
        # Clear any existing protector singleton
        import shared.security.service_auth as auth_module
        auth_module._replay_protector = None
        
        # Create token
        token = service_auth.create_token(expires_in=300)
        
        # First verification succeeds
        payload1 = service_auth.verify_token(token)
        assert payload1 is not None
        assert payload1.get("service_id") == "test-service"
        
        # Second verification fails (replay)
        with pytest.raises(ValueError, match="[Rr]eplay"):
            service_auth.verify_token(token)
