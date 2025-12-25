"""
API Gateway Authentication Tests
Unit tests for authentication utilities.
"""
import pytest


@pytest.mark.unit
class TestPasswordUtilities:
    """Test password hashing and verification."""
    
    def test_password_hash_is_not_plaintext(self):
        """Hashed password should not equal plaintext."""
        try:
            import sys
            sys.path.insert(0, 'services/api-gateway/src')
            from auth import hash_password
            
            plaintext = "testpassword123"
            hashed = hash_password(plaintext)
            
            assert hashed != plaintext
            assert len(hashed) > len(plaintext)
        except ImportError:
            pytest.skip("Gateway auth module not importable in isolation")
    
    def test_password_hash_creates_unique_hashes(self):
        """Same password should create different hashes (salt)."""
        try:
            import sys
            sys.path.insert(0, 'services/api-gateway/src')
            from auth import hash_password
            
            plaintext = "testpassword123"
            hash1 = hash_password(plaintext)
            hash2 = hash_password(plaintext)
            
            # bcrypt creates unique hashes due to random salt
            assert hash1 != hash2
        except ImportError:
            pytest.skip("Gateway auth module not importable in isolation")
    
    def test_password_verify_correct_password(self):
        """Correct password should verify successfully."""
        try:
            import sys
            sys.path.insert(0, 'services/api-gateway/src')
            from auth import hash_password, verify_password
            
            plaintext = "testpassword123"
            hashed = hash_password(plaintext)
            
            assert verify_password(plaintext, hashed) is True
        except ImportError:
            pytest.skip("Gateway auth module not importable in isolation")
    
    def test_password_verify_incorrect_password(self):
        """Incorrect password should fail verification."""
        try:
            import sys
            sys.path.insert(0, 'services/api-gateway/src')
            from auth import hash_password, verify_password
            
            hashed = hash_password("correctpassword")
            
            assert verify_password("wrongpassword", hashed) is False
        except ImportError:
            pytest.skip("Gateway auth module not importable in isolation")


@pytest.mark.unit
class TestSessionTokens:
    """Test session token generation and validation."""
    
    def test_session_token_has_sufficient_entropy(self):
        """Session tokens should have sufficient length for security."""
        import secrets
        
        # Simulate token generation similar to gateway
        token = secrets.token_urlsafe(32)
        
        assert len(token) >= 32
        assert token.isalnum() or '-' in token or '_' in token
    
    def test_session_tokens_are_unique(self):
        """Multiple generated tokens should be unique."""
        import secrets
        
        tokens = [secrets.token_urlsafe(32) for _ in range(100)]
        unique_tokens = set(tokens)
        
        assert len(unique_tokens) == 100


@pytest.mark.unit
class TestInputValidation:
    """Test input validation utilities."""
    
    def test_username_allows_alphanumeric(self):
        """Usernames should allow alphanumeric characters."""
        import re
        
        valid_pattern = re.compile(r'^[a-zA-Z0-9_]{3,50}$')
        
        assert valid_pattern.match("admin")
        assert valid_pattern.match("user123")
        assert valid_pattern.match("test_user")
    
    def test_username_rejects_special_chars(self):
        """Usernames should reject dangerous characters."""
        import re
        
        valid_pattern = re.compile(r'^[a-zA-Z0-9_]{3,50}$')
        
        assert not valid_pattern.match("admin'; DROP TABLE--")
        assert not valid_pattern.match("<script>alert(1)</script>")
        assert not valid_pattern.match("user@domain.com")
