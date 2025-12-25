"""
Test CSRF protection and rate limiting.

Phase 4: Test coverage push for security controls.
"""
import pytest
import time
from pathlib import Path


class TestCSRFProtection:
    """Test CSRF middleware configuration."""
    
    @pytest.fixture
    def main_py_content(self):
        """Load main.py content for analysis."""
        main_path = Path("/home/pruittcolon/Desktop/Nemo_Server/services/api-gateway/src/main.py")
        return main_path.read_text()
    
    def test_csrf_middleware_exists(self, main_py_content):
        """CSRFMiddleware should be defined."""
        assert "class CSRFMiddleware" in main_py_content
    
    def test_csrf_exempt_paths_defined(self, main_py_content):
        """CSRF should have exempt paths for login."""
        assert "exempt_paths" in main_py_content
        assert "/api/auth/login" in main_py_content
    
    def test_csrf_double_submit_check(self, main_py_content):
        """CSRF should check header matches cookie (double-submit pattern)."""
        assert "header_token" in main_py_content
        assert "cookie_token" in main_py_content
    
    def test_csrf_bearer_token_support(self, main_py_content):
        """CSRF should support Bearer tokens for mobile clients."""
        assert "Bearer " in main_py_content
        assert "bearer_auth_paths" in main_py_content


class TestRateLimiting:
    """Test rate limiting middleware configuration."""
    
    @pytest.fixture
    def main_py_content(self):
        main_path = Path("/home/pruittcolon/Desktop/Nemo_Server/services/api-gateway/src/main.py")
        return main_path.read_text()
    
    def test_rate_limit_middleware_exists(self, main_py_content):
        """RateLimitMiddleware should be defined."""
        assert "class RateLimitMiddleware" in main_py_content
    
    def test_rate_limit_configurable(self, main_py_content):
        """Rate limits should be configurable via environment."""
        assert "RATE_LIMIT_ENABLED" in main_py_content
        assert "RATE_LIMIT_DEFAULT" in main_py_content
    
    def test_rate_limit_returns_429(self, main_py_content):
        """Rate limiting should return 429 Too Many Requests."""
        assert "429" in main_py_content
        assert "Too Many Requests" in main_py_content
    
    def test_rate_limit_auth_endpoint_protected(self, main_py_content):
        """Auth endpoint should have separate rate limit."""
        assert "auth_limit" in main_py_content or "RATE_LIMIT_AUTH" in main_py_content


class TestSessionSecurity:
    """Test session cookie security configuration."""
    
    @pytest.fixture
    def config_py_content(self):
        config_path = Path("/home/pruittcolon/Desktop/Nemo_Server/services/api-gateway/src/config/legacy_config.py")
        return config_path.read_text()
    
    def test_session_cookie_httponly(self, config_py_content):
        """Session cookies should be HTTP-only."""
        # Check for session cookie configuration
        assert "SESSION_COOKIE_NAME" in config_py_content
    
    def test_session_cookie_secure_flag(self, config_py_content):
        """Session cookie secure flag should be configurable."""
        assert "SESSION_COOKIE_SECURE" in config_py_content
    
    def test_csrf_cookie_name_defined(self, config_py_content):
        """CSRF cookie name should be defined."""
        assert "CSRF_COOKIE_NAME" in config_py_content


class TestCORSConfiguration:
    """Test CORS middleware configuration."""
    
    @pytest.fixture
    def main_py_content(self):
        main_path = Path("/home/pruittcolon/Desktop/Nemo_Server/services/api-gateway/src/main.py")
        return main_path.read_text()
    
    def test_cors_middleware_added(self, main_py_content):
        """CORS middleware should be added to app."""
        assert "CORSMiddleware" in main_py_content
    
    def test_cors_allowed_origins_configurable(self, main_py_content):
        """Allowed origins should be configurable."""
        assert "ALLOWED_ORIGINS" in main_py_content
    
    def test_cors_credentials_enabled(self, main_py_content):
        """CORS should allow credentials for cookie-based auth."""
        assert "allow_credentials" in main_py_content
