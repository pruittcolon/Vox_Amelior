"""
Test security headers configuration.

Phase 1: Security Hardening - TLS and Security Headers Validation
"""
import pytest
from pathlib import Path


class TestSecurityHeaders:
    """Test security headers are properly configured by checking source code."""
    
    @pytest.fixture
    def main_py_content(self):
        """Load main.py content for analysis."""
        main_path = Path("/home/pruittcolon/Desktop/Nemo_Server/services/api-gateway/src/main.py")
        return main_path.read_text()
    
    def test_security_headers_middleware_defined(self, main_py_content):
        """SecurityHeadersMiddleware should be defined."""
        assert "class SecurityHeadersMiddleware" in main_py_content
    
    def test_csp_config_has_frame_ancestors(self, main_py_content):
        """CSP should include frame-ancestors 'none' for clickjacking protection."""
        assert "frame-ancestors 'none'" in main_py_content
    
    def test_hsts_header_configured(self, main_py_content):
        """HSTS should be configured with appropriate max-age."""
        assert "Strict-Transport-Security" in main_py_content
        assert "max-age=" in main_py_content
    
    def test_x_content_type_options(self, main_py_content):
        """X-Content-Type-Options nosniff should be set."""
        assert "X-Content-Type-Options" in main_py_content
        assert "nosniff" in main_py_content
    
    def test_x_frame_options(self, main_py_content):
        """X-Frame-Options DENY should be set."""
        assert "X-Frame-Options" in main_py_content
        assert "DENY" in main_py_content


class TestContainerSecurity:
    """Test container security configuration."""
    
    def test_docker_compose_has_read_only(self):
        """Docker compose should have read_only for api-gateway."""
        from pathlib import Path
        
        compose_path = Path("/home/pruittcolon/Desktop/Nemo_Server/docker/docker-compose.yml")
        content = compose_path.read_text()
        
        assert "read_only: true" in content
    
    def test_docker_compose_has_cap_drop(self):
        """Docker compose should have cap_drop: ALL for api-gateway."""
        from pathlib import Path
        
        compose_path = Path("/home/pruittcolon/Desktop/Nemo_Server/docker/docker-compose.yml")
        content = compose_path.read_text()
        
        assert "cap_drop:" in content
        assert "- ALL" in content
    
    def test_docker_compose_has_no_new_privileges(self):
        """Docker compose should have no-new-privileges security option."""
        from pathlib import Path
        
        compose_path = Path("/home/pruittcolon/Desktop/Nemo_Server/docker/docker-compose.yml")
        content = compose_path.read_text()
        
        assert "no-new-privileges" in content
    
    def test_secure_mode_enabled(self):
        """SECURE_MODE should default to true in docker-compose."""
        from pathlib import Path
        
        compose_path = Path("/home/pruittcolon/Desktop/Nemo_Server/docker/docker-compose.yml")
        content = compose_path.read_text()
        
        assert (
            'SECURE_MODE: "${SECURE_MODE:-true}"' in content
            or 'SECURE_MODE: "true"' in content
        )
