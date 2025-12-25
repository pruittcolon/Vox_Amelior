"""
Integration tests for Week 11: Frontend Security & OWASP Compliance.

Tests cover:
- CSP validation and auditing
- XSS detection and prevention
- HTML sanitization
- Context-aware encoding
"""

import os
import sys
from pathlib import Path

import pytest

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shared" / "security"))


class TestCSPValidator:
    """Tests for CSP validation."""
    
    def test_parse_simple_csp(self) -> None:
        """Simple CSP headers can be parsed."""
        from csp_validator import CSPValidator
        
        validator = CSPValidator()
        policy = validator.parse("default-src 'self'")
        
        assert policy.has_directive("default-src")
        assert "'self'" in policy.get_sources("default-src")
    
    def test_parse_complex_csp(self) -> None:
        """Complex CSP headers with multiple directives."""
        from csp_validator import CSPValidator
        
        csp = "default-src 'self'; script-src 'self' https://cdn.example.com; img-src *"
        
        validator = CSPValidator()
        policy = validator.parse(csp)
        
        assert len(policy.directives) == 3
        assert "https://cdn.example.com" in policy.get_sources("script-src")
        assert "*" in policy.get_sources("img-src")
    
    def test_to_header_roundtrip(self) -> None:
        """Policy can be converted back to header."""
        from csp_validator import CSPValidator
        
        original = "default-src 'self'; script-src 'self'"
        
        validator = CSPValidator()
        policy = validator.parse(original)
        generated = policy.to_header()
        
        # Both should contain same directives
        assert "default-src" in generated
        assert "script-src" in generated


class TestCSPAuditor:
    """Tests for CSP security auditing."""
    
    def test_detect_missing_default_src(self) -> None:
        """Missing default-src is flagged."""
        from csp_validator import CSPValidator, IssueSeverity
        
        validator = CSPValidator()
        policy = validator.parse("script-src 'self'")
        issues = validator.audit(policy)
        
        default_issues = [i for i in issues if i.directive == "default-src"]
        assert len(default_issues) > 0
        assert default_issues[0].severity == IssueSeverity.HIGH
    
    def test_detect_unsafe_inline(self) -> None:
        """'unsafe-inline' is flagged."""
        from csp_validator import CSPValidator, IssueSeverity
        
        validator = CSPValidator(strict_mode=True)
        policy = validator.parse("script-src 'self' 'unsafe-inline'")
        issues = validator.audit(policy)
        
        unsafe_issues = [i for i in issues if "unsafe-inline" in i.message]
        assert len(unsafe_issues) > 0
        assert unsafe_issues[0].severity == IssueSeverity.HIGH
    
    def test_detect_unsafe_eval(self) -> None:
        """'unsafe-eval' is flagged."""
        from csp_validator import CSPValidator
        
        validator = CSPValidator()
        policy = validator.parse("script-src 'self' 'unsafe-eval'")
        issues = validator.audit(policy)
        
        unsafe_issues = [i for i in issues if "unsafe-eval" in i.message]
        assert len(unsafe_issues) > 0
    
    def test_detect_wildcard(self) -> None:
        """Wildcard sources are flagged."""
        from csp_validator import CSPValidator, IssueSeverity
        
        validator = CSPValidator()
        policy = validator.parse("script-src *")
        issues = validator.audit(policy)
        
        wildcard_issues = [i for i in issues if "wildcard" in i.message.lower()]
        assert len(wildcard_issues) > 0
        assert wildcard_issues[0].severity == IssueSeverity.HIGH
    
    def test_detect_missing_frame_ancestors(self) -> None:
        """Missing frame-ancestors (clickjacking) is flagged."""
        from csp_validator import CSPValidator
        
        validator = CSPValidator()
        policy = validator.parse("default-src 'self'")
        issues = validator.audit(policy)
        
        frame_issues = [i for i in issues if i.directive == "frame-ancestors"]
        assert len(frame_issues) > 0
    
    def test_secure_policy_passes(self) -> None:
        """Secure policy has minimal issues."""
        from csp_validator import CSPValidator, IssueSeverity
        
        secure_csp = """
            default-src 'self';
            script-src 'self';
            style-src 'self';
            img-src 'self' data:;
            object-src 'none';
            frame-ancestors 'none';
            upgrade-insecure-requests
        """
        
        validator = CSPValidator()
        policy = validator.parse(secure_csp)
        issues = validator.audit(policy)
        
        # Should have no HIGH/CRITICAL issues
        severe = [i for i in issues if i.severity in [IssueSeverity.HIGH, IssueSeverity.CRITICAL]]
        assert len(severe) == 0


class TestCSPGeneration:
    """Tests for CSP policy generation."""
    
    def test_generate_default_policy(self) -> None:
        """Default secure policy can be generated."""
        from csp_validator import CSPValidator
        
        validator = CSPValidator()
        policy = validator.generate_policy()
        
        assert policy.has_directive("default-src")
        assert policy.has_directive("script-src")
        assert policy.has_directive("frame-ancestors")
    
    def test_generate_custom_policy(self) -> None:
        """Custom policy with overrides."""
        from csp_validator import CSPValidator
        
        validator = CSPValidator()
        policy = validator.generate_policy({
            "connect-src": ["'self'", "https://api.example.com"],
        })
        
        assert "https://api.example.com" in policy.get_sources("connect-src")


class TestXSSDetector:
    """Tests for XSS detection."""
    
    def test_detect_script_tag(self) -> None:
        """Script tags are detected."""
        from xss import XSSDetector, XSSThreatLevel
        
        detector = XSSDetector()
        result = detector.scan("<script>alert('xss')</script>")
        
        assert result.is_safe is False
        assert result.threat_level == XSSThreatLevel.CRITICAL
        assert "script_tag" in result.patterns_matched
    
    def test_detect_event_handler(self) -> None:
        """Event handlers are detected."""
        from xss import XSSDetector, XSSThreatLevel
        
        detector = XSSDetector()
        result = detector.scan('<img src="x" onerror="alert(1)">')
        
        assert result.is_safe is False
        assert result.threat_level == XSSThreatLevel.HIGH
        assert "event_handler" in result.patterns_matched
    
    def test_detect_javascript_protocol(self) -> None:
        """javascript: protocol is detected."""
        from xss import XSSDetector
        
        detector = XSSDetector()
        result = detector.scan('<a href="javascript:alert(1)">Click</a>')
        
        assert result.is_safe is False
        assert "javascript_protocol" in result.patterns_matched
    
    def test_detect_svg_onload(self) -> None:
        """SVG onload attacks are detected."""
        from xss import XSSDetector, XSSThreatLevel
        
        detector = XSSDetector()
        result = detector.scan('<svg onload="alert(1)">')
        
        assert result.is_safe is False
        assert result.threat_level == XSSThreatLevel.CRITICAL
    
    def test_safe_content_passes(self) -> None:
        """Safe content is not flagged."""
        from xss import XSSDetector, XSSThreatLevel
        
        detector = XSSDetector()
        result = detector.scan("Hello, this is safe text with <b>bold</b>")
        
        assert result.is_safe is True
        assert result.threat_level == XSSThreatLevel.NONE
    
    def test_sanitize_removes_xss(self) -> None:
        """Sanitize removes XSS patterns."""
        from xss import XSSDetector
        
        detector = XSSDetector()
        malicious = '<script>alert(1)</script><p>Safe content</p>'
        
        result = detector.scan(malicious)
        
        assert "<script>" not in result.sanitized
        assert "Safe content" in result.sanitized


class TestXSSEncoder:
    """Tests for context-aware encoding."""
    
    def test_encode_html(self) -> None:
        """HTML encoding escapes dangerous characters."""
        from xss import XSSEncoder
        
        encoder = XSSEncoder()
        result = encoder.encode_html("<script>alert('xss')</script>")
        
        assert "<" not in result
        assert "&lt;script&gt;" in result
    
    def test_encode_attribute(self) -> None:
        """Attribute encoding is more aggressive."""
        from xss import XSSEncoder
        
        encoder = XSSEncoder()
        result = encoder.encode_attribute('"><script>alert(1)</script>')
        
        assert '"' not in result
        assert '<' not in result
    
    def test_encode_javascript(self) -> None:
        """JavaScript encoding uses JSON."""
        from xss import XSSEncoder
        
        encoder = XSSEncoder()
        result = encoder.encode_javascript('"; alert(1); "')
        
        # Should be safely escaped for JS string
        assert '"' not in result or '\\"' in result
    
    def test_encode_url(self) -> None:
        """URL encoding percent-encodes special chars."""
        from xss import XSSEncoder
        
        encoder = XSSEncoder()
        result = encoder.encode_url("<script>")
        
        assert "<" not in result
        assert "%3C" in result


class TestHTMLSanitizer:
    """Tests for HTML sanitization."""
    
    def test_sanitize_script_tags(self) -> None:
        """Script tags are removed."""
        from xss import HTMLSanitizer
        
        sanitizer = HTMLSanitizer()
        result = sanitizer.sanitize('<p>Hello</p><script>bad()</script>')
        
        assert "<script>" not in result
        assert "Hello" in result
    
    def test_preserve_safe_tags(self) -> None:
        """Safe tags are preserved."""
        from xss import HTMLSanitizer
        
        sanitizer = HTMLSanitizer()
        result = sanitizer.sanitize('<p>Hello <strong>World</strong></p>')
        
        assert "<p>" in result
        assert "<strong>" in result
    
    def test_remove_event_handlers(self) -> None:
        """Event handler attributes are removed."""
        from xss import HTMLSanitizer
        
        sanitizer = HTMLSanitizer()
        result = sanitizer.sanitize('<div onclick="alert(1)">Click me</div>')
        
        assert "onclick" not in result
        assert "Click me" in result
    
    def test_remove_style_attribute(self) -> None:
        """Style attributes are removed."""
        from xss import HTMLSanitizer
        
        sanitizer = HTMLSanitizer()
        result = sanitizer.sanitize('<p style="background:url(javascript:alert(1))">Text</p>')
        
        assert "style=" not in result


class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_sanitize_html(self) -> None:
        """sanitize_html convenience function works."""
        from xss import sanitize_html
        
        result = sanitize_html('<script>alert(1)</script><p>Safe</p>')
        
        assert "<script>" not in result
    
    def test_escape_html(self) -> None:
        """escape_html convenience function works."""
        from xss import escape_html
        
        result = escape_html('<script>')
        
        assert "&lt;script&gt;" == result
    
    def test_escape_js(self) -> None:
        """escape_js convenience function works."""
        from xss import escape_js
        
        result = escape_js('"test"')
        
        assert '\\"' in result
    
    def test_detect_xss(self) -> None:
        """detect_xss convenience function works."""
        from xss import detect_xss
        
        result = detect_xss('<script>alert(1)</script>')
        
        assert result.is_safe is False
    
    def test_validate_csp_header(self) -> None:
        """validate_csp_header convenience function works."""
        from csp_validator import validate_csp_header
        
        result = validate_csp_header("default-src 'self'; script-src 'self'")
        
        assert "valid" in result
        assert "issues" in result
        assert "issue_count" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
