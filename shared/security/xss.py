"""
XSS Prevention Utilities - Cross-Site Scripting protection.

Provides input sanitization, output encoding, and XSS detection
for enterprise security compliance.

Usage:
    from shared.security.xss import XSSProtection, sanitize_html, escape_js
    
    # Sanitize HTML content
    safe_html = sanitize_html(user_input)
    
    # Escape for JavaScript context
    safe_js = escape_js(user_input)
    
    # Detect XSS attempts
    detector = XSSDetector()
    result = detector.scan(user_input)
"""

import html
import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class XSSContext(str, Enum):
    """Context for output encoding."""
    HTML_BODY = "html_body"
    HTML_ATTRIBUTE = "html_attribute"
    JAVASCRIPT = "javascript"
    URL = "url"
    CSS = "css"


class XSSThreatLevel(str, Enum):
    """XSS threat classification."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class XSSScanResult:
    """Result of XSS scan."""
    is_safe: bool
    threat_level: XSSThreatLevel
    patterns_matched: list[str] = field(default_factory=list)
    sanitized: str = ""
    details: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "is_safe": self.is_safe,
            "threat_level": self.threat_level.value,
            "patterns_matched": self.patterns_matched,
            "sanitized": self.sanitized,
            "details": self.details,
        }


# XSS attack patterns
XSS_PATTERNS = [
    # Script tags
    (re.compile(r'<script\b[^>]*>.*?</script>', re.I | re.S), "script_tag", XSSThreatLevel.CRITICAL),
    (re.compile(r'<script\b[^>]*>', re.I), "script_open", XSSThreatLevel.CRITICAL),
    
    # Event handlers
    (re.compile(r'\bon\w+\s*=', re.I), "event_handler", XSSThreatLevel.HIGH),
    
    # JavaScript protocol
    (re.compile(r'javascript\s*:', re.I), "javascript_protocol", XSSThreatLevel.HIGH),
    (re.compile(r'vbscript\s*:', re.I), "vbscript_protocol", XSSThreatLevel.HIGH),
    
    # Data URLs with script
    (re.compile(r'data\s*:[^,]*;base64', re.I), "data_url_base64", XSSThreatLevel.MEDIUM),
    
    # Expression in CSS
    (re.compile(r'expression\s*\(', re.I), "css_expression", XSSThreatLevel.HIGH),
    
    # Object/embed/iframe
    (re.compile(r'<(object|embed|iframe)\b', re.I), "dangerous_tag", XSSThreatLevel.HIGH),
    
    # SVG with script
    (re.compile(r'<svg\b[^>]*onload', re.I), "svg_onload", XSSThreatLevel.CRITICAL),
    
    # Template injection
    (re.compile(r'\{\{.*\}\}'), "template_injection", XSSThreatLevel.MEDIUM),
    (re.compile(r'\$\{.*\}'), "template_literal", XSSThreatLevel.MEDIUM),
    
    # Encoded XSS attempts
    (re.compile(r'&#x?[0-9a-f]+;', re.I), "html_encoded", XSSThreatLevel.LOW),
    (re.compile(r'%3C|%3E|%22|%27', re.I), "url_encoded", XSSThreatLevel.LOW),
]

# Allowed HTML tags for sanitization
ALLOWED_TAGS = {
    'a', 'abbr', 'acronym', 'b', 'blockquote', 'br', 'code', 'div',
    'em', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'hr', 'i', 'li',
    'ol', 'p', 'pre', 'span', 'strong', 'table', 'tbody', 'td',
    'th', 'thead', 'tr', 'ul',
}

# Allowed attributes per tag
ALLOWED_ATTRS = {
    'a': {'href', 'title', 'class'},
    'abbr': {'title'},
    'acronym': {'title'},
    'div': {'class', 'id'},
    'span': {'class'},
    'table': {'class'},
    'td': {'colspan', 'rowspan'},
    'th': {'colspan', 'rowspan', 'scope'},
}

# Dangerous attributes to always remove
DANGEROUS_ATTRS = {
    'onclick', 'ondblclick', 'onmousedown', 'onmouseup', 'onmouseover',
    'onmouseout', 'onmousemove', 'onkeydown', 'onkeyup', 'onkeypress',
    'onfocus', 'onblur', 'onchange', 'onsubmit', 'onreset', 'onload',
    'onerror', 'onabort', 'style', 'srcdoc', 'formaction',
}


class XSSDetector:
    """
    XSS attack detector and classifier.
    
    Scans input for common XSS patterns and classifies threat level.
    """
    
    def __init__(self, custom_patterns: list[tuple] | None = None):
        """
        Initialize XSS detector.
        
        Args:
            custom_patterns: Additional patterns to check
        """
        self.patterns = list(XSS_PATTERNS)
        if custom_patterns:
            self.patterns.extend(custom_patterns)
    
    def scan(self, content: str) -> XSSScanResult:
        """
        Scan content for XSS patterns.
        
        Args:
            content: Content to scan
            
        Returns:
            XSSScanResult with threat assessment
        """
        if not content:
            return XSSScanResult(
                is_safe=True,
                threat_level=XSSThreatLevel.NONE,
                sanitized=content,
            )
        
        matched = []
        max_threat = XSSThreatLevel.NONE
        
        for pattern, name, threat_level in self.patterns:
            if pattern.search(content):
                matched.append(name)
                if self._threat_level_rank(threat_level) > self._threat_level_rank(max_threat):
                    max_threat = threat_level
        
        # Sanitize the content
        sanitized = self.sanitize(content)
        
        return XSSScanResult(
            is_safe=len(matched) == 0,
            threat_level=max_threat,
            patterns_matched=matched,
            sanitized=sanitized,
            details={
                "original_length": len(content),
                "sanitized_length": len(sanitized),
                "patterns_checked": len(self.patterns),
            },
        )
    
    def sanitize(self, content: str) -> str:
        """Remove XSS patterns from content."""
        result = content
        
        # Remove script tags
        result = re.sub(r'<script\b[^>]*>.*?</script>', '', result, flags=re.I | re.S)
        result = re.sub(r'<script\b[^>]*>', '', result, flags=re.I)
        
        # Remove event handlers
        result = re.sub(r'\s+on\w+\s*=\s*["\'][^"\']*["\']', '', result, flags=re.I)
        result = re.sub(r'\s+on\w+\s*=\s*\S+', '', result, flags=re.I)
        
        # Remove javascript: protocol
        result = re.sub(r'javascript\s*:', '', result, flags=re.I)
        
        # Remove dangerous tags
        result = re.sub(r'<(script|iframe|object|embed|applet|form)\b[^>]*>.*?</\1>', '', result, flags=re.I | re.S)
        result = re.sub(r'<(script|iframe|object|embed|applet|form)\b[^>]*/?>', '', result, flags=re.I)
        
        return result
    
    def _threat_level_rank(self, level: XSSThreatLevel) -> int:
        """Get numeric rank for threat level."""
        ranks = {
            XSSThreatLevel.NONE: 0,
            XSSThreatLevel.LOW: 1,
            XSSThreatLevel.MEDIUM: 2,
            XSSThreatLevel.HIGH: 3,
            XSSThreatLevel.CRITICAL: 4,
        }
        return ranks.get(level, 0)


class XSSEncoder:
    """
    Context-aware output encoder for XSS prevention.
    
    Encodes output based on the context where it will be rendered.
    """
    
    @staticmethod
    def encode_html(content: str) -> str:
        """Encode for HTML body context."""
        return html.escape(content, quote=True)
    
    @staticmethod
    def encode_attribute(content: str) -> str:
        """Encode for HTML attribute context."""
        # More aggressive encoding for attributes
        result = content.replace('&', '&amp;')
        result = result.replace('<', '&lt;')
        result = result.replace('>', '&gt;')
        result = result.replace('"', '&quot;')
        result = result.replace("'", '&#x27;')
        result = result.replace('/', '&#x2F;')
        result = result.replace('`', '&#x60;')
        return result
    
    @staticmethod
    def encode_javascript(content: str) -> str:
        """Encode for JavaScript context."""
        # Use JSON encoding for safety
        return json.dumps(content)[1:-1]  # Remove surrounding quotes
    
    @staticmethod
    def encode_url(content: str) -> str:
        """Encode for URL context."""
        from urllib.parse import quote
        return quote(content, safe='')
    
    @staticmethod
    def encode_css(content: str) -> str:
        """Encode for CSS context."""
        # Remove potentially dangerous characters
        result = re.sub(r'[<>"\'&()\\]', '', content)
        # CSS escape special characters
        result = re.sub(r'([^a-zA-Z0-9_-])', lambda m: f'\\{ord(m.group(1)):x} ', result)
        return result
    
    def encode(self, content: str, context: XSSContext) -> str:
        """
        Encode content for specific context.
        
        Args:
            content: Content to encode
            context: Output context
            
        Returns:
            Safely encoded content
        """
        encoders = {
            XSSContext.HTML_BODY: self.encode_html,
            XSSContext.HTML_ATTRIBUTE: self.encode_attribute,
            XSSContext.JAVASCRIPT: self.encode_javascript,
            XSSContext.URL: self.encode_url,
            XSSContext.CSS: self.encode_css,
        }
        
        encoder = encoders.get(context, self.encode_html)
        return encoder(content)


class HTMLSanitizer:
    """
    HTML content sanitizer with allowlist approach.
    
    Removes dangerous HTML while preserving safe formatting.
    """
    
    def __init__(
        self,
        allowed_tags: set[str] | None = None,
        allowed_attrs: dict[str, set[str]] | None = None,
    ):
        """
        Initialize sanitizer.
        
        Args:
            allowed_tags: Set of allowed HTML tags
            allowed_attrs: Dict of tag -> allowed attributes
        """
        self.allowed_tags = allowed_tags or ALLOWED_TAGS
        self.allowed_attrs = allowed_attrs or ALLOWED_ATTRS
    
    def sanitize(self, html_content: str) -> str:
        """
        Sanitize HTML content.
        
        Args:
            html_content: HTML to sanitize
            
        Returns:
            Sanitized HTML
        """
        try:
            from html.parser import HTMLParser
        except ImportError:
            # Fallback to basic sanitization
            return self._basic_sanitize(html_content)
        
        return self._sanitize_with_parser(html_content)
    
    def _basic_sanitize(self, content: str) -> str:
        """Basic sanitization without parser."""
        result = content
        
        # Remove script tags
        result = re.sub(r'<script\b[^>]*>.*?</script>', '', result, flags=re.I | re.S)
        
        # Remove event handlers
        result = re.sub(r'\s+on\w+\s*=\s*["\'][^"\']*["\']', '', result, flags=re.I)
        
        # Remove style attributes
        result = re.sub(r'\s+style\s*=\s*["\'][^"\']*["\']', '', result, flags=re.I)
        
        return result
    
    def _sanitize_with_parser(self, content: str) -> str:
        """Sanitize using HTML parser."""
        # Simple regex-based sanitization for allowed tags
        result = content
        
        # Remove all tags not in allowed list
        def replace_tag(match):
            tag = match.group(1).lower()
            if tag.lstrip('/') in self.allowed_tags:
                return match.group(0)
            return ''
        
        result = re.sub(r'<(/?\w+)[^>]*>', replace_tag, result)
        
        # Remove dangerous attributes
        for attr in DANGEROUS_ATTRS:
            result = re.sub(rf'\s+{attr}\s*=\s*["\'][^"\']*["\']', '', result, flags=re.I)
            result = re.sub(rf'\s+{attr}\s*=\s*\S+', '', result, flags=re.I)
        
        return result


# Convenience functions
def sanitize_html(content: str) -> str:
    """Sanitize HTML content using default settings."""
    sanitizer = HTMLSanitizer()
    return sanitizer.sanitize(content)


def escape_html(content: str) -> str:
    """Escape content for HTML body."""
    return XSSEncoder.encode_html(content)


def escape_js(content: str) -> str:
    """Escape content for JavaScript context."""
    return XSSEncoder.encode_javascript(content)


def escape_url(content: str) -> str:
    """Escape content for URL context."""
    return XSSEncoder.encode_url(content)


def detect_xss(content: str) -> XSSScanResult:
    """Convenience function to scan for XSS."""
    detector = XSSDetector()
    return detector.scan(content)
