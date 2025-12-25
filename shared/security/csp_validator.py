"""
CSP Validator - Content Security Policy validation and enforcement.

Provides CSP policy parsing, validation, and compliance checking
for enterprise frontend security.

Usage:
    from shared.security.csp_validator import CSPValidator, CSPPolicy
    
    validator = CSPValidator()
    
    # Parse and validate a CSP header
    policy = validator.parse("default-src 'self'; script-src 'self' 'unsafe-inline'")
    
    # Check for security issues
    issues = validator.audit(policy)
    for issue in issues:
        print(f"{issue.severity}: {issue.message}")
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class CSPDirective(str, Enum):
    """CSP directive types."""
    DEFAULT_SRC = "default-src"
    SCRIPT_SRC = "script-src"
    STYLE_SRC = "style-src"
    IMG_SRC = "img-src"
    FONT_SRC = "font-src"
    CONNECT_SRC = "connect-src"
    MEDIA_SRC = "media-src"
    OBJECT_SRC = "object-src"
    FRAME_SRC = "frame-src"
    CHILD_SRC = "child-src"
    WORKER_SRC = "worker-src"
    FRAME_ANCESTORS = "frame-ancestors"
    FORM_ACTION = "form-action"
    BASE_URI = "base-uri"
    REPORT_URI = "report-uri"
    REPORT_TO = "report-to"
    UPGRADE_INSECURE_REQUESTS = "upgrade-insecure-requests"
    BLOCK_ALL_MIXED_CONTENT = "block-all-mixed-content"


class IssueSeverity(str, Enum):
    """Security issue severity."""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class CSPIssue:
    """A CSP security issue."""
    directive: str
    severity: IssueSeverity
    message: str
    recommendation: str = ""
    
    def to_dict(self) -> dict:
        return {
            "directive": self.directive,
            "severity": self.severity.value,
            "message": self.message,
            "recommendation": self.recommendation,
        }


@dataclass
class CSPPolicy:
    """Parsed CSP policy."""
    raw: str
    directives: dict[str, list[str]] = field(default_factory=dict)
    
    def get_sources(self, directive: str) -> list[str]:
        """Get sources for a directive."""
        return self.directives.get(directive, [])
    
    def has_directive(self, directive: str) -> bool:
        """Check if directive is present."""
        return directive in self.directives
    
    def to_header(self) -> str:
        """Convert back to CSP header string."""
        parts = []
        for directive, sources in self.directives.items():
            if sources:
                parts.append(f"{directive} {' '.join(sources)}")
            else:
                parts.append(directive)
        return "; ".join(parts)


# Unsafe CSP values that should be avoided
UNSAFE_VALUES = {
    "'unsafe-inline'": "Allows inline scripts/styles, enabling XSS",
    "'unsafe-eval'": "Allows eval(), enabling code injection",
    "data:": "Allows data: URIs, can bypass CSP",
    "*": "Wildcard allows any source",
}

# Recommended enterprise CSP policy
RECOMMENDED_POLICY = {
    "default-src": ["'self'"],
    "script-src": ["'self'"],
    "style-src": ["'self'"],
    "img-src": ["'self'", "data:", "https:"],
    "font-src": ["'self'", "https://fonts.gstatic.com"],
    "connect-src": ["'self'"],
    "frame-ancestors": ["'none'"],
    "base-uri": ["'self'"],
    "form-action": ["'self'"],
    "object-src": ["'none'"],
    "upgrade-insecure-requests": [],
}


class CSPValidator:
    """
    Content Security Policy validator and auditor.
    
    Parses CSP headers, identifies security issues, and provides
    recommendations for enterprise-grade policies.
    """
    
    def __init__(self, strict_mode: bool = True):
        """
        Initialize CSP validator.
        
        Args:
            strict_mode: If True, flags all unsafe values as issues
        """
        self.strict_mode = strict_mode
    
    def parse(self, csp_header: str) -> CSPPolicy:
        """
        Parse a CSP header string into a policy object.
        
        Args:
            csp_header: Raw CSP header value
            
        Returns:
            Parsed CSPPolicy
        """
        policy = CSPPolicy(raw=csp_header)
        
        # Split by semicolons
        parts = [p.strip() for p in csp_header.split(";") if p.strip()]
        
        for part in parts:
            tokens = part.split()
            if not tokens:
                continue
            
            directive = tokens[0].lower()
            sources = tokens[1:] if len(tokens) > 1 else []
            
            policy.directives[directive] = sources
        
        return policy
    
    def audit(self, policy: CSPPolicy) -> list[CSPIssue]:
        """
        Audit a CSP policy for security issues.
        
        Args:
            policy: Parsed CSP policy
            
        Returns:
            List of security issues found
        """
        issues = []
        
        # Check for missing critical directives
        issues.extend(self._check_missing_directives(policy))
        
        # Check for unsafe values
        issues.extend(self._check_unsafe_values(policy))
        
        # Check for overly permissive sources
        issues.extend(self._check_permissive_sources(policy))
        
        # Check for clickjacking protection
        issues.extend(self._check_clickjacking(policy))
        
        # Check for HTTPS enforcement
        issues.extend(self._check_https(policy))
        
        return issues
    
    def _check_missing_directives(self, policy: CSPPolicy) -> list[CSPIssue]:
        """Check for missing important directives."""
        issues = []
        
        critical_directives = [
            ("default-src", "Fallback for other directives"),
            ("script-src", "Controls script execution"),
            ("object-src", "Controls plugins like Flash"),
        ]
        
        for directive, description in critical_directives:
            if not policy.has_directive(directive):
                issues.append(CSPIssue(
                    directive=directive,
                    severity=IssueSeverity.HIGH,
                    message=f"Missing {directive} directive",
                    recommendation=f"Add {directive} directive. {description}",
                ))
        
        return issues
    
    def _check_unsafe_values(self, policy: CSPPolicy) -> list[CSPIssue]:
        """Check for unsafe CSP values."""
        issues = []
        
        for directive, sources in policy.directives.items():
            for source in sources:
                source_lower = source.lower()
                
                if source_lower in ["'unsafe-inline'", "'unsafe-eval'"]:
                    issues.append(CSPIssue(
                        directive=directive,
                        severity=IssueSeverity.HIGH if self.strict_mode else IssueSeverity.MEDIUM,
                        message=f"{directive} contains {source}",
                        recommendation=UNSAFE_VALUES.get(source_lower, "Remove unsafe value"),
                    ))
                
                elif source_lower == "*":
                    issues.append(CSPIssue(
                        directive=directive,
                        severity=IssueSeverity.HIGH,
                        message=f"{directive} uses wildcard (*)",
                        recommendation="Replace wildcard with specific sources",
                    ))
        
        return issues
    
    def _check_permissive_sources(self, policy: CSPPolicy) -> list[CSPIssue]:
        """Check for overly permissive source configurations."""
        issues = []
        
        # Check for http: in script-src
        script_sources = policy.get_sources("script-src")
        if "http:" in script_sources:
            issues.append(CSPIssue(
                directive="script-src",
                severity=IssueSeverity.CRITICAL,
                message="script-src allows http: protocol",
                recommendation="Use https: only for script sources",
            ))
        
        # Check for data: in script-src
        if "data:" in script_sources:
            issues.append(CSPIssue(
                directive="script-src",
                severity=IssueSeverity.HIGH,
                message="script-src allows data: URIs",
                recommendation="Remove data: from script-src",
            ))
        
        return issues
    
    def _check_clickjacking(self, policy: CSPPolicy) -> list[CSPIssue]:
        """Check for clickjacking protection."""
        issues = []
        
        if not policy.has_directive("frame-ancestors"):
            issues.append(CSPIssue(
                directive="frame-ancestors",
                severity=IssueSeverity.MEDIUM,
                message="Missing frame-ancestors directive",
                recommendation="Add frame-ancestors 'self' or 'none' to prevent clickjacking",
            ))
        else:
            sources = policy.get_sources("frame-ancestors")
            if "*" in sources:
                issues.append(CSPIssue(
                    directive="frame-ancestors",
                    severity=IssueSeverity.HIGH,
                    message="frame-ancestors allows any origin",
                    recommendation="Restrict frame-ancestors to specific origins",
                ))
        
        return issues
    
    def _check_https(self, policy: CSPPolicy) -> list[CSPIssue]:
        """Check for HTTPS enforcement."""
        issues = []
        
        if not policy.has_directive("upgrade-insecure-requests"):
            if not policy.has_directive("block-all-mixed-content"):
                issues.append(CSPIssue(
                    directive="upgrade-insecure-requests",
                    severity=IssueSeverity.LOW,
                    message="Missing HTTPS upgrade directive",
                    recommendation="Add upgrade-insecure-requests for HTTPS enforcement",
                ))
        
        return issues
    
    def generate_policy(self, options: dict[str, list[str]] | None = None) -> CSPPolicy:
        """
        Generate a secure CSP policy.
        
        Args:
            options: Optional overrides for default policy
            
        Returns:
            Generated CSPPolicy
        """
        directives = dict(RECOMMENDED_POLICY)
        
        if options:
            directives.update(options)
        
        # Build header string
        parts = []
        for directive, sources in directives.items():
            if sources:
                parts.append(f"{directive} {' '.join(sources)}")
            else:
                parts.append(directive)
        
        return self.parse("; ".join(parts))
    
    def compare_policies(self, current: CSPPolicy, recommended: CSPPolicy) -> list[dict]:
        """Compare current policy against recommended."""
        differences = []
        
        all_directives = set(current.directives.keys()) | set(recommended.directives.keys())
        
        for directive in all_directives:
            current_sources = set(current.get_sources(directive))
            recommended_sources = set(recommended.get_sources(directive))
            
            if current_sources != recommended_sources:
                differences.append({
                    "directive": directive,
                    "current": list(current_sources),
                    "recommended": list(recommended_sources),
                    "missing": list(recommended_sources - current_sources),
                    "extra": list(current_sources - recommended_sources),
                })
        
        return differences


def validate_csp_header(csp_header: str) -> dict:
    """
    Convenience function to validate a CSP header.
    
    Returns dict with policy info and any issues.
    """
    validator = CSPValidator()
    policy = validator.parse(csp_header)
    issues = validator.audit(policy)
    
    return {
        "valid": len([i for i in issues if i.severity in [IssueSeverity.HIGH, IssueSeverity.CRITICAL]]) == 0,
        "directives": list(policy.directives.keys()),
        "issues": [i.to_dict() for i in issues],
        "issue_count": {
            "critical": len([i for i in issues if i.severity == IssueSeverity.CRITICAL]),
            "high": len([i for i in issues if i.severity == IssueSeverity.HIGH]),
            "medium": len([i for i in issues if i.severity == IssueSeverity.MEDIUM]),
            "low": len([i for i in issues if i.severity == IssueSeverity.LOW]),
        },
    }
