"""
Web Application Firewall (WAF) Package.

Provides application-layer attack detection and prevention:
- SQL injection detection
- XSS (Cross-Site Scripting) detection
- Command injection detection
- Path traversal detection
- OWASP Core Rule Set compatible

Part of the 6-Month Security Hardening Plan - Month 3.
"""

from shared.security.waf.engine import (
    WAFEngine,
    WAFConfig,
    WAFResult,
    ThreatLevel,
)
from shared.security.waf.rules import (
    WAFRule,
    RuleCategory,
    get_default_rules,
)

__all__ = [
    "WAFEngine",
    "WAFConfig",
    "WAFResult",
    "ThreatLevel",
    "WAFRule",
    "RuleCategory",
    "get_default_rules",
]
