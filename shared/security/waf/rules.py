"""
WAF Rule Definitions.

Defines rules for detecting various web application attacks.
Based on OWASP Core Rule Set (CRS) patterns.

Categories:
- SQL Injection (SQLi)
- Cross-Site Scripting (XSS)
- Command Injection (RCE)
- Path Traversal (LFI/RFI)
- Protocol Attacks
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Pattern


class RuleCategory(Enum):
    """Categories of WAF rules."""
    
    SQL_INJECTION = "sqli"
    XSS = "xss"
    COMMAND_INJECTION = "rce"
    PATH_TRAVERSAL = "lfi"
    PROTOCOL_ATTACK = "protocol"
    REQUEST_SMUGGLING = "smuggling"
    SCANNER_DETECTION = "scanner"
    SESSION_FIXATION = "session"
    JAVA_ATTACK = "java"
    PHP_ATTACK = "php"
    GENERIC_ATTACK = "generic"


class RuleAction(Enum):
    """Actions to take when rule matches."""
    
    BLOCK = "block"           # Block request
    LOG = "log"               # Log but allow
    CHALLENGE = "challenge"   # Present CAPTCHA
    RATE_LIMIT = "rate_limit" # Apply rate limiting


@dataclass
class WAFRule:
    """A single WAF detection rule.
    
    Attributes:
        rule_id: Unique identifier (e.g., "942100" for SQLi)
        name: Human-readable rule name
        category: Attack category
        pattern: Regex pattern to match
        action: Default action when matched
        severity: 1-5 (5 being most severe)
        targets: Where to look (url, body, headers, cookies)
        description: What this rule detects
    """
    
    rule_id: str
    name: str
    category: RuleCategory
    pattern: str
    action: RuleAction = RuleAction.BLOCK
    severity: int = 3
    targets: list[str] = field(default_factory=lambda: ["url", "body"])
    description: str = ""
    enabled: bool = True
    
    _compiled: Optional[Pattern] = field(default=None, repr=False)
    
    def __post_init__(self):
        """Compile the regex pattern."""
        try:
            self._compiled = re.compile(self.pattern, re.IGNORECASE | re.MULTILINE)
        except re.error as e:
            raise ValueError(f"Invalid regex in rule {self.rule_id}: {e}")
    
    def matches(self, content: str) -> Optional[re.Match]:
        """Check if content matches this rule."""
        if not self.enabled or not self._compiled:
            return None
        return self._compiled.search(content)


# =============================================================================
# SQL Injection Rules (OWASP CRS 942xxx)
# =============================================================================

SQL_INJECTION_RULES = [
    WAFRule(
        rule_id="942100",
        name="SQL Injection Attack Detected via libinjection",
        category=RuleCategory.SQL_INJECTION,
        pattern=r"(?i)(?:'|\")?(?:\s*(?:union|select|insert|update|delete|drop|truncate|alter|create|exec|execute)\s)",
        severity=5,
        targets=["url", "body", "headers"],
        description="Detects common SQL keywords in suspicious contexts",
    ),
    WAFRule(
        rule_id="942110",
        name="SQL Injection Attack: Common Injection Testing",
        category=RuleCategory.SQL_INJECTION,
        pattern=r"(?i)(?:'\s*(?:or|and)\s*'?\d|'\s*(?:or|and)\s*'?[a-z]+=)",
        severity=4,
        description="Detects classic OR/AND based SQL injection",
    ),
    WAFRule(
        rule_id="942120",
        name="SQL Injection Attack: SQL Operator Detected",
        category=RuleCategory.SQL_INJECTION,
        pattern=r"(?i)(?:'\s*;|--\s*$|/\*.*\*/|#\s*$)",
        severity=4,
        description="Detects SQL comment and termination patterns",
    ),
    WAFRule(
        rule_id="942130",
        name="SQL Injection Attack: SQL Tautology",
        category=RuleCategory.SQL_INJECTION,
        pattern=r"(?i)(?:'\s*=\s*'|'\s*<>\s*'|1\s*=\s*1|'[^']*'\s*=\s*'[^']*')",
        severity=5,
        description="Detects tautology-based SQL injection (1=1, '1'='1')",
    ),
    WAFRule(
        rule_id="942140",
        name="SQL Injection Attack: DB Names Detected",
        category=RuleCategory.SQL_INJECTION,
        pattern=r"(?i)(?:information_schema|sysobjects|syscolumns|msysace|pg_catalog)",
        severity=4,
        description="Detects database metadata table access attempts",
    ),
    WAFRule(
        rule_id="942150",
        name="SQL Injection Attack: SQL Function Names",
        category=RuleCategory.SQL_INJECTION,
        pattern=r"(?i)(?:concat\s*\(|char\s*\(|substring\s*\(|ascii\s*\(|hex\s*\(|unhex\s*\()",
        severity=3,
        description="Detects SQL function usage in injection context",
    ),
    WAFRule(
        rule_id="942160",
        name="SQL Injection Attack: SLEEP/BENCHMARK",
        category=RuleCategory.SQL_INJECTION,
        pattern=r"(?i)(?:sleep\s*\(\s*\d|benchmark\s*\(|waitfor\s+delay|pg_sleep)",
        severity=5,
        description="Detects time-based blind SQL injection",
    ),
    WAFRule(
        rule_id="942170",
        name="SQL Injection Attack: Stacked Queries",
        category=RuleCategory.SQL_INJECTION,
        pattern=r"(?i);\s*(?:select|insert|update|delete|drop|exec|execute|union)\s",
        severity=5,
        description="Detects stacked SQL queries",
    ),
]


# =============================================================================
# XSS Rules (OWASP CRS 941xxx)
# =============================================================================

XSS_RULES = [
    WAFRule(
        rule_id="941100",
        name="XSS Attack Detected via libinjection",
        category=RuleCategory.XSS,
        pattern=r"<script[^>]*>.*?</script>",
        severity=5,
        targets=["url", "body", "headers"],
        description="Detects inline script tags",
    ),
    WAFRule(
        rule_id="941110",
        name="XSS Filter - Category 1: Script Tag Vector",
        category=RuleCategory.XSS,
        pattern=r"(?i)<script[^>]*(?:src|href)\s*=",
        severity=5,
        description="Detects script tags with external sources",
    ),
    WAFRule(
        rule_id="941120",
        name="XSS Filter - Category 2: Event Handler Vector",
        category=RuleCategory.XSS,
        pattern=r"(?i)(?:on(?:load|error|click|mouse|focus|blur|key|submit|change|abort|resize)\s*=)",
        severity=4,
        description="Detects JavaScript event handlers in HTML",
    ),
    WAFRule(
        rule_id="941130",
        name="XSS Filter - Category 3: Attribute Vector",
        category=RuleCategory.XSS,
        pattern=r"(?i)(?:javascript|vbscript|data):\s*",
        severity=5,
        description="Detects JavaScript/VBScript protocol in attributes",
    ),
    WAFRule(
        rule_id="941140",
        name="XSS Filter - Category 4: CSS Import",
        category=RuleCategory.XSS,
        pattern=r"(?i)(?:@import|expression\s*\(|behavior\s*:)",
        severity=4,
        description="Detects CSS-based XSS vectors",
    ),
    WAFRule(
        rule_id="941150",
        name="XSS Filter - Category 5: Meta Refresh",
        category=RuleCategory.XSS,
        pattern=r"(?i)<meta[^>]*(?:http-equiv\s*=\s*['\"]?refresh)",
        severity=4,
        description="Detects meta refresh for redirection",
    ),
    WAFRule(
        rule_id="941160",
        name="XSS Filter - SVG/Object Tags",
        category=RuleCategory.XSS,
        pattern=r"(?i)<(?:svg|object|embed|applet|iframe|frame|frameset)[^>]*>",
        severity=4,
        description="Detects potentially dangerous HTML tags",
    ),
    WAFRule(
        rule_id="941170",
        name="XSS Filter - Base64 Encoded",
        category=RuleCategory.XSS,
        pattern=r"(?i)(?:data:\s*(?:text|application)/(?:html|javascript)[^,]*,|base64[^,]*,)",
        severity=4,
        description="Detects base64 encoded XSS payloads",
    ),
]


# =============================================================================
# Command Injection Rules (OWASP CRS 932xxx)
# =============================================================================

COMMAND_INJECTION_RULES = [
    WAFRule(
        rule_id="932100",
        name="Remote Command Execution: Unix Command Detected",
        category=RuleCategory.COMMAND_INJECTION,
        pattern=r"(?i)(?:;|\||\|\||&&|\$\(|`)\s*(?:cat|ls|pwd|whoami|id|uname|wget|curl|bash|sh|nc)\b",
        severity=5,
        targets=["url", "body"],
        description="Detects Unix command execution attempts",
    ),
    WAFRule(
        rule_id="932105",
        name="Remote Command Execution: Unix Shell Detected",
        category=RuleCategory.COMMAND_INJECTION,
        pattern=r"(?i)/(?:bin|usr/bin|etc)/(?:bash|sh|csh|ksh|zsh|python|perl|ruby|php)",
        severity=5,
        description="Detects paths to Unix shells",
    ),
    WAFRule(
        rule_id="932110",
        name="Remote Command Execution: Windows Command Detected",
        category=RuleCategory.COMMAND_INJECTION,
        pattern=r"(?i)(?:cmd(?:\.exe)?|powershell(?:\.exe)?)\s*(?:/c|/k|-c|-command)",
        severity=5,
        description="Detects Windows command execution",
    ),
    WAFRule(
        rule_id="932115",
        name="Remote Command Execution: Command Chaining",
        category=RuleCategory.COMMAND_INJECTION,
        pattern=r"(?:;|\|{1,2}|&&|`|\$\()\s*\w+",
        severity=4,
        description="Detects command chaining operators",
    ),
    WAFRule(
        rule_id="932120",
        name="Remote Command Execution: Reverse Shell",
        category=RuleCategory.COMMAND_INJECTION,
        pattern=r"(?i)(?:nc\s+-[lnvp]+|bash\s+-i\s+>|/dev/tcp/|mkfifo|0<&196)",
        severity=5,
        description="Detects reverse shell attempts",
    ),
]


# =============================================================================
# Path Traversal Rules (OWASP CRS 930xxx)
# =============================================================================

PATH_TRAVERSAL_RULES = [
    WAFRule(
        rule_id="930100",
        name="Path Traversal Attack (/../)",
        category=RuleCategory.PATH_TRAVERSAL,
        pattern=r"(?:\.\./|\.\.\\|%2e%2e%2f|%2e%2e/|\.%2e/|%2e\./)",
        severity=4,
        targets=["url", "body"],
        description="Detects directory traversal sequences",
    ),
    WAFRule(
        rule_id="930110",
        name="Path Traversal Attack: OS File Access",
        category=RuleCategory.PATH_TRAVERSAL,
        pattern=r"(?i)(?:/etc/passwd|/etc/shadow|/etc/hosts|/proc/self|/dev/null|boot\.ini|win\.ini)",
        severity=5,
        description="Detects access to sensitive OS files",
    ),
    WAFRule(
        rule_id="930120",
        name="Path Traversal Attack: Encoded Sequences",
        category=RuleCategory.PATH_TRAVERSAL,
        pattern=r"(?:%c0%ae|%c0%2e|%252e|%00)",
        severity=5,
        description="Detects encoded path traversal",
    ),
    WAFRule(
        rule_id="930130",
        name="Local File Inclusion: Common Files",
        category=RuleCategory.PATH_TRAVERSAL,
        pattern=r"(?i)(?:\.htaccess|\.htpasswd|\.bash_history|\.ssh/|id_rsa|authorized_keys)",
        severity=5,
        description="Detects access to sensitive dotfiles",
    ),
]


# =============================================================================
# Protocol Attack Rules (OWASP CRS 920xxx)
# =============================================================================

PROTOCOL_ATTACK_RULES = [
    WAFRule(
        rule_id="920100",
        name="Invalid HTTP Request Line",
        category=RuleCategory.PROTOCOL_ATTACK,
        pattern=r"^(?!GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS|CONNECT|TRACE)",
        severity=3,
        targets=["method"],
        description="Detects invalid HTTP methods",
        action=RuleAction.LOG,  # Log only, don't block
    ),
    WAFRule(
        rule_id="920200",
        name="HTTP Request Smuggling Attack",
        category=RuleCategory.REQUEST_SMUGGLING,
        pattern=r"(?i)(?:content-length.*content-length|transfer-encoding.*chunked.*content-length)",
        severity=5,
        targets=["headers"],
        description="Detects HTTP request smuggling attempts",
    ),
    WAFRule(
        rule_id="920210",
        name="HTTP Response Splitting",
        category=RuleCategory.PROTOCOL_ATTACK,
        pattern=r"(?:\r\n|\n)(?:content-type|set-cookie|location):",
        severity=5,
        targets=["headers", "body"],
        description="Detects HTTP response splitting/injection",
    ),
]


# =============================================================================
# Scanner Detection Rules
# =============================================================================

SCANNER_DETECTION_RULES = [
    WAFRule(
        rule_id="913100",
        name="Scanner Detection: Known Vulnerability Scanners",
        category=RuleCategory.SCANNER_DETECTION,
        pattern=r"(?i)(?:nikto|nessus|nmap|sqlmap|burp|owasp|zap|acunetix|appscan|havij)",
        severity=3,
        targets=["headers", "body", "url"],
        description="Detects known security scanner signatures",
        action=RuleAction.LOG,
    ),
    WAFRule(
        rule_id="913110",
        name="Scanner Detection: Unusual User-Agent",
        category=RuleCategory.SCANNER_DETECTION,
        pattern=r"(?i)(?:python-requests|curl/|wget/|libwww-perl|go-http-client|httpx)",
        severity=2,
        targets=["headers"],
        description="Detects automated HTTP clients",
        action=RuleAction.LOG,
    ),
]


def get_default_rules() -> list[WAFRule]:
    """Get all default WAF rules."""
    return (
        SQL_INJECTION_RULES +
        XSS_RULES +
        COMMAND_INJECTION_RULES +
        PATH_TRAVERSAL_RULES +
        PROTOCOL_ATTACK_RULES +
        SCANNER_DETECTION_RULES
    )


def get_rules_by_category(category: RuleCategory) -> list[WAFRule]:
    """Get rules filtered by category."""
    return [r for r in get_default_rules() if r.category == category]


def get_rule_by_id(rule_id: str) -> Optional[WAFRule]:
    """Get a specific rule by ID."""
    for rule in get_default_rules():
        if rule.rule_id == rule_id:
            return rule
    return None
