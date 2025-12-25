"""
AI Guardrails - Safety and compliance controls for AI endpoints.

Implements content filtering, rate limiting, and PII protection
for all AI-generated content in the platform.

Usage:
    from shared.ai.guardrails import Guardrails, get_guardrails
    
    guardrails = get_guardrails()
    
    # Check input before sending to model
    input_result = await guardrails.check_input(user_message)
    if not input_result.is_safe:
        return {"error": input_result.reason}
    
    # Check output before returning to user
    output_result = await guardrails.check_output(model_response)
    if output_result.requires_redaction:
        return output_result.redacted_content
"""

import hashlib
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any
from collections import defaultdict

logger = logging.getLogger(__name__)


class ContentCategory(str, Enum):
    """Categories of content violations."""
    SAFE = "safe"
    PII = "pii"
    PROFANITY = "profanity"
    INJECTION = "injection"
    HARMFUL = "harmful"
    RATE_LIMITED = "rate_limited"
    POLICY_VIOLATION = "policy_violation"


class Severity(str, Enum):
    """Severity levels for violations."""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class GuardrailResult:
    """Result of a guardrail check."""
    is_safe: bool
    category: ContentCategory
    severity: Severity = Severity.INFO
    reason: str = ""
    details: dict = field(default_factory=dict)
    requires_redaction: bool = False
    redacted_content: str | None = None
    
    def to_dict(self) -> dict:
        return {
            "is_safe": self.is_safe,
            "category": self.category.value,
            "severity": self.severity.value,
            "reason": self.reason,
            "details": self.details,
            "requires_redaction": self.requires_redaction,
        }


@dataclass
class RateLimitConfig:
    """Rate limit configuration."""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    tokens_per_minute: int = 100000
    burst_limit: int = 10


class RateLimiter:
    """Token bucket rate limiter."""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self._requests: dict[str, list[datetime]] = defaultdict(list)
        self._tokens: dict[str, list[tuple[datetime, int]]] = defaultdict(list)
    
    def check(self, client_id: str, tokens: int = 0) -> tuple[bool, str]:
        """Check if request is within rate limits."""
        now = datetime.now(timezone.utc)
        minute_ago = now - timedelta(minutes=1)
        hour_ago = now - timedelta(hours=1)
        
        # Clean old entries
        self._requests[client_id] = [
            t for t in self._requests[client_id] if t > minute_ago
        ]
        
        # Check requests per minute
        if len(self._requests[client_id]) >= self.config.requests_per_minute:
            return False, f"Rate limit: {self.config.requests_per_minute}/min exceeded"
        
        # Check burst
        last_second = now - timedelta(seconds=1)
        burst_count = sum(1 for t in self._requests[client_id] if t > last_second)
        if burst_count >= self.config.burst_limit:
            return False, f"Burst limit: {self.config.burst_limit}/sec exceeded"
        
        # Check token limits
        if tokens > 0:
            self._tokens[client_id] = [
                (t, n) for t, n in self._tokens[client_id] if t > minute_ago
            ]
            total_tokens = sum(n for _, n in self._tokens[client_id])
            if total_tokens + tokens > self.config.tokens_per_minute:
                return False, f"Token limit: {self.config.tokens_per_minute}/min exceeded"
            self._tokens[client_id].append((now, tokens))
        
        # Record request
        self._requests[client_id].append(now)
        return True, ""
    
    def get_remaining(self, client_id: str) -> dict:
        """Get remaining rate limit quota."""
        now = datetime.now(timezone.utc)
        minute_ago = now - timedelta(minutes=1)
        
        requests = [t for t in self._requests.get(client_id, []) if t > minute_ago]
        tokens = sum(n for t, n in self._tokens.get(client_id, []) if t > minute_ago)
        
        return {
            "requests_remaining": max(0, self.config.requests_per_minute - len(requests)),
            "tokens_remaining": max(0, self.config.tokens_per_minute - tokens),
            "reset_at": (minute_ago + timedelta(minutes=1)).isoformat(),
        }


# PII patterns for detection and redaction
PII_PATTERNS = {
    "email": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
    "phone": re.compile(r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'),
    "ssn": re.compile(r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b'),
    "credit_card": re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b'),
    "ip_address": re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'),
    "api_key": re.compile(r'\b(?:sk-|pk-|api[_-]?key[=:\s]+)[A-Za-z0-9-_]{20,}\b', re.I),
}

# Prompt injection patterns
INJECTION_PATTERNS = [
    re.compile(r'\bignore\s+(?:all\s+)?(?:previous|above|prior)\s+instructions?\b', re.I),
    re.compile(r'\byou\s+are\s+now\s+(?:DAN|jailbreak|unrestricted)\b', re.I),
    re.compile(r'\bsystem\s*:\s*', re.I),
    re.compile(r'\b(?:act|pretend|roleplay)\s+as\s+(?:if\s+)?(?:you\s+are\s+)?(?:a\s+)?(?:different|new|evil)\b', re.I),
    re.compile(r'\bforget\s+(?:all\s+)?(?:your\s+)?(?:instructions|rules|guidelines)\b', re.I),
]


class Guardrails:
    """
    AI content guardrails for safety and compliance.
    
    Provides input/output filtering, PII detection, and rate limiting.
    """
    
    def __init__(
        self,
        rate_limit_config: RateLimitConfig | None = None,
        enable_pii_detection: bool = True,
        enable_injection_detection: bool = True,
        custom_blocklist: list[str] | None = None,
    ):
        """Initialize guardrails."""
        self.rate_limiter = RateLimiter(rate_limit_config or RateLimitConfig())
        self.enable_pii_detection = enable_pii_detection
        self.enable_injection_detection = enable_injection_detection
        self.blocklist = set(word.lower() for word in (custom_blocklist or []))
        
        # Load blocklist from environment if configured
        blocklist_file = os.getenv("AI_BLOCKLIST_FILE")
        if blocklist_file and os.path.exists(blocklist_file):
            with open(blocklist_file) as f:
                for line in f:
                    word = line.strip().lower()
                    if word and not word.startswith("#"):
                        self.blocklist.add(word)
        
        logger.info(f"Guardrails initialized (PII: {enable_pii_detection}, injection: {enable_injection_detection})")
    
    async def check_input(
        self,
        content: str,
        client_id: str = "anonymous",
        token_estimate: int = 0,
    ) -> GuardrailResult:
        """
        Check user input before sending to AI model.
        
        Args:
            content: User input to check
            client_id: Client identifier for rate limiting
            token_estimate: Estimated tokens (for rate limiting)
            
        Returns:
            GuardrailResult with safety assessment
        """
        # Rate limit check
        allowed, reason = self.rate_limiter.check(client_id, token_estimate)
        if not allowed:
            return GuardrailResult(
                is_safe=False,
                category=ContentCategory.RATE_LIMITED,
                severity=Severity.MEDIUM,
                reason=reason,
            )
        
        # Injection detection
        if self.enable_injection_detection:
            for pattern in INJECTION_PATTERNS:
                if pattern.search(content):
                    return GuardrailResult(
                        is_safe=False,
                        category=ContentCategory.INJECTION,
                        severity=Severity.HIGH,
                        reason="Potential prompt injection detected",
                        details={"pattern": pattern.pattern},
                    )
        
        # Blocklist check
        content_lower = content.lower()
        for word in self.blocklist:
            if word in content_lower:
                return GuardrailResult(
                    is_safe=False,
                    category=ContentCategory.POLICY_VIOLATION,
                    severity=Severity.MEDIUM,
                    reason="Content contains blocked term",
                )
        
        return GuardrailResult(
            is_safe=True,
            category=ContentCategory.SAFE,
        )
    
    async def check_output(
        self,
        content: str,
        redact_pii: bool = True,
    ) -> GuardrailResult:
        """
        Check AI output before returning to user.
        
        Args:
            content: AI-generated content
            redact_pii: Whether to redact detected PII
            
        Returns:
            GuardrailResult with safety assessment and optional redaction
        """
        detected_pii = []
        redacted = content
        
        # PII detection and redaction
        if self.enable_pii_detection:
            for pii_type, pattern in PII_PATTERNS.items():
                matches = pattern.findall(content)
                if matches:
                    detected_pii.append({"type": pii_type, "count": len(matches)})
                    
                    if redact_pii:
                        redacted = pattern.sub(f"[REDACTED_{pii_type.upper()}]", redacted)
        
        if detected_pii:
            return GuardrailResult(
                is_safe=False,
                category=ContentCategory.PII,
                severity=Severity.HIGH,
                reason="PII detected in output",
                details={"pii_types": detected_pii},
                requires_redaction=True,
                redacted_content=redacted if redact_pii else None,
            )
        
        # Blocklist check on output
        content_lower = content.lower()
        for word in self.blocklist:
            if word in content_lower:
                return GuardrailResult(
                    is_safe=False,
                    category=ContentCategory.POLICY_VIOLATION,
                    severity=Severity.MEDIUM,
                    reason="Output contains blocked term",
                )
        
        return GuardrailResult(
            is_safe=True,
            category=ContentCategory.SAFE,
        )
    
    def detect_pii(self, content: str) -> list[dict]:
        """Detect PII in content without redaction."""
        findings = []
        for pii_type, pattern in PII_PATTERNS.items():
            matches = pattern.findall(content)
            for match in matches:
                findings.append({
                    "type": pii_type,
                    "value": match,
                    "masked": self._mask_value(match, pii_type),
                })
        return findings
    
    def redact_pii(self, content: str) -> str:
        """Redact all PII from content."""
        redacted = content
        for pii_type, pattern in PII_PATTERNS.items():
            redacted = pattern.sub(f"[REDACTED_{pii_type.upper()}]", redacted)
        return redacted
    
    def _mask_value(self, value: str, pii_type: str) -> str:
        """Partially mask a PII value."""
        if pii_type == "email":
            parts = value.split("@")
            if len(parts) == 2:
                return f"{parts[0][:2]}***@{parts[1]}"
        elif pii_type == "phone":
            digits = re.sub(r'\D', '', value)
            if len(digits) >= 4:
                return f"***-***-{digits[-4:]}"
        elif pii_type == "ssn":
            return "***-**-" + value[-4:]
        elif pii_type == "credit_card":
            digits = re.sub(r'\D', '', value)
            return f"****-****-****-{digits[-4:]}"
        return "***"
    
    def get_rate_limit_status(self, client_id: str) -> dict:
        """Get rate limit status for a client."""
        return self.rate_limiter.get_remaining(client_id)


# Singleton instance
_guardrails: Guardrails | None = None


def get_guardrails() -> Guardrails:
    """Get or create guardrails singleton."""
    global _guardrails
    if _guardrails is None:
        _guardrails = Guardrails()
    return _guardrails
