"""
Security Policy Engine.

Enforces organizational security policies across the application:
- Password complexity requirements
- Session management policies
- Data handling policies
- Access control policies

Policies are defined declaratively and enforced programmatically.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Optional

import logging

logger = logging.getLogger(__name__)


class PolicyAction(Enum):
    """Actions when policy is violated."""
    
    DENY = "deny"           # Block the action
    WARN = "warn"           # Allow but warn
    AUDIT = "audit"         # Allow and log
    QUARANTINE = "quarantine"  # Isolate the request


class PolicyCategory(Enum):
    """Policy categories."""
    
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_HANDLING = "data_handling"
    SESSION = "session"
    NETWORK = "network"
    ENCRYPTION = "encryption"


@dataclass
class PolicyViolation:
    """Record of a policy violation."""
    
    policy_id: str
    policy_name: str
    category: PolicyCategory
    severity: int  # 1-10
    message: str
    action: PolicyAction
    context: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class PolicyResult:
    """Result of policy evaluation."""
    
    passed: bool
    violations: list[PolicyViolation] = field(default_factory=list)
    
    @property
    def should_deny(self) -> bool:
        """Check if any violation requires denial."""
        return any(v.action == PolicyAction.DENY for v in self.violations)
    
    @property
    def highest_severity(self) -> int:
        """Get highest severity among violations."""
        if not self.violations:
            return 0
        return max(v.severity for v in self.violations)


@dataclass
class SecurityPolicy:
    """A single security policy definition."""
    
    policy_id: str
    name: str
    description: str
    category: PolicyCategory
    severity: int = 5
    action: PolicyAction = PolicyAction.DENY
    enabled: bool = True
    
    # Policy rule function
    rule: Optional[Callable[..., bool]] = None
    
    def evaluate(self, **context) -> Optional[PolicyViolation]:
        """Evaluate the policy.
        
        Returns:
            PolicyViolation if violated, None if passed
        """
        if not self.enabled:
            return None
        
        if self.rule and not self.rule(**context):
            return PolicyViolation(
                policy_id=self.policy_id,
                policy_name=self.name,
                category=self.category,
                severity=self.severity,
                message=self.description,
                action=self.action,
                context=context,
            )
        return None


class PolicyEngine:
    """Security policy enforcement engine."""
    
    def __init__(self):
        """Initialize the policy engine."""
        self._policies: dict[str, SecurityPolicy] = {}
        self._load_default_policies()
        logger.info("PolicyEngine initialized with %d policies", len(self._policies))
    
    def _load_default_policies(self) -> None:
        """Load default security policies."""
        
        # Password policies
        self.add_policy(SecurityPolicy(
            policy_id="PWD-001",
            name="Password Minimum Length",
            description="Password must be at least 12 characters",
            category=PolicyCategory.AUTHENTICATION,
            severity=8,
            rule=lambda password="", **_: len(password) >= 12,
        ))
        
        self.add_policy(SecurityPolicy(
            policy_id="PWD-002",
            name="Password Complexity",
            description="Password must contain uppercase, lowercase, number, and symbol",
            category=PolicyCategory.AUTHENTICATION,
            severity=7,
            rule=lambda password="", **_: (
                bool(re.search(r"[A-Z]", password)) and
                bool(re.search(r"[a-z]", password)) and
                bool(re.search(r"\d", password)) and
                bool(re.search(r"[!@#$%^&*(),.?\":{}|<>]", password))
            ),
        ))
        
        self.add_policy(SecurityPolicy(
            policy_id="PWD-003",
            name="Password No Common Patterns",
            description="Password must not contain common patterns",
            category=PolicyCategory.AUTHENTICATION,
            severity=6,
            rule=lambda password="", **_: not any(
                p in password.lower() for p in [
                    "password", "123456", "qwerty", "admin", "letmein",
                ]
            ),
        ))
        
        # Session policies
        self.add_policy(SecurityPolicy(
            policy_id="SES-001",
            name="Session Maximum Age",
            description="Session must not exceed 24 hours",
            category=PolicyCategory.SESSION,
            severity=6,
            rule=lambda session_age_hours=0, **_: session_age_hours <= 24,
        ))
        
        self.add_policy(SecurityPolicy(
            policy_id="SES-002",
            name="Session Idle Timeout",
            description="Session must not be idle for more than 30 minutes",
            category=PolicyCategory.SESSION,
            severity=5,
            rule=lambda idle_minutes=0, **_: idle_minutes <= 30,
        ))
        
        # Data handling policies
        self.add_policy(SecurityPolicy(
            policy_id="DATA-001",
            name="No PII in Logs",
            description="Logs must not contain PII (email, SSN, phone)",
            category=PolicyCategory.DATA_HANDLING,
            severity=9,
            rule=lambda log_content="", **_: not any([
                re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", log_content),
                re.search(r"\b\d{3}-\d{2}-\d{4}\b", log_content),
                re.search(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", log_content),
            ]),
        ))
        
        self.add_policy(SecurityPolicy(
            policy_id="DATA-002",
            name="Encryption Required for Sensitive Data",
            description="Sensitive data must be encrypted at rest",
            category=PolicyCategory.DATA_HANDLING,
            severity=10,
            rule=lambda is_encrypted=True, is_sensitive=False, **_: is_encrypted or not is_sensitive,
        ))
        
        # Network policies
        self.add_policy(SecurityPolicy(
            policy_id="NET-001",
            name="TLS Required",
            description="All connections must use TLS 1.2 or higher",
            category=PolicyCategory.NETWORK,
            severity=10,
            rule=lambda tls_version="", **_: tls_version in ["1.2", "1.3", "TLSv1.2", "TLSv1.3"] if tls_version else True,
        ))
        
        self.add_policy(SecurityPolicy(
            policy_id="NET-002",
            name="No Wildcard CORS",
            description="CORS must not use wildcard origins",
            category=PolicyCategory.NETWORK,
            severity=7,
            rule=lambda cors_origin="", **_: cors_origin != "*",
        ))
        
        # Encryption policies
        self.add_policy(SecurityPolicy(
            policy_id="ENC-001",
            name="AES Key Length",
            description="AES keys must be at least 256 bits",
            category=PolicyCategory.ENCRYPTION,
            severity=9,
            rule=lambda key_bits=256, **_: key_bits >= 256,
        ))
        
        self.add_policy(SecurityPolicy(
            policy_id="ENC-002",
            name="No Deprecated Algorithms",
            description="MD5, SHA1, DES, 3DES, RC4 are not allowed",
            category=PolicyCategory.ENCRYPTION,
            severity=10,
            rule=lambda algorithm="", **_: algorithm.upper() not in [
                "MD5", "SHA1", "SHA-1", "DES", "3DES", "RC4", "RC2",
            ],
        ))
        
        # Authorization policies
        self.add_policy(SecurityPolicy(
            policy_id="AUTHZ-001",
            name="Least Privilege",
            description="Access must be granted based on least privilege principle",
            category=PolicyCategory.AUTHORIZATION,
            severity=8,
            action=PolicyAction.AUDIT,  # Log but don't block
            rule=lambda has_excessive_permissions=False, **_: not has_excessive_permissions,
        ))
    
    def add_policy(self, policy: SecurityPolicy) -> None:
        """Add a policy to the engine."""
        self._policies[policy.policy_id] = policy
    
    def remove_policy(self, policy_id: str) -> None:
        """Remove a policy."""
        if policy_id in self._policies:
            del self._policies[policy_id]
    
    def enable_policy(self, policy_id: str) -> None:
        """Enable a policy."""
        if policy_id in self._policies:
            self._policies[policy_id].enabled = True
    
    def disable_policy(self, policy_id: str) -> None:
        """Disable a policy."""
        if policy_id in self._policies:
            self._policies[policy_id].enabled = False
    
    def evaluate(
        self,
        categories: Optional[list[PolicyCategory]] = None,
        **context,
    ) -> PolicyResult:
        """Evaluate all applicable policies.
        
        Args:
            categories: Filter to specific categories (None = all)
            **context: Context data for policy evaluation
            
        Returns:
            PolicyResult with all violations
        """
        violations = []
        
        for policy in self._policies.values():
            if categories and policy.category not in categories:
                continue
            
            violation = policy.evaluate(**context)
            if violation:
                violations.append(violation)
                logger.warning(
                    "Policy violation: %s - %s",
                    violation.policy_id, violation.message,
                )
        
        return PolicyResult(
            passed=len(violations) == 0,
            violations=violations,
        )
    
    def evaluate_password(self, password: str) -> PolicyResult:
        """Evaluate password against password policies."""
        return self.evaluate(
            categories=[PolicyCategory.AUTHENTICATION],
            password=password,
        )
    
    def evaluate_session(self, session_age_hours: float, idle_minutes: float) -> PolicyResult:
        """Evaluate session against session policies."""
        return self.evaluate(
            categories=[PolicyCategory.SESSION],
            session_age_hours=session_age_hours,
            idle_minutes=idle_minutes,
        )
    
    def get_all_policies(self) -> list[SecurityPolicy]:
        """Get all registered policies."""
        return list(self._policies.values())
    
    def get_policy(self, policy_id: str) -> Optional[SecurityPolicy]:
        """Get a specific policy."""
        return self._policies.get(policy_id)


# Singleton
_policy_engine: Optional[PolicyEngine] = None


def get_policy_engine() -> PolicyEngine:
    """Get or create global policy engine."""
    global _policy_engine
    
    if _policy_engine is None:
        _policy_engine = PolicyEngine()
    
    return _policy_engine
