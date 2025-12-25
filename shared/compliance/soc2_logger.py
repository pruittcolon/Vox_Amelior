"""
SOC 2 Type II Compliant Audit Logger.

Provides structured audit logging that meets SOC 2 requirements:
- Common Event Format (CEF) structured logs
- Immutable audit trail with HMAC signing
- Required fields for SOC 2 auditors
- Automatic evidence collection

CEF Format: CEF:Version|Device Vendor|Device Product|Device Version|Signature ID|Name|Severity|Extension

Usage:
    logger = SOC2AuditLogger()
    logger.log_auth_event(
        user_id="user123",
        action="LOGIN_SUCCESS",
        ip_address="192.168.1.1",
    )
"""

import hashlib
import hmac
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class SOC2Category(Enum):
    """SOC 2 Trust Services Categories."""
    
    SECURITY = "CC"          # Common Criteria (Security)
    AVAILABILITY = "A"       # Availability
    PROCESSING_INTEGRITY = "PI"  # Processing Integrity
    CONFIDENTIALITY = "C"    # Confidentiality
    PRIVACY = "P"            # Privacy


class EventSeverity(Enum):
    """CEF Severity Levels (0-10)."""
    
    UNKNOWN = 0
    LOW = 3
    MEDIUM = 5
    HIGH = 7
    CRITICAL = 10


class AuditAction(Enum):
    """Standardized audit actions."""
    
    # Authentication
    LOGIN_SUCCESS = "AUTH_LOGIN_SUCCESS"
    LOGIN_FAILURE = "AUTH_LOGIN_FAILURE"
    LOGOUT = "AUTH_LOGOUT"
    MFA_CHALLENGE = "AUTH_MFA_CHALLENGE"
    MFA_SUCCESS = "AUTH_MFA_SUCCESS"
    MFA_FAILURE = "AUTH_MFA_FAILURE"
    PASSWORD_CHANGE = "AUTH_PASSWORD_CHANGE"
    PASSWORD_RESET = "AUTH_PASSWORD_RESET"
    ACCOUNT_LOCKED = "AUTH_ACCOUNT_LOCKED"
    SESSION_EXPIRED = "AUTH_SESSION_EXPIRED"
    
    # Authorization
    ACCESS_GRANTED = "AUTHZ_ACCESS_GRANTED"
    ACCESS_DENIED = "AUTHZ_ACCESS_DENIED"
    PRIVILEGE_ESCALATION = "AUTHZ_PRIVILEGE_ESCALATION"
    ROLE_CHANGE = "AUTHZ_ROLE_CHANGE"
    
    # Data Access
    DATA_READ = "DATA_READ"
    DATA_CREATE = "DATA_CREATE"
    DATA_UPDATE = "DATA_UPDATE"
    DATA_DELETE = "DATA_DELETE"
    DATA_EXPORT = "DATA_EXPORT"
    DATA_IMPORT = "DATA_IMPORT"
    
    # Security Events
    SECURITY_ALERT = "SEC_ALERT"
    WAF_BLOCK = "SEC_WAF_BLOCK"
    RATE_LIMIT = "SEC_RATE_LIMIT"
    ANOMALY_DETECTED = "SEC_ANOMALY"
    THREAT_BLOCKED = "SEC_THREAT_BLOCKED"
    
    # System Events
    SERVICE_START = "SYS_SERVICE_START"
    SERVICE_STOP = "SYS_SERVICE_STOP"
    CONFIG_CHANGE = "SYS_CONFIG_CHANGE"
    KEY_ROTATION = "SYS_KEY_ROTATION"
    BACKUP_CREATED = "SYS_BACKUP"
    
    # Compliance Events
    POLICY_VIOLATION = "COMPL_POLICY_VIOLATION"
    EVIDENCE_COLLECTED = "COMPL_EVIDENCE"
    AUDIT_STARTED = "COMPL_AUDIT_START"
    AUDIT_COMPLETED = "COMPL_AUDIT_COMPLETE"


@dataclass
class AuditEvent:
    """SOC 2 compliant audit event.
    
    Contains all fields required for SOC 2 Type II audit evidence.
    """
    
    # Required fields
    event_id: str                      # Unique event identifier
    timestamp: str                     # ISO 8601 timestamp
    action: str                        # Action performed
    outcome: str                       # SUCCESS or FAILURE
    
    # Actor information
    actor_id: Optional[str] = None     # User/service ID
    actor_type: str = "user"           # user, service, system
    actor_ip: Optional[str] = None     # Source IP address
    actor_user_agent: Optional[str] = None
    
    # Resource information
    resource_type: Optional[str] = None   # Type of resource accessed
    resource_id: Optional[str] = None     # Resource identifier
    resource_name: Optional[str] = None   # Human-readable name
    
    # Context
    service_name: str = "nemo"         # Originating service
    environment: str = "production"    # Environment name
    tenant_id: Optional[str] = None    # Multi-tenant identifier
    session_id: Optional[str] = None   # Session identifier
    request_id: Optional[str] = None   # Request correlation ID
    
    # SOC 2 specific
    soc2_category: str = "CC"          # Trust Services Category
    severity: int = 3                  # 0-10 severity
    justification: Optional[str] = None  # Why action was taken
    
    # Details
    details: dict = field(default_factory=dict)
    
    # Integrity
    signature: Optional[str] = None    # HMAC signature
    
    def to_cef(self) -> str:
        """Convert to CEF format string."""
        # CEF:Version|Vendor|Product|Version|SignatureID|Name|Severity|Extension
        extensions = [
            f"act={self.action}",
            f"outcome={self.outcome}",
            f"rt={self.timestamp}",
            f"duid={self.event_id}",
        ]
        
        if self.actor_id:
            extensions.append(f"suid={self.actor_id}")
        if self.actor_ip:
            extensions.append(f"src={self.actor_ip}")
        if self.resource_type:
            extensions.append(f"cs1={self.resource_type}")
            extensions.append("cs1Label=resourceType")
        if self.resource_id:
            extensions.append(f"cs2={self.resource_id}")
            extensions.append("cs2Label=resourceId")
        if self.tenant_id:
            extensions.append(f"cs3={self.tenant_id}")
            extensions.append("cs3Label=tenantId")
        if self.session_id:
            extensions.append(f"cs4={self.session_id}")
            extensions.append("cs4Label=sessionId")
        if self.signature:
            extensions.append(f"cs5={self.signature}")
            extensions.append("cs5Label=hmacSignature")
        
        return (
            f"CEF:0|Nemo|NemoServer|1.0|{self.action}|"
            f"{self.action}|{self.severity}|{' '.join(extensions)}"
        )
    
    def to_json(self) -> str:
        """Convert to JSON format."""
        return json.dumps(asdict(self), default=str)


class SOC2AuditLogger:
    """SOC 2 Type II compliant audit logger.
    
    Features:
    - CEF format output for SIEM integration
    - JSON format for analysis
    - HMAC signing for integrity
    - Automatic field population
    - Evidence collection ready
    """
    
    def __init__(
        self,
        service_name: str = "nemo",
        environment: Optional[str] = None,
        signing_key: Optional[bytes] = None,
        log_path: Optional[str] = None,
    ):
        """Initialize the audit logger.
        
        Args:
            service_name: Name of the originating service
            environment: Deployment environment
            signing_key: HMAC signing key for integrity
            log_path: Path for audit log files
        """
        self.service_name = service_name
        self.environment = environment or os.getenv("ENVIRONMENT", "production")
        
        # Load signing key
        self._signing_key = signing_key
        if not signing_key:
            key_path = Path("/run/secrets/audit_hmac_key")
            if key_path.exists():
                self._signing_key = key_path.read_bytes().strip()
            else:
                # Generate ephemeral key for development
                self._signing_key = os.urandom(32)
        
        # Setup logging
        self._log_path = log_path or os.getenv(
            "AUDIT_LOG_PATH",
            "/var/log/nemo/audit.log"
        )
        self._setup_logger()
        
        # Event counter for IDs
        self._event_counter = 0
        
        logger.info(
            "SOC2AuditLogger initialized: service=%s, env=%s",
            service_name, self.environment,
        )
    
    def _setup_logger(self) -> None:
        """Setup dedicated audit file logger."""
        self._audit_logger = logging.getLogger(f"audit.{self.service_name}")
        self._audit_logger.setLevel(logging.INFO)
        
        # Avoid duplicate handlers
        if not self._audit_logger.handlers:
            # Console handler (CEF format)
            console = logging.StreamHandler()
            console.setFormatter(logging.Formatter("%(message)s"))
            self._audit_logger.addHandler(console)
            
            # File handler if path exists
            log_dir = Path(self._log_path).parent
            if log_dir.exists():
                file_handler = logging.FileHandler(self._log_path)
                file_handler.setFormatter(logging.Formatter("%(message)s"))
                self._audit_logger.addHandler(file_handler)
    
    def _generate_event_id(self) -> str:
        """Generate unique event ID."""
        self._event_counter += 1
        timestamp = int(time.time() * 1000)
        return f"{self.service_name}-{timestamp}-{self._event_counter:06d}"
    
    def _sign_event(self, event: AuditEvent) -> str:
        """Generate HMAC signature for event."""
        # Create canonical string
        canonical = f"{event.event_id}|{event.timestamp}|{event.action}|{event.outcome}"
        if event.actor_id:
            canonical += f"|{event.actor_id}"
        if event.resource_id:
            canonical += f"|{event.resource_id}"
        
        signature = hmac.new(
            self._signing_key,
            canonical.encode(),
            hashlib.sha256,
        ).hexdigest()[:32]
        
        return signature
    
    def log(
        self,
        action: AuditAction,
        outcome: str = "SUCCESS",
        actor_id: Optional[str] = None,
        actor_ip: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        resource_name: Optional[str] = None,
        tenant_id: Optional[str] = None,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
        severity: EventSeverity = EventSeverity.LOW,
        soc2_category: SOC2Category = SOC2Category.SECURITY,
        justification: Optional[str] = None,
        details: Optional[dict] = None,
    ) -> AuditEvent:
        """Log an audit event.
        
        Args:
            action: The action being audited
            outcome: SUCCESS or FAILURE
            actor_id: User or service ID
            actor_ip: Source IP address
            resource_type: Type of resource
            resource_id: Resource identifier
            resource_name: Human-readable resource name
            tenant_id: Multi-tenant ID
            session_id: Session ID
            request_id: Request correlation ID
            severity: Event severity
            soc2_category: SOC 2 Trust Services Category
            justification: Reason for the action
            details: Additional details
            
        Returns:
            The logged AuditEvent
        """
        event = AuditEvent(
            event_id=self._generate_event_id(),
            timestamp=datetime.now(timezone.utc).isoformat(),
            action=action.value if isinstance(action, AuditAction) else action,
            outcome=outcome,
            actor_id=actor_id,
            actor_ip=actor_ip,
            resource_type=resource_type,
            resource_id=resource_id,
            resource_name=resource_name,
            service_name=self.service_name,
            environment=self.environment,
            tenant_id=tenant_id,
            session_id=session_id,
            request_id=request_id,
            soc2_category=soc2_category.value if isinstance(soc2_category, SOC2Category) else soc2_category,
            severity=severity.value if isinstance(severity, EventSeverity) else severity,
            justification=justification,
            details=details or {},
        )
        
        # Sign the event
        event.signature = self._sign_event(event)
        
        # Log in CEF format
        self._audit_logger.info(event.to_cef())
        
        return event
    
    # Convenience methods for common events
    
    def log_auth_event(
        self,
        action: AuditAction,
        user_id: str,
        ip_address: str,
        outcome: str = "SUCCESS",
        **kwargs,
    ) -> AuditEvent:
        """Log authentication event."""
        return self.log(
            action=action,
            outcome=outcome,
            actor_id=user_id,
            actor_ip=ip_address,
            soc2_category=SOC2Category.SECURITY,
            severity=EventSeverity.MEDIUM if outcome == "FAILURE" else EventSeverity.LOW,
            **kwargs,
        )
    
    def log_data_access(
        self,
        action: AuditAction,
        user_id: str,
        resource_type: str,
        resource_id: str,
        **kwargs,
    ) -> AuditEvent:
        """Log data access event."""
        return self.log(
            action=action,
            actor_id=user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            soc2_category=SOC2Category.CONFIDENTIALITY,
            **kwargs,
        )
    
    def log_security_event(
        self,
        action: AuditAction,
        severity: EventSeverity,
        details: dict,
        **kwargs,
    ) -> AuditEvent:
        """Log security event."""
        return self.log(
            action=action,
            severity=severity,
            soc2_category=SOC2Category.SECURITY,
            details=details,
            **kwargs,
        )
    
    def log_config_change(
        self,
        actor_id: str,
        resource_name: str,
        old_value: Any,
        new_value: Any,
        **kwargs,
    ) -> AuditEvent:
        """Log configuration change."""
        return self.log(
            action=AuditAction.CONFIG_CHANGE,
            actor_id=actor_id,
            resource_type="configuration",
            resource_name=resource_name,
            severity=EventSeverity.HIGH,
            details={"old_value": str(old_value), "new_value": str(new_value)},
            soc2_category=SOC2Category.SECURITY,
            **kwargs,
        )


# Singleton
_audit_logger: Optional[SOC2AuditLogger] = None


def get_audit_logger(service_name: str = "nemo") -> SOC2AuditLogger:
    """Get or create global audit logger."""
    global _audit_logger
    
    if _audit_logger is None:
        _audit_logger = SOC2AuditLogger(service_name=service_name)
    
    return _audit_logger
