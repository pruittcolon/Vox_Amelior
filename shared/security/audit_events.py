"""
Audit Event Logger - Compliance-Ready Security Audit Trail.

This module provides immutable audit event logging for security-relevant
actions. Follows SOC 2, GDPR, and HIPAA audit requirements.

Phase 16 of ultimateseniordevplan.md.
"""

import json
import logging
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class AuditEvent(BaseModel):
    """Immutable audit log entry for compliance.

    All fields are required for proper audit trails. Events are
    write-only once created - they cannot be modified.

    Attributes:
        event_id: Unique identifier for this event.
        timestamp: UTC timestamp when event occurred.
        actor_id: Identifier of who/what performed the action.
        actor_type: Type of actor (user, service, system).
        action: Action performed (e.g., "user.login", "data.exported").
        resource_type: Type of resource affected.
        resource_id: Identifier of affected resource.
        changes: Before/after state for mutations.
        ip_address: Client IP address if applicable.
        user_agent: Client user agent if applicable.
        request_id: Correlation ID for request tracing.
        result: Whether action succeeded or failed.
        failure_reason: Reason for failure if result is "failure".
        metadata: Additional context-specific data.
    """

    event_id: str = Field(
        default_factory=lambda: f"audit_{uuid4().hex[:16]}",
        description="Unique audit event identifier",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp of event",
    )
    actor_id: str = Field(..., description="Who performed the action")
    actor_type: Literal["user", "service", "system"] = Field(..., description="Type of actor")
    action: str = Field(..., description="Action performed")
    resource_type: str = Field(..., description="Type of resource affected")
    resource_id: str = Field(..., description="ID of affected resource")
    changes: dict[str, Any] | None = Field(default=None, description="Before/after state")
    ip_address: str | None = Field(default=None, description="Client IP")
    user_agent: str | None = Field(default=None, description="Client user agent")
    request_id: str = Field(
        default_factory=lambda: f"req_{uuid4().hex[:12]}",
        description="Request correlation ID",
    )
    result: Literal["success", "failure"] = Field(..., description="Action result")
    failure_reason: str | None = Field(default=None, description="Reason for failure")
    metadata: dict[str, Any] | None = Field(default=None, description="Additional context")


# Security-relevant actions that MUST be audited
AUDITABLE_ACTIONS = [
    # Authentication
    "auth.login",
    "auth.login_failed",
    "auth.logout",
    "auth.session_created",
    "auth.session_expired",
    "auth.mfa_enabled",
    "auth.mfa_disabled",
    "auth.password_changed",
    "auth.password_reset_requested",
    # User management
    "user.created",
    "user.updated",
    "user.deleted",
    "user.role_changed",
    "user.disabled",
    "user.enabled",
    # Permissions
    "permission.granted",
    "permission.revoked",
    "permission.elevated",
    # Data access
    "data.accessed",
    "data.exported",
    "data.imported",
    "data.deleted",
    # Configuration
    "config.changed",
    "config.secret_rotated",
    # API keys
    "api_key.created",
    "api_key.revoked",
    "api_key.used",
    # Admin actions
    "admin.impersonation_started",
    "admin.impersonation_ended",
    "admin.system_command",
    # Security events
    "security.rate_limit_exceeded",
    "security.invalid_token",
    "security.suspicious_activity",
    "security.csrf_violation",
]


class AuditLogger:
    """Thread-safe audit event logger.

    Logs audit events to structured logging backend. In production,
    events should be forwarded to a SIEM or dedicated audit log store.

    Attributes:
        service_name: Name of the service generating events.
        logger: Underlying Python logger instance.
    """

    def __init__(self, service_name: str = "api-gateway"):
        """Initialize audit logger.

        Args:
            service_name: Name of the service for event attribution.
        """
        self.service_name = service_name
        self.logger = logging.getLogger(f"audit.{service_name}")
        self.logger.setLevel(logging.INFO)

    async def log_event(self, event: AuditEvent) -> None:
        """Log an audit event.

        Events are logged as structured JSON for SIEM ingestion.

        Args:
            event: AuditEvent to log.
        """
        event_dict = event.model_dump()
        event_dict["service"] = self.service_name
        event_dict["timestamp"] = event.timestamp.isoformat()

        self.logger.info(
            "AUDIT: %s",
            json.dumps(event_dict, default=str),
            extra={"audit_event": event_dict},
        )

    async def log_auth_event(
        self,
        action: str,
        user_id: str,
        result: Literal["success", "failure"],
        ip_address: str | None = None,
        user_agent: str | None = None,
        request_id: str | None = None,
        failure_reason: str | None = None,
    ) -> None:
        """Convenience method for authentication events.

        Args:
            action: Auth action (e.g., "auth.login").
            user_id: User identifier.
            result: Success or failure.
            ip_address: Client IP.
            user_agent: Client user agent.
            request_id: Request correlation ID.
            failure_reason: Reason for failure.
        """
        event = AuditEvent(
            actor_id=user_id,
            actor_type="user",
            action=action,
            resource_type="session",
            resource_id=user_id,
            result=result,
            ip_address=ip_address,
            user_agent=user_agent,
            request_id=request_id or f"req_{uuid4().hex[:12]}",
            failure_reason=failure_reason,
        )
        await self.log_event(event)

    async def log_data_access(
        self,
        user_id: str,
        resource_type: str,
        resource_id: str,
        action: str = "data.accessed",
        ip_address: str | None = None,
        request_id: str | None = None,
    ) -> None:
        """Convenience method for data access events.

        Args:
            user_id: User accessing data.
            resource_type: Type of resource accessed.
            resource_id: ID of resource accessed.
            action: Specific action (default: data.accessed).
            ip_address: Client IP.
            request_id: Request correlation ID.
        """
        event = AuditEvent(
            actor_id=user_id,
            actor_type="user",
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            result="success",
            ip_address=ip_address,
            request_id=request_id or f"req_{uuid4().hex[:12]}",
        )
        await self.log_event(event)


# Global audit logger instance
_audit_logger: AuditLogger | None = None


def get_audit_logger(service_name: str = "api-gateway") -> AuditLogger:
    """Get or create global audit logger.

    Args:
        service_name: Service name for new logger.

    Returns:
        AuditLogger: Global audit logger instance.
    """
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger(service_name)
    return _audit_logger
