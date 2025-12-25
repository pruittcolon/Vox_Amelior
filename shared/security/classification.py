"""
Data Classification Module - Security Policy Enforcement.

This module provides data classification levels following enterprise
security standards. All sensitive data should be annotated with
appropriate classification levels.

Phase 17 of ultimateseniordevplan.md.
"""

from enum import Enum

from pydantic import BaseModel


class DataClassification(str, Enum):
    """Data sensitivity levels per security policy.

    Classification hierarchy (highest to lowest):
    - RESTRICTED: PII, financial data, authentication secrets
    - CONFIDENTIAL: Internal business data, limited access
    - INTERNAL: Internal use only, not for external sharing
    - PUBLIC: Can be shared externally

    Attributes:
        RESTRICTED: Most sensitive - requires encryption and audit
        CONFIDENTIAL: Sensitive business data
        INTERNAL: Internal documents and data
        PUBLIC: Publicly available information
    """

    RESTRICTED = "restricted"
    CONFIDENTIAL = "confidential"
    INTERNAL = "internal"
    PUBLIC = "public"


class ClassificationRules(BaseModel):
    """Rules for handling data at a specific classification level.

    Attributes:
        encryption_required: Whether data must be encrypted at rest.
        audit_access: Whether all access must be logged.
        retention_max_days: Maximum retention period in days (None = unlimited).
        export_allowed: Whether data can be exported from system.
        logging_redaction: Whether to redact from logs.
        requires_mfa: Whether MFA is required for access.
    """

    encryption_required: bool = False
    audit_access: bool = False
    retention_max_days: int | None = None
    export_allowed: bool = True
    logging_redaction: bool = False
    requires_mfa: bool = False


# Classification rule definitions
CLASSIFICATION_RULES: dict[DataClassification, ClassificationRules] = {
    DataClassification.RESTRICTED: ClassificationRules(
        encryption_required=True,
        audit_access=True,
        retention_max_days=365,
        export_allowed=False,
        logging_redaction=True,
        requires_mfa=True,
    ),
    DataClassification.CONFIDENTIAL: ClassificationRules(
        encryption_required=True,
        audit_access=True,
        retention_max_days=730,
        export_allowed=True,
        logging_redaction=True,
        requires_mfa=False,
    ),
    DataClassification.INTERNAL: ClassificationRules(
        encryption_required=False,
        audit_access=False,
        retention_max_days=1095,
        export_allowed=True,
        logging_redaction=False,
        requires_mfa=False,
    ),
    DataClassification.PUBLIC: ClassificationRules(
        encryption_required=False,
        audit_access=False,
        retention_max_days=None,
        export_allowed=True,
        logging_redaction=False,
        requires_mfa=False,
    ),
}


def get_classification_rules(
    classification: DataClassification,
) -> ClassificationRules:
    """Get handling rules for a data classification level.

    Args:
        classification: Data classification level.

    Returns:
        ClassificationRules: Rules for handling data at this level.

    Example:
        >>> rules = get_classification_rules(DataClassification.RESTRICTED)
        >>> print(rules.encryption_required)  # True
    """
    return CLASSIFICATION_RULES[classification]


def should_redact_in_logs(classification: DataClassification) -> bool:
    """Check if data should be redacted from logs.

    Args:
        classification: Data classification level.

    Returns:
        bool: True if data should be redacted from logs.
    """
    return CLASSIFICATION_RULES[classification].logging_redaction


def requires_encryption(classification: DataClassification) -> bool:
    """Check if data requires encryption at rest.

    Args:
        classification: Data classification level.

    Returns:
        bool: True if data must be encrypted.
    """
    return CLASSIFICATION_RULES[classification].encryption_required


# Field classifications for common data types
FIELD_CLASSIFICATIONS: dict[str, DataClassification] = {
    # Restricted fields (PII, auth)
    "password": DataClassification.RESTRICTED,
    "password_hash": DataClassification.RESTRICTED,
    "ssn": DataClassification.RESTRICTED,
    "social_security_number": DataClassification.RESTRICTED,
    "credit_card": DataClassification.RESTRICTED,
    "credit_card_number": DataClassification.RESTRICTED,
    "cvv": DataClassification.RESTRICTED,
    "bank_account": DataClassification.RESTRICTED,
    "routing_number": DataClassification.RESTRICTED,
    "session_token": DataClassification.RESTRICTED,
    "api_key": DataClassification.RESTRICTED,
    "jwt_secret": DataClassification.RESTRICTED,
    "encryption_key": DataClassification.RESTRICTED,
    # Confidential fields
    "email": DataClassification.CONFIDENTIAL,
    "phone": DataClassification.CONFIDENTIAL,
    "phone_number": DataClassification.CONFIDENTIAL,
    "address": DataClassification.CONFIDENTIAL,
    "date_of_birth": DataClassification.CONFIDENTIAL,
    "salary": DataClassification.CONFIDENTIAL,
    "income": DataClassification.CONFIDENTIAL,
    # Internal fields
    "user_id": DataClassification.INTERNAL,
    "username": DataClassification.INTERNAL,
    "role": DataClassification.INTERNAL,
    "created_at": DataClassification.INTERNAL,
    "updated_at": DataClassification.INTERNAL,
}


def classify_field(field_name: str) -> DataClassification:
    """Get classification for a field by name.

    Uses fuzzy matching to classify fields based on naming patterns.

    Args:
        field_name: Name of the field to classify.

    Returns:
        DataClassification: Detected classification (defaults to INTERNAL).

    Example:
        >>> classify_field("user_password")  # RESTRICTED
        >>> classify_field("status")  # PUBLIC
    """
    field_lower = field_name.lower()

    # Check exact matches first
    if field_lower in FIELD_CLASSIFICATIONS:
        return FIELD_CLASSIFICATIONS[field_lower]

    # Check partial matches
    for pattern, classification in FIELD_CLASSIFICATIONS.items():
        if pattern in field_lower:
            return classification

    # Default to INTERNAL for unknown fields
    return DataClassification.INTERNAL
