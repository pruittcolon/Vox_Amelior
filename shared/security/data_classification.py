"""
Data Classification Policy Module

Implements enterprise data classification for SOC2 and GDPR compliance.
Assigns classification levels to data types and enforces appropriate controls.

Classification Levels:
- RESTRICTED: PII, credentials, encryption keys - requires encryption + audit
- CONFIDENTIAL: Financial data, transcripts - requires encryption + access control
- INTERNAL: Analytics, logs - requires access control
- PUBLIC: API docs, status - no special controls

Usage:
    from shared.security.data_classification import DataClassifier, ClassificationLevel
    
    classifier = DataClassifier()
    level = classifier.classify_field("email")  # Returns RESTRICTED
    controls = classifier.get_required_controls(level)
"""

import enum
import re
from dataclasses import dataclass
from typing import Any


class ClassificationLevel(enum.Enum):
    """Data classification levels per SOC2/GDPR requirements."""
    
    RESTRICTED = "restricted"      # Highest: PII, credentials, keys
    CONFIDENTIAL = "confidential"  # Medium-high: Financial, transcripts
    INTERNAL = "internal"          # Medium: Analytics, operational data
    PUBLIC = "public"              # Lowest: Documentation, status


@dataclass(frozen=True)
class SecurityControls:
    """Required security controls for a classification level."""
    
    encryption_at_rest: bool
    encryption_in_transit: bool
    audit_logging: bool
    access_control: bool
    pii_masking: bool
    retention_days: int | None


# Classification control requirements
CLASSIFICATION_CONTROLS: dict[ClassificationLevel, SecurityControls] = {
    ClassificationLevel.RESTRICTED: SecurityControls(
        encryption_at_rest=True,
        encryption_in_transit=True,
        audit_logging=True,
        access_control=True,
        pii_masking=True,
        retention_days=365,  # 1 year max for PII
    ),
    ClassificationLevel.CONFIDENTIAL: SecurityControls(
        encryption_at_rest=True,
        encryption_in_transit=True,
        audit_logging=True,
        access_control=True,
        pii_masking=False,
        retention_days=730,  # 2 years
    ),
    ClassificationLevel.INTERNAL: SecurityControls(
        encryption_at_rest=False,
        encryption_in_transit=True,
        audit_logging=False,
        access_control=True,
        pii_masking=False,
        retention_days=None,  # No limit
    ),
    ClassificationLevel.PUBLIC: SecurityControls(
        encryption_at_rest=False,
        encryption_in_transit=False,
        audit_logging=False,
        access_control=False,
        pii_masking=False,
        retention_days=None,
    ),
}

# Field patterns and their classifications
FIELD_CLASSIFICATIONS: dict[str, ClassificationLevel] = {
    # RESTRICTED - PII and secrets
    "password": ClassificationLevel.RESTRICTED,
    "password_hash": ClassificationLevel.RESTRICTED,
    "secret": ClassificationLevel.RESTRICTED,
    "api_key": ClassificationLevel.RESTRICTED,
    "jwt": ClassificationLevel.RESTRICTED,
    "token": ClassificationLevel.RESTRICTED,
    "ssn": ClassificationLevel.RESTRICTED,
    "social_security": ClassificationLevel.RESTRICTED,
    "credit_card": ClassificationLevel.RESTRICTED,
    "card_number": ClassificationLevel.RESTRICTED,
    "cvv": ClassificationLevel.RESTRICTED,
    "email": ClassificationLevel.RESTRICTED,
    "phone": ClassificationLevel.RESTRICTED,
    "ip_address": ClassificationLevel.RESTRICTED,
    "date_of_birth": ClassificationLevel.RESTRICTED,
    "dob": ClassificationLevel.RESTRICTED,
    
    # CONFIDENTIAL - Business data
    "account_number": ClassificationLevel.CONFIDENTIAL,
    "routing_number": ClassificationLevel.CONFIDENTIAL,
    "balance": ClassificationLevel.CONFIDENTIAL,
    "transcript": ClassificationLevel.CONFIDENTIAL,
    "financial": ClassificationLevel.CONFIDENTIAL,
    "salary": ClassificationLevel.CONFIDENTIAL,
    "income": ClassificationLevel.CONFIDENTIAL,
    
    # INTERNAL - Operational data
    "analytics": ClassificationLevel.INTERNAL,
    "metrics": ClassificationLevel.INTERNAL,
    "log": ClassificationLevel.INTERNAL,
    "request_id": ClassificationLevel.INTERNAL,
    "user_agent": ClassificationLevel.INTERNAL,
    
    # PUBLIC - Documentation and status
    "version": ClassificationLevel.PUBLIC,
    "status": ClassificationLevel.PUBLIC,
    "documentation": ClassificationLevel.PUBLIC,
    "readme": ClassificationLevel.PUBLIC,
}

# Regex patterns for auto-detection
PII_PATTERNS: dict[str, re.Pattern] = {
    "email": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", re.IGNORECASE),
    "ssn": re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b"),
    "phone": re.compile(r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
    "ip_address": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
}


class DataClassifier:
    """
    Classifies data based on field names and content patterns.
    
    Assigns SOC2/GDPR-compliant classification levels and determines
    required security controls for each data type.
    """
    
    def __init__(self):
        self.field_classifications = FIELD_CLASSIFICATIONS.copy()
        self.patterns = PII_PATTERNS.copy()
    
    def classify_field(self, field_name: str) -> ClassificationLevel:
        """
        Classify a field by its name.
        
        Args:
            field_name: Name of the field (e.g., "email", "username")
            
        Returns:
            Classification level for the field.
        """
        normalized = field_name.lower().strip()
        
        # Direct match
        if normalized in self.field_classifications:
            return self.field_classifications[normalized]
        
        # Partial match (e.g., "user_email" contains "email")
        for pattern, level in self.field_classifications.items():
            if pattern in normalized:
                return level
        
        # Default to INTERNAL for unknown fields
        return ClassificationLevel.INTERNAL
    
    def classify_value(self, value: Any) -> ClassificationLevel:
        """
        Classify a value by its content (pattern matching).
        
        Args:
            value: The value to classify
            
        Returns:
            Highest classification level detected in the value.
        """
        if value is None:
            return ClassificationLevel.PUBLIC
        
        # Priority: lower number = higher/stricter classification
        level_priority = {
            ClassificationLevel.RESTRICTED: 1,
            ClassificationLevel.CONFIDENTIAL: 2,
            ClassificationLevel.INTERNAL: 3,
            ClassificationLevel.PUBLIC: 4,
        }
        
        str_value = str(value)
        highest_level = ClassificationLevel.PUBLIC
        highest_priority = level_priority[highest_level]
        
        for pattern_name, pattern in self.patterns.items():
            if pattern.search(str_value):
                field_level = self.classify_field(pattern_name)
                field_priority = level_priority.get(field_level, 4)
                if field_priority < highest_priority:
                    highest_level = field_level
                    highest_priority = field_priority
        
        return highest_level
    
    def get_required_controls(self, level: ClassificationLevel) -> SecurityControls:
        """
        Get required security controls for a classification level.
        
        Args:
            level: Classification level
            
        Returns:
            SecurityControls dataclass with required measures.
        """
        return CLASSIFICATION_CONTROLS[level]
    
    def validate_controls(self, level: ClassificationLevel, 
                          has_encryption: bool = False,
                          has_audit: bool = False,
                          has_access_control: bool = False) -> tuple[bool, list[str]]:
        """
        Validate that required controls are in place for a classification level.
        
        Args:
            level: Classification level to check
            has_encryption: Whether encryption is enabled
            has_audit: Whether audit logging is enabled
            has_access_control: Whether access control is enforced
            
        Returns:
            Tuple of (is_compliant, list of missing controls)
        """
        controls = self.get_required_controls(level)
        missing = []
        
        if controls.encryption_at_rest and not has_encryption:
            missing.append("encryption_at_rest")
        if controls.audit_logging and not has_audit:
            missing.append("audit_logging")
        if controls.access_control and not has_access_control:
            missing.append("access_control")
        
        return len(missing) == 0, missing


# Singleton instance
_classifier: DataClassifier | None = None


def get_data_classifier() -> DataClassifier:
    """Get the global data classifier instance."""
    global _classifier
    if _classifier is None:
        _classifier = DataClassifier()
    return _classifier


def classify_field(field_name: str) -> ClassificationLevel:
    """Convenience function to classify a field by name."""
    return get_data_classifier().classify_field(field_name)


def classify_value(value: Any) -> ClassificationLevel:
    """Convenience function to classify a value by content."""
    return get_data_classifier().classify_value(value)
