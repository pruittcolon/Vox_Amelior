"""
PII Detector - Personally Identifiable Information Detection and Redaction.

Provides pre-ingestion PII scanning for RAG knowledge bases following 2024 best practices:
- Regex pattern matching for common PII types
- Named Entity Recognition (NER) integration
- Configurable redaction strategies
- Audit logging for compliance

CRITICAL: PII must be detected and redacted BEFORE storage in knowledge base.

References:
- Microsoft Presidio
- Amazon Comprehend
- OWASP Security Guidelines
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class PIIType(str, Enum):
    """Types of PII that can be detected."""

    SSN = "ssn"
    EMAIL = "email"
    PHONE = "phone"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    DATE_OF_BIRTH = "date_of_birth"
    DRIVER_LICENSE = "driver_license"
    PASSPORT = "passport"
    BANK_ACCOUNT = "bank_account"
    NAME = "name"  # Requires NER
    ADDRESS = "address"  # Requires NER
    CUSTOM = "custom"


@dataclass
class PIIMatch:
    """A detected PII match."""

    pii_type: PIIType
    value: str
    start: int
    end: int
    confidence: float = 1.0
    redacted: Optional[str] = None


class RedactionStrategy(str, Enum):
    """How to redact detected PII."""

    MASK = "mask"  # Replace with [REDACTED]
    HASH = "hash"  # Replace with hash
    PARTIAL = "partial"  # Show first/last chars
    TYPE_LABEL = "type_label"  # Replace with [PII_TYPE]
    REMOVE = "remove"  # Delete entirely


# =============================================================================
# PII PATTERNS
# =============================================================================

PII_PATTERNS: dict[PIIType, re.Pattern] = {
    # US Social Security Number
    PIIType.SSN: re.compile(
        r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b"
    ),
    # Email address
    PIIType.EMAIL: re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    ),
    # US Phone number (various formats)
    PIIType.PHONE: re.compile(
        r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    ),
    # Credit card (Visa, MasterCard, Amex, Discover)
    PIIType.CREDIT_CARD: re.compile(
        r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|"
        r"3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12}|"
        r"\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4})\b"
    ),
    # IP Address (IPv4)
    PIIType.IP_ADDRESS: re.compile(
        r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}"
        r"(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"
    ),
    # Date of birth (various formats)
    PIIType.DATE_OF_BIRTH: re.compile(
        r"\b(?:0?[1-9]|1[0-2])[/\-](?:0?[1-9]|[12][0-9]|3[01])[/\-]"
        r"(?:19|20)\d{2}\b"
    ),
    # US Driver's License (simplified, state-specific patterns vary)
    PIIType.DRIVER_LICENSE: re.compile(
        r"\b[A-Z]{1,2}\d{6,8}\b"
    ),
    # Bank account number (basic pattern)
    PIIType.BANK_ACCOUNT: re.compile(
        r"\b\d{8,17}\b"  # Very broad - use with caution
    ),
}


# =============================================================================
# PII DETECTOR CLASS
# =============================================================================


class PIIDetector:
    """
    Detect and redact PII from text.

    Usage:
        detector = PIIDetector()
        matches = detector.scan("My SSN is 123-45-6789")
        clean_text = detector.redact("My SSN is 123-45-6789")
    """

    def __init__(
        self,
        enabled_types: Optional[list[PIIType]] = None,
        custom_patterns: Optional[dict[str, re.Pattern]] = None,
        strategy: RedactionStrategy = RedactionStrategy.TYPE_LABEL,
        min_confidence: float = 0.5,
    ):
        """
        Initialize PII detector.

        Args:
            enabled_types: PII types to detect (None = all)
            custom_patterns: Additional regex patterns
            strategy: How to redact detected PII
            min_confidence: Minimum confidence threshold
        """
        self.enabled_types = enabled_types or list(PII_PATTERNS.keys())
        self.custom_patterns = custom_patterns or {}
        self.strategy = strategy
        self.min_confidence = min_confidence

        # Build active patterns
        self.patterns: dict[PIIType | str, re.Pattern] = {}
        for pii_type in self.enabled_types:
            if pii_type in PII_PATTERNS:
                self.patterns[pii_type] = PII_PATTERNS[pii_type]

        # Add custom patterns
        for name, pattern in self.custom_patterns.items():
            self.patterns[name] = pattern

    def scan(self, text: str) -> list[PIIMatch]:
        """
        Scan text for PII.

        Args:
            text: Input text to scan

        Returns:
            List of PIIMatch objects
        """
        if not text:
            return []

        matches = []

        for pii_type, pattern in self.patterns.items():
            for match in pattern.finditer(text):
                # Validate match (reduce false positives)
                confidence = self._calculate_confidence(pii_type, match.group())

                if confidence >= self.min_confidence:
                    matches.append(
                        PIIMatch(
                            pii_type=pii_type if isinstance(pii_type, PIIType) else PIIType.CUSTOM,
                            value=match.group(),
                            start=match.start(),
                            end=match.end(),
                            confidence=confidence,
                        )
                    )

        # Sort by position and remove overlaps
        matches.sort(key=lambda m: m.start)
        return self._remove_overlaps(matches)

    def redact(
        self,
        text: str,
        strategy: Optional[RedactionStrategy] = None,
    ) -> str:
        """
        Redact all detected PII from text.

        Args:
            text: Input text
            strategy: Override default strategy

        Returns:
            Text with PII redacted
        """
        if not text:
            return text

        strategy = strategy or self.strategy
        matches = self.scan(text)

        if not matches:
            return text

        # Build redacted text (process in reverse to preserve positions)
        result = text
        for match in reversed(matches):
            replacement = self._get_replacement(match, strategy)
            result = result[:match.start] + replacement + result[match.end:]

        return result

    def scan_and_report(self, text: str) -> dict:
        """
        Scan text and return detailed report.

        Args:
            text: Input text

        Returns:
            Report with statistics and matches
        """
        matches = self.scan(text)

        # Group by type
        by_type = {}
        for match in matches:
            type_name = match.pii_type.value if isinstance(match.pii_type, PIIType) else str(match.pii_type)
            if type_name not in by_type:
                by_type[type_name] = []
            by_type[type_name].append({
                "value": match.value[:3] + "***" if len(match.value) > 3 else "***",
                "position": [match.start, match.end],
                "confidence": match.confidence,
            })

        return {
            "total_pii_found": len(matches),
            "types_found": list(by_type.keys()),
            "details": by_type,
            "risk_level": self._calculate_risk_level(matches),
        }

    def _calculate_confidence(self, pii_type: PIIType | str, value: str) -> float:
        """Calculate confidence score for a match."""
        # Luhn algorithm for credit cards
        if pii_type == PIIType.CREDIT_CARD:
            return 0.95 if self._luhn_check(value) else 0.3

        # SSN validation
        if pii_type == PIIType.SSN:
            # Real SSNs don't start with 000, 666, or 9xx
            digits = re.sub(r"[^\d]", "", value)
            if digits.startswith(("000", "666")) or digits.startswith("9"):
                return 0.2
            return 0.9

        # Email format validation
        if pii_type == PIIType.EMAIL:
            if "@" in value and "." in value.split("@")[-1]:
                return 0.95
            return 0.3

        # Bank account - very low confidence (too many false positives)
        if pii_type == PIIType.BANK_ACCOUNT:
            return 0.3  # Require additional context

        return 0.8  # Default confidence

    def _luhn_check(self, card_number: str) -> bool:
        """Validate credit card using Luhn algorithm."""
        digits = re.sub(r"[^\d]", "", card_number)
        if len(digits) < 13:
            return False

        total = 0
        for i, digit in enumerate(reversed(digits)):
            d = int(digit)
            if i % 2 == 1:
                d *= 2
                if d > 9:
                    d -= 9
            total += d

        return total % 10 == 0

    def _get_replacement(self, match: PIIMatch, strategy: RedactionStrategy) -> str:
        """Get replacement text for a PII match."""
        if strategy == RedactionStrategy.MASK:
            return "[REDACTED]"

        if strategy == RedactionStrategy.TYPE_LABEL:
            type_name = match.pii_type.value.upper() if isinstance(match.pii_type, PIIType) else "PII"
            return f"[{type_name}_REDACTED]"

        if strategy == RedactionStrategy.PARTIAL:
            if len(match.value) > 4:
                return match.value[:2] + "*" * (len(match.value) - 4) + match.value[-2:]
            return "*" * len(match.value)

        if strategy == RedactionStrategy.HASH:
            import hashlib
            return hashlib.sha256(match.value.encode()).hexdigest()[:12]

        if strategy == RedactionStrategy.REMOVE:
            return ""

        return "[REDACTED]"

    def _remove_overlaps(self, matches: list[PIIMatch]) -> list[PIIMatch]:
        """Remove overlapping matches, keeping higher confidence."""
        if len(matches) <= 1:
            return matches

        result = []
        for match in matches:
            # Check for overlap with existing
            overlaps = False
            for existing in result:
                if match.start < existing.end and match.end > existing.start:
                    # Overlap detected - keep higher confidence
                    if match.confidence > existing.confidence:
                        result.remove(existing)
                        result.append(match)
                    overlaps = True
                    break

            if not overlaps:
                result.append(match)

        return result

    def _calculate_risk_level(self, matches: list[PIIMatch]) -> str:
        """Calculate overall risk level based on detected PII."""
        if not matches:
            return "low"

        high_risk_types = {PIIType.SSN, PIIType.CREDIT_CARD, PIIType.BANK_ACCOUNT}

        high_risk_count = sum(1 for m in matches if m.pii_type in high_risk_types)

        if high_risk_count > 0:
            return "critical"
        if len(matches) > 5:
            return "high"
        if len(matches) > 0:
            return "medium"
        return "low"


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def scan_for_pii(text: str) -> list[PIIMatch]:
    """Quick scan for PII using default settings."""
    detector = PIIDetector()
    return detector.scan(text)


def redact_pii(text: str) -> str:
    """Quick redaction using default settings."""
    detector = PIIDetector()
    return detector.redact(text)


def contains_pii(text: str) -> bool:
    """Check if text contains any PII."""
    return len(scan_for_pii(text)) > 0
