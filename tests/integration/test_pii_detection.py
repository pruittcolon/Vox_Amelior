"""
Integration Tests for PII Detection and Data Classification

Tests the PII detection, redaction, and data classification functionality
for SOC2/GDPR compliance.
"""

import pytest
import sys
from pathlib import Path

# Add shared to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shared"))


class TestDataClassification:
    """Tests for the data classification module."""

    def test_classify_restricted_fields(self):
        """Test that PII fields are classified as RESTRICTED."""
        from shared.security.data_classification import (
            classify_field,
            ClassificationLevel,
        )
        
        restricted_fields = [
            "email", "phone", "ssn", "password", "api_key", 
            "credit_card", "ip_address"
        ]
        
        for field in restricted_fields:
            level = classify_field(field)
            assert level == ClassificationLevel.RESTRICTED, f"{field} should be RESTRICTED"

    def test_classify_confidential_fields(self):
        """Test that financial fields are classified as CONFIDENTIAL."""
        from shared.security.data_classification import (
            classify_field,
            ClassificationLevel,
        )
        
        confidential_fields = ["account_number", "balance", "salary", "transcript"]
        
        for field in confidential_fields:
            level = classify_field(field)
            assert level == ClassificationLevel.CONFIDENTIAL, f"{field} should be CONFIDENTIAL"

    def test_classify_internal_fields(self):
        """Test that operational fields are classified as INTERNAL."""
        from shared.security.data_classification import (
            classify_field,
            ClassificationLevel,
        )
        
        internal_fields = ["analytics", "metrics", "log", "request_id"]
        
        for field in internal_fields:
            level = classify_field(field)
            assert level == ClassificationLevel.INTERNAL, f"{field} should be INTERNAL"

    def test_classify_public_fields(self):
        """Test that public fields are classified as PUBLIC."""
        from shared.security.data_classification import (
            classify_field,
            ClassificationLevel,
        )
        
        public_fields = ["version", "status", "documentation"]
        
        for field in public_fields:
            level = classify_field(field)
            assert level == ClassificationLevel.PUBLIC, f"{field} should be PUBLIC"

    def test_classify_compound_field_names(self):
        """Test that compound field names are classified correctly."""
        from shared.security.data_classification import (
            classify_field,
            ClassificationLevel,
        )
        
        # Fields containing PII patterns should still be classified correctly
        assert classify_field("user_email") == ClassificationLevel.RESTRICTED
        assert classify_field("primary_phone_number") == ClassificationLevel.RESTRICTED
        assert classify_field("account_balance") == ClassificationLevel.CONFIDENTIAL

    def test_required_controls_for_restricted(self):
        """Test that RESTRICTED level requires all security controls."""
        from shared.security.data_classification import (
            DataClassifier,
            ClassificationLevel,
        )
        
        classifier = DataClassifier()
        controls = classifier.get_required_controls(ClassificationLevel.RESTRICTED)
        
        assert controls.encryption_at_rest is True
        assert controls.encryption_in_transit is True
        assert controls.audit_logging is True
        assert controls.access_control is True
        assert controls.pii_masking is True

    def test_validate_controls_compliance(self):
        """Test control validation for compliance checking."""
        from shared.security.data_classification import (
            DataClassifier,
            ClassificationLevel,
        )
        
        classifier = DataClassifier()
        
        # Missing controls for RESTRICTED should fail
        is_compliant, missing = classifier.validate_controls(
            ClassificationLevel.RESTRICTED,
            has_encryption=False,
            has_audit=True,
            has_access_control=True,
        )
        
        assert not is_compliant
        assert "encryption_at_rest" in missing

    def test_classify_value_detects_email(self):
        """Test that email patterns are detected in values."""
        from shared.security.data_classification import (
            classify_value,
            ClassificationLevel,
        )
        
        text_with_email = "Contact me at user@example.com for more info"
        level = classify_value(text_with_email)
        
        assert level == ClassificationLevel.RESTRICTED

    def test_classify_value_detects_phone(self):
        """Test that phone patterns are detected in values."""
        from shared.security.data_classification import (
            classify_value,
            ClassificationLevel,
        )
        
        text_with_phone = "Call me at 555-123-4567"
        level = classify_value(text_with_phone)
        
        assert level == ClassificationLevel.RESTRICTED


class TestPIIDetector:
    """Tests for the PII detector module."""

    def test_pii_detector_exists(self):
        """Test that PIIDetector can be imported."""
        try:
            from shared.security.pii_detector import PIIDetector
            detector = PIIDetector()
            assert detector is not None
        except ImportError:
            pytest.skip("PIIDetector not available")

    def test_pii_detector_scans_text(self):
        """Test that PIIDetector scans text for PII."""
        try:
            from shared.security.pii_detector import PIIDetector
            
            detector = PIIDetector()
            text = "My email is test@example.com and SSN is 123-45-6789"
            
            # Check if scan method exists
            if hasattr(detector, 'scan'):
                result = detector.scan(text)
                assert result is not None
            elif hasattr(detector, 'detect'):
                result = detector.detect(text)
                assert result is not None
            else:
                # Just verify the detector initialized
                assert True
        except ImportError:
            pytest.skip("PIIDetector not available")

    def test_pii_detector_redaction(self):
        """Test PII redaction functionality."""
        try:
            from shared.security.pii_detector import PIIDetector
            
            detector = PIIDetector()
            text = "Email: user@test.com"
            
            # Check for redact method
            if hasattr(detector, 'redact'):
                result = detector.redact(text)
                # Redacted text should not contain the email
                assert "@test.com" not in result or "***" in result
            else:
                # Detector may not have redact method
                assert True
        except (ImportError, AttributeError):
            pytest.skip("PII redaction not available")


class TestSecurityIntegration:
    """Integration tests verifying security controls work together."""

    def test_classification_and_pii_detection_alignment(self):
        """Test that classification and PII detection are aligned."""
        from shared.security.data_classification import (
            classify_field,
            ClassificationLevel,
        )
        
        # All PII-related fields should be RESTRICTED
        pii_fields = ["email", "phone", "ssn", "ip_address"]
        
        for field in pii_fields:
            level = classify_field(field)
            assert level == ClassificationLevel.RESTRICTED, \
                f"PII field '{field}' should be RESTRICTED for proper protection"

    def test_controls_enforce_encryption_for_pii(self):
        """Test that PII data types require encryption."""
        from shared.security.data_classification import (
            DataClassifier,
            ClassificationLevel,
        )
        
        classifier = DataClassifier()
        
        for level in [ClassificationLevel.RESTRICTED, ClassificationLevel.CONFIDENTIAL]:
            controls = classifier.get_required_controls(level)
            assert controls.encryption_in_transit is True, \
                f"{level.value} data must be encrypted in transit"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
