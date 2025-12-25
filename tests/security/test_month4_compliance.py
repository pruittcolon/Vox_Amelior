"""
Month 4 Security Tests - Compliance and Audit.

Tests for SOC 2 audit logging, policy engine, and evidence collection:
- CEF formatted audit logs
- Policy enforcement
- Evidence collection
- Security gates
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ============================================================================
# SOC 2 Audit Logger Tests
# ============================================================================


class TestSOC2AuditLogger:
    """Tests for SOC 2 compliant audit logging."""

    def test_logger_initialization(self):
        """Logger initializes correctly."""
        from shared.compliance.soc2_logger import SOC2AuditLogger

        logger = SOC2AuditLogger(service_name="test-service")
        assert logger.service_name == "test-service"

    def test_event_creation(self):
        """Audit events are created with required fields."""
        from shared.compliance.soc2_logger import (
            AuditAction,
            AuditEvent,
            SOC2AuditLogger,
        )

        logger = SOC2AuditLogger()
        event = logger.log(
            action=AuditAction.LOGIN_SUCCESS,
            actor_id="user123",
            actor_ip="192.168.1.1",
        )

        assert event.event_id is not None
        assert event.timestamp is not None
        assert event.action == "AUTH_LOGIN_SUCCESS"
        assert event.actor_id == "user123"
        assert event.signature is not None  # HMAC signed

    def test_cef_format_output(self):
        """Events generate valid CEF format."""
        from shared.compliance.soc2_logger import AuditAction, SOC2AuditLogger

        logger = SOC2AuditLogger()
        event = logger.log(
            action=AuditAction.DATA_READ,
            actor_id="user456",
            resource_type="document",
            resource_id="doc-123",
        )

        cef = event.to_cef()

        assert cef.startswith("CEF:0|Nemo|NemoServer|1.0|")
        assert "DATA_READ" in cef
        assert "suid=user456" in cef

    def test_event_signature(self):
        """Events are HMAC signed."""
        from shared.compliance.soc2_logger import AuditAction, SOC2AuditLogger

        logger = SOC2AuditLogger()
        event = logger.log(action=AuditAction.LOGIN_SUCCESS)

        assert event.signature is not None
        assert len(event.signature) == 32  # Truncated SHA-256

    def test_auth_event_helper(self):
        """Authentication event helper works."""
        from shared.compliance.soc2_logger import AuditAction, SOC2AuditLogger

        logger = SOC2AuditLogger()
        event = logger.log_auth_event(
            action=AuditAction.LOGIN_FAILURE,
            user_id="baduser",
            ip_address="10.0.0.1",
            outcome="FAILURE",
        )

        assert event.outcome == "FAILURE"
        assert event.actor_ip == "10.0.0.1"
        assert event.severity >= 5  # Medium for failures

    def test_data_access_logging(self):
        """Data access events are logged correctly."""
        from shared.compliance.soc2_logger import AuditAction, SOC2AuditLogger

        logger = SOC2AuditLogger()
        event = logger.log_data_access(
            action=AuditAction.DATA_EXPORT,
            user_id="analyst",
            resource_type="report",
            resource_id="rpt-789",
        )

        assert event.resource_type == "report"
        assert event.soc2_category == "C"  # Confidentiality

    def test_config_change_logging(self):
        """Configuration changes are logged with old/new values."""
        from shared.compliance.soc2_logger import SOC2AuditLogger

        logger = SOC2AuditLogger()
        event = logger.log_config_change(
            actor_id="admin",
            resource_name="rate_limit",
            old_value=100,
            new_value=200,
        )

        assert event.details["old_value"] == "100"
        assert event.details["new_value"] == "200"
        assert event.severity >= 7  # High severity


class TestAuditActions:
    """Tests for standardized audit actions."""

    def test_all_actions_have_values(self):
        """All audit actions have string values."""
        from shared.compliance.soc2_logger import AuditAction

        for action in AuditAction:
            assert isinstance(action.value, str)
            assert len(action.value) > 0

    def test_action_categories(self):
        """Actions are properly categorized."""
        from shared.compliance.soc2_logger import AuditAction

        auth_actions = [a for a in AuditAction if a.value.startswith("AUTH_")]
        data_actions = [a for a in AuditAction if a.value.startswith("DATA_")]
        security_actions = [a for a in AuditAction if a.value.startswith("SEC_")]

        assert len(auth_actions) >= 5
        assert len(data_actions) >= 4
        assert len(security_actions) >= 4


# ============================================================================
# Policy Engine Tests
# ============================================================================


class TestPolicyEngine:
    """Tests for security policy enforcement."""

    def test_engine_initialization(self):
        """Policy engine initializes with default policies."""
        from shared.compliance.policy_engine import PolicyEngine

        engine = PolicyEngine()
        policies = engine.get_all_policies()

        assert len(policies) >= 10  # At least 10 default policies

    def test_password_length_policy(self):
        """Password length policy works."""
        from shared.compliance.policy_engine import PolicyEngine

        engine = PolicyEngine()

        # Too short
        result = engine.evaluate_password("short")
        assert not result.passed
        assert any("PWD-001" in v.policy_id for v in result.violations)

        # Long enough
        result = engine.evaluate_password("ThisIsALongPassword123!")
        pwd_length_violations = [v for v in result.violations if v.policy_id == "PWD-001"]
        assert len(pwd_length_violations) == 0

    def test_password_complexity_policy(self):
        """Password complexity is enforced."""
        from shared.compliance.policy_engine import PolicyEngine

        engine = PolicyEngine()

        # Missing special char
        result = engine.evaluate(password="Password123abc")
        complexity_violations = [v for v in result.violations if v.policy_id == "PWD-002"]
        assert len(complexity_violations) > 0

        # Has all requirements
        result = engine.evaluate(password="Password123!abc")
        complexity_violations = [v for v in result.violations if v.policy_id == "PWD-002"]
        assert len(complexity_violations) == 0

    def test_session_timeout_policy(self):
        """Session timeout is enforced."""
        from shared.compliance.policy_engine import PolicyEngine

        engine = PolicyEngine()

        # Session too old
        result = engine.evaluate_session(session_age_hours=30, idle_minutes=10)
        assert not result.passed

        # Valid session
        result = engine.evaluate_session(session_age_hours=2, idle_minutes=10)
        session_violations = [v for v in result.violations if "SES-" in v.policy_id]
        assert len(session_violations) == 0

    def test_encryption_algorithm_policy(self):
        """Deprecated algorithms are blocked."""
        from shared.compliance.policy_engine import PolicyEngine

        engine = PolicyEngine()

        # Deprecated algorithm
        result = engine.evaluate(algorithm="MD5")
        assert not result.passed
        assert any("ENC-002" in v.policy_id for v in result.violations)

        # Approved algorithm
        result = engine.evaluate(algorithm="AES-256-GCM")
        enc_violations = [v for v in result.violations if v.policy_id == "ENC-002"]
        assert len(enc_violations) == 0

    def test_policy_disable(self):
        """Policies can be disabled."""
        from shared.compliance.policy_engine import PolicyEngine

        engine = PolicyEngine()
        engine.disable_policy("PWD-001")

        result = engine.evaluate(password="short")
        assert "PWD-001" not in [v.policy_id for v in result.violations]

    def test_should_deny_property(self):
        """Result correctly identifies denial requirement."""
        from shared.compliance.policy_engine import PolicyEngine

        engine = PolicyEngine()

        result = engine.evaluate(password="bad")
        assert result.should_deny  # Password policies deny by default


# ============================================================================
# Evidence Collector Tests
# ============================================================================


class TestEvidenceCollector:
    """Tests for compliance evidence collection."""

    def test_collector_initialization(self):
        """Evidence collector initializes correctly."""
        from shared.compliance.evidence_collector import EvidenceCollector

        with tempfile.TemporaryDirectory() as tmpdir:
            collector = EvidenceCollector(evidence_path=tmpdir)
            assert collector.evidence_path.exists()

    def test_evidence_collection(self):
        """Evidence can be collected and saved."""
        from shared.compliance.evidence_collector import (
            EvidenceCollector,
            EvidenceType,
            SOC2Control,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            collector = EvidenceCollector(evidence_path=tmpdir)

            evidence = collector.collect(
                evidence_type=EvidenceType.CONFIGURATION,
                title="Test Config",
                description="Test configuration evidence",
                content="config_key=value",
                controls=[SOC2Control.CC7_1],
            )

            assert evidence.evidence_id.startswith("EVD-")
            assert evidence.content_hash is not None
            assert "CC7.1" in evidence.controls

    def test_evidence_hashing(self):
        """Evidence content is hashed for integrity."""
        from shared.compliance.evidence_collector import (
            EvidenceCollector,
            EvidenceType,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            collector = EvidenceCollector(evidence_path=tmpdir)

            evidence = collector.collect(
                evidence_type=EvidenceType.TEST_RESULT,
                title="Test Results",
                description="Security test output",
                content="All tests passed",
            )

            assert len(evidence.content_hash) == 64  # SHA-256 hex

    def test_test_results_collection(self):
        """Test results helper works."""
        from shared.compliance.evidence_collector import EvidenceCollector

        with tempfile.TemporaryDirectory() as tmpdir:
            collector = EvidenceCollector(evidence_path=tmpdir)

            evidence = collector.collect_test_results(
                test_output="10 passed, 0 failed",
                test_name="Security Suite",
            )

            assert "Security Suite" in evidence.title
            assert "CC4.1" in evidence.controls  # Monitoring control

    def test_evidence_index_generation(self):
        """Evidence index can be generated."""
        from shared.compliance.evidence_collector import (
            EvidenceCollector,
            EvidenceType,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            collector = EvidenceCollector(evidence_path=tmpdir)

            # Collect some evidence
            collector.collect(
                evidence_type=EvidenceType.CONFIGURATION,
                title="Config 1",
                description="Test",
                content="content1",
            )
            collector.collect(
                evidence_type=EvidenceType.SECURITY_SCAN,
                title="Scan 1",
                description="Test",
                content="content2",
            )

            index = collector.generate_evidence_index()

            assert index["total_evidence"] == 2
            assert len(index["evidence"]) == 2


# ============================================================================
# Integration Tests
# ============================================================================


class TestComplianceIntegration:
    """Integration tests for compliance modules."""

    def test_audit_with_policy_violation(self):
        """Policy violations are logged to audit."""
        from shared.compliance.policy_engine import PolicyEngine
        from shared.compliance.soc2_logger import AuditAction, SOC2AuditLogger

        engine = PolicyEngine()
        logger = SOC2AuditLogger()

        # Check policy
        result = engine.evaluate(password="weak")

        # Log violations
        if not result.passed:
            for violation in result.violations:
                event = logger.log(
                    action=AuditAction.POLICY_VIOLATION,
                    outcome="FAILURE",
                    details={
                        "policy_id": violation.policy_id,
                        "message": violation.message,
                    },
                )

        assert event.action == "COMPL_POLICY_VIOLATION"

    def test_evidence_from_audit_log(self):
        """Audit logs can be collected as evidence."""
        from shared.compliance.evidence_collector import (
            EvidenceCollector,
            EvidenceType,
            SOC2Control,
        )
        from shared.compliance.soc2_logger import AuditAction, SOC2AuditLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = SOC2AuditLogger()
            collector = EvidenceCollector(evidence_path=tmpdir)

            # Generate audit events
            events = []
            for i in range(3):
                event = logger.log(
                    action=AuditAction.LOGIN_SUCCESS,
                    actor_id=f"user{i}",
                )
                events.append(event.to_json())

            # Collect as evidence
            evidence = collector.collect(
                evidence_type=EvidenceType.ACCESS_LOG,
                title="Login Audit Trail",
                description="Authentication events for audit period",
                content="\n".join(events),
                controls=[SOC2Control.CC6_1, SOC2Control.CC6_2],
            )

            assert evidence.content_hash is not None
            assert "CC6.1" in evidence.controls
