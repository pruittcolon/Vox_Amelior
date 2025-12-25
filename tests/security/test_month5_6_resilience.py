"""
Month 5-6 Security Tests - Resilience and Certification.

Tests for backup, disaster recovery, and security assessment:
- Encrypted backup creation/restoration
- DR procedure execution
- Security assessment accuracy
"""

import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ============================================================================
# Secure Backup Tests
# ============================================================================


class TestSecureBackup:
    """Tests for encrypted backup functionality."""

    def test_backup_manager_initialization(self):
        """Backup manager initializes correctly."""
        from shared.ops.secure_backup import SecureBackupManager

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = SecureBackupManager(backup_path=tmpdir)
            assert manager.backup_path.exists()

    def test_backup_directory(self):
        """Directories can be backed up."""
        from shared.ops.secure_backup import BackupStatus, SecureBackupManager

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create source directory
            source = Path(tmpdir) / "source"
            source.mkdir()
            (source / "file.txt").write_text("test content")

            backup_path = Path(tmpdir) / "backups"
            manager = SecureBackupManager(backup_path=str(backup_path))

            # Create backup
            metadata = manager.backup_directory(str(source))

            assert metadata.status == BackupStatus.COMPLETED
            assert metadata.encrypted_size > 0
            assert metadata.checksum is not None

    def test_backup_encryption(self):
        """Backups are encrypted."""
        from shared.ops.secure_backup import SecureBackupManager

        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source"
            source.mkdir()
            (source / "secret.txt").write_text("sensitive data")

            backup_path = Path(tmpdir) / "backups"
            manager = SecureBackupManager(backup_path=str(backup_path))

            metadata = manager.backup_directory(str(source))

            # Read encrypted file
            encrypted_file = Path(metadata.file_path)
            encrypted_content = encrypted_file.read_bytes()

            # Should not contain plaintext
            assert b"sensitive data" not in encrypted_content

    def test_backup_restore(self):
        """Backups can be restored."""
        from shared.ops.secure_backup import SecureBackupManager

        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source"
            source.mkdir()
            (source / "data.txt").write_text("restore me")

            backup_path = Path(tmpdir) / "backups"
            restore_path = Path(tmpdir) / "restored"
            
            manager = SecureBackupManager(backup_path=str(backup_path))

            # Backup
            metadata = manager.backup_directory(str(source))

            # Restore
            success = manager.restore_backup(metadata.backup_id, str(restore_path))

            assert success
            restored_file = restore_path / "source" / "data.txt"
            assert restored_file.read_text() == "restore me"

    def test_backup_verification(self):
        """Backup integrity can be verified."""
        from shared.ops.secure_backup import SecureBackupManager

        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source"
            source.mkdir()
            (source / "verify.txt").write_text("integrity check")

            backup_path = Path(tmpdir) / "backups"
            manager = SecureBackupManager(backup_path=str(backup_path))

            metadata = manager.backup_directory(str(source))

            # Verify
            is_valid = manager.verify_backup(metadata.backup_id)
            assert is_valid

    def test_list_backups(self):
        """Backups can be listed."""
        from shared.ops.secure_backup import BackupType, SecureBackupManager

        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source"
            source.mkdir()
            (source / "file.txt").write_text("content")

            backup_path = Path(tmpdir) / "backups"
            manager = SecureBackupManager(backup_path=str(backup_path))

            # Create multiple backups
            manager.backup_directory(str(source), BackupType.CONFIGURATION)
            manager.backup_directory(str(source), BackupType.AUDIT_LOGS)

            backups = manager.list_backups()
            assert len(backups) == 2


# ============================================================================
# Disaster Recovery Tests
# ============================================================================


class TestDisasterRecovery:
    """Tests for disaster recovery procedures."""

    def test_dr_manager_initialization(self):
        """DR manager initializes with RTO/RPO."""
        from shared.ops.disaster_recovery import DisasterRecoveryManager

        manager = DisasterRecoveryManager(rto_minutes=60, rpo_minutes=30)

        assert manager.rto_minutes == 60
        assert manager.rpo_minutes == 30

    def test_register_recovery_point(self):
        """Recovery points can be registered."""
        from shared.ops.disaster_recovery import DisasterRecoveryManager

        manager = DisasterRecoveryManager()

        rp = manager.register_recovery_point(backup_id="backup-123")

        assert rp.point_id.startswith("RP-")
        assert rp.backup_id == "backup-123"

    def test_get_latest_recovery_point(self):
        """Latest recovery point is retrieved."""
        from shared.ops.disaster_recovery import DisasterRecoveryManager

        manager = DisasterRecoveryManager()

        manager.register_recovery_point(backup_id="old")
        manager.register_recovery_point(backup_id="new")

        latest = manager.get_latest_recovery_point()
        assert latest.backup_id == "new"

    def test_rto_status(self):
        """RTO/RPO status is calculated."""
        from shared.ops.disaster_recovery import DisasterRecoveryManager

        manager = DisasterRecoveryManager(rpo_minutes=60)
        manager.register_recovery_point()

        status = manager.get_rto_status()

        assert "rto_minutes" in status
        assert "rpo_minutes" in status
        assert "rpo_status" in status

    def test_dry_run_recovery(self):
        """Dry run recovery simulates without changes."""
        from shared.ops.disaster_recovery import DisasterRecoveryManager, RecoveryPhase

        manager = DisasterRecoveryManager()
        manager.register_recovery_point(backup_id="test-backup")

        results = manager.execute_recovery(dry_run=True)

        # Should have multiple phases
        assert len(results) > 0
        # At least some phases should complete
        assert any(r.success for r in results)


# ============================================================================
# Security Assessment Tests
# ============================================================================


class TestSecurityAssessment:
    """Tests for security assessment tool."""

    def test_assessment_initialization(self):
        """Assessment initializes correctly."""
        # Import from scripts directory
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
        
        from security_assessment import SecurityAssessment

        assessment = SecurityAssessment()
        assert len(assessment.findings) == 0

    def test_run_assessment(self):
        """Full assessment can be run."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
        
        from security_assessment import SecurityAssessment

        assessment = SecurityAssessment()
        result = assessment.run_assessment()

        assert result.overall_score >= 0
        assert result.overall_score <= 10
        assert result.timestamp is not None

    def test_finding_severity_levels(self):
        """Findings have proper severity levels."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
        
        from security_assessment import AssessmentFinding

        finding = AssessmentFinding(
            category="Test",
            severity="HIGH",
            title="Test Finding",
            description="Test description",
            recommendation="Fix it",
        )

        assert finding.severity in ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]


# ============================================================================
# Integration Tests
# ============================================================================


class TestResilienceIntegration:
    """Integration tests for resilience features."""

    def test_backup_and_dr_integration(self):
        """Backup creates recovery point for DR."""
        from shared.ops.disaster_recovery import DisasterRecoveryManager
        from shared.ops.secure_backup import SecureBackupManager

        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source"
            source.mkdir()
            (source / "data.txt").write_text("important")

            backup_path = Path(tmpdir) / "backups"
            backup_mgr = SecureBackupManager(backup_path=str(backup_path))
            dr_mgr = DisasterRecoveryManager()

            # Create backup
            metadata = backup_mgr.backup_directory(str(source))

            # Register as recovery point
            rp = dr_mgr.register_recovery_point(backup_id=metadata.backup_id)

            # Verify linkage
            latest_rp = dr_mgr.get_latest_recovery_point()
            assert latest_rp.backup_id == metadata.backup_id

    def test_rpo_compliance_tracking(self):
        """RPO compliance is tracked correctly."""
        from shared.ops.disaster_recovery import DisasterRecoveryManager

        manager = DisasterRecoveryManager(rpo_minutes=1)  # 1 minute RPO

        # Register recent recovery point
        manager.register_recovery_point()

        status = manager.get_rto_status()
        # Just created, should be compliant
        assert status["rpo_status"] == "COMPLIANT"
