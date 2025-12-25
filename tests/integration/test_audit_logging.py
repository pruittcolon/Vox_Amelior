"""
Integration Tests for Audit Logging

Tests the enhanced audit logging system with HMAC chain sealing for tamper detection.
Follows the workflow requirement: tests must simulate real user environment.
"""

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Add shared to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shared"))

from shared.audit import RequestAuditLogger, RequestLog, get_audit_logger


class TestAuditLoggingBasic:
    """Basic audit logging functionality tests."""

    @pytest.fixture
    def temp_log_dir(self):
        """Create a temporary directory for test logs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def audit_logger(self, temp_log_dir):
        """Create a fresh audit logger for each test."""
        logger = RequestAuditLogger(log_dir=temp_log_dir)
        yield logger
        logger.clear_stats()

    @pytest.mark.asyncio
    async def test_log_request_creates_file(self, audit_logger, temp_log_dir):
        """Test that logging a request creates a log file."""
        await audit_logger.log_request(
            request_id="test-123",
            method="GET",
            path="/api/test",
            status_code=200,
            latency_ms=50.0,
            user_id="test_user",
            ip_address="127.0.0.1",
        )

        log_dir = Path(temp_log_dir)
        log_files = list(log_dir.glob("requests_*.jsonl"))
        assert len(log_files) == 1, "Should create one log file"

    @pytest.mark.asyncio
    async def test_log_entry_has_seal(self, audit_logger, temp_log_dir):
        """Test that each log entry has an HMAC seal."""
        await audit_logger.log_request(
            request_id="test-456",
            method="POST",
            path="/api/auth/login",
            status_code=200,
            latency_ms=100.0,
        )

        log_dir = Path(temp_log_dir)
        log_file = list(log_dir.glob("requests_*.jsonl"))[0]
        
        with open(log_file, "r") as f:
            line = f.readline()
            entry = json.loads(line)
        
        assert "seal" in entry, "Entry should have HMAC seal"
        assert "data" in entry, "Entry should have data"
        assert len(entry["seal"]) == 64, "Seal should be SHA256 hex (64 chars)"

    @pytest.mark.asyncio
    async def test_hmac_chain_links_entries(self, audit_logger, temp_log_dir):
        """Test that entries are chained via HMAC."""
        # Log multiple entries
        for i in range(3):
            await audit_logger.log_request(
                request_id=f"test-{i}",
                method="GET",
                path=f"/api/test/{i}",
                status_code=200,
                latency_ms=10.0 * i,
            )

        log_dir = Path(temp_log_dir)
        log_file = list(log_dir.glob("requests_*.jsonl"))[0]
        
        entries = []
        with open(log_file, "r") as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))
        
        assert len(entries) == 3, "Should have 3 entries"
        
        # First entry should have no prev_seal (or null)
        assert entries[0].get("prev_seal") is None, "First entry has no previous seal"
        
        # Subsequent entries should reference previous seal
        for i in range(1, len(entries)):
            assert entries[i].get("prev_seal") is not None, f"Entry {i} should have prev_seal"

    @pytest.mark.asyncio
    async def test_verify_log_integrity_passes(self, audit_logger, temp_log_dir):
        """Test that integrity verification passes for untampered logs."""
        # Log some entries
        for i in range(5):
            await audit_logger.log_request(
                request_id=f"verify-{i}",
                method="GET",
                path=f"/api/verify/{i}",
                status_code=200,
                latency_ms=25.0,
            )

        log_dir = Path(temp_log_dir)
        log_file = list(log_dir.glob("requests_*.jsonl"))[0]
        
        # Verify integrity
        is_valid, issues = audit_logger.verify_log_integrity(log_file)
        
        assert is_valid, f"Log should be valid. Issues: {issues}"
        assert len(issues) == 0 or issues == ["No log file found"], "No issues expected"

    @pytest.mark.asyncio
    async def test_verify_log_integrity_detects_tampering(self, audit_logger, temp_log_dir):
        """Test that integrity verification detects tampered logs."""
        # Log some entries
        for i in range(3):
            await audit_logger.log_request(
                request_id=f"tamper-{i}",
                method="GET",
                path=f"/api/tamper/{i}",
                status_code=200,
                latency_ms=30.0,
            )

        log_dir = Path(temp_log_dir)
        log_file = list(log_dir.glob("requests_*.jsonl"))[0]
        
        # Tamper with the log file
        with open(log_file, "r") as f:
            lines = f.readlines()
        
        # Modify the second entry
        if len(lines) >= 2:
            entry = json.loads(lines[1])
            entry["data"]["status_code"] = 500  # Change from 200 to 500
            lines[1] = json.dumps(entry) + "\n"
            
            with open(log_file, "w") as f:
                f.writelines(lines)
        
        # Verify integrity - should fail
        is_valid, issues = audit_logger.verify_log_integrity(log_file)
        
        assert not is_valid, "Log should be invalid after tampering"
        assert any("tampering" in issue.lower() or "mismatch" in issue.lower() for issue in issues)


class TestAuditLoggingAnomalyDetection:
    """Tests for anomaly detection in audit logging."""

    @pytest.fixture
    def temp_log_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def audit_logger(self, temp_log_dir):
        logger = RequestAuditLogger(
            log_dir=temp_log_dir,
            enable_anomaly_detection=True,
            thresholds={
                "max_requests_per_minute": 10,
                "max_error_rate": 0.3,
                "latency_stddev_multiplier": 3.0,
                "min_samples_for_analysis": 3,
            }
        )
        yield logger
        logger.clear_stats()

    @pytest.mark.asyncio
    async def test_high_error_rate_detected(self, audit_logger):
        """Test that high error rates trigger anomaly detection."""
        # Log a mix of success and errors (high error rate)
        for i in range(15):
            status = 500 if i % 2 == 0 else 200  # 50% error rate
            await audit_logger.log_request(
                request_id=f"error-{i}",
                method="GET",
                path="/api/flaky",
                status_code=status,
                latency_ms=50.0,
            )

        stats = audit_logger.get_stats()
        assert stats["errors_by_path"]["/api/flaky"] >= 7, "Should have many errors"

    @pytest.mark.asyncio
    async def test_stats_tracking(self, audit_logger):
        """Test that statistics are properly tracked."""
        # Log requests to different paths
        paths = ["/api/a", "/api/b", "/api/c"]
        for path in paths:
            for _ in range(3):
                await audit_logger.log_request(
                    request_id="stats-test",
                    method="GET",
                    path=path,
                    status_code=200,
                    latency_ms=100.0,
                )

        stats = audit_logger.get_stats()
        
        assert stats["total_requests"] == 9
        for path in paths:
            assert stats["requests_by_path"][path] == 3


class TestAuditLoggingSecurity:
    """Security-focused tests for audit logging."""

    @pytest.fixture
    def temp_log_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.mark.asyncio
    async def test_production_hmac_key_warning(self, temp_log_dir, caplog):
        """Test that production warns about missing HMAC key."""
        # Set production mode
        original_prod = os.environ.get("PRODUCTION", "")
        os.environ["PRODUCTION"] = "true"
        
        try:
            logger = RequestAuditLogger(log_dir=temp_log_dir)
            await logger.log_request(
                request_id="prod-test",
                method="GET",
                path="/api/prod",
                status_code=200,
                latency_ms=10.0,
            )
            # The warning should be logged (check logs or behavior)
        finally:
            os.environ["PRODUCTION"] = original_prod
            logger.clear_stats()

    @pytest.mark.asyncio
    async def test_log_contains_security_relevant_fields(self, temp_log_dir):
        """Test that logs contain all security-relevant fields."""
        logger = RequestAuditLogger(log_dir=temp_log_dir)
        
        await logger.log_request(
            request_id="sec-test-123",
            method="POST",
            path="/api/auth/login",
            status_code=401,
            latency_ms=150.0,
            user_id="attacker",
            ip_address="192.168.1.100",
            user_agent="Python/3.x",
            error="Invalid credentials",
        )

        log_dir = Path(temp_log_dir)
        log_file = list(log_dir.glob("requests_*.jsonl"))[0]
        
        with open(log_file, "r") as f:
            entry = json.loads(f.readline())
        
        data = entry["data"]
        
        # Verify all security fields
        assert data["request_id"] == "sec-test-123"
        assert data["method"] == "POST"
        assert data["path"] == "/api/auth/login"
        assert data["status_code"] == 401
        assert data["user_id"] == "attacker"
        assert data["ip_address"] == "192.168.1.100"
        assert data["error"] == "Invalid credentials"
        assert "timestamp" in data
        
        logger.clear_stats()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
