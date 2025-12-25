"""
MFA Security Tests.

Tests for TOTP multi-factor authentication:
- TOTP setup flow
- TOTP verification
- Backup code usage
- MFA bypass prevention
"""

import pytest
import time
from unittest.mock import patch, MagicMock
import tempfile
import os

# Import the MFA module
import sys
sys.path.insert(0, "/home/pruittcolon/Desktop/Nemo_Server")

from shared.security.mfa import (
    generate_secret,
    generate_backup_codes,
    generate_totp,
    verify_totp,
    generate_provisioning_uri,
    hash_backup_code,
    MFAManager,
    MFASetup,
    MFAStatus,
    TOTP_DIGITS,
    TOTP_PERIOD,
)


class TestTOTPGeneration:
    """Test TOTP code generation."""
    
    def test_generate_secret_is_base32(self):
        """Generated secret should be valid base32."""
        secret = generate_secret()
        assert isinstance(secret, str)
        assert len(secret) >= 26  # 20 bytes = 32 base32 chars (without padding)
        # Base32 alphabet
        valid_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ234567")
        assert all(c in valid_chars for c in secret)
    
    def test_generate_secret_is_random(self):
        """Each generated secret should be unique."""
        secrets = [generate_secret() for _ in range(10)]
        assert len(set(secrets)) == 10
    
    def test_generate_totp_returns_digits(self):
        """Generated TOTP should be correct number of digits."""
        secret = generate_secret()
        code = generate_totp(secret)
        assert len(code) == TOTP_DIGITS
        assert code.isdigit()
    
    def test_generate_totp_consistent_for_same_time(self):
        """Same secret and timestamp should produce same code."""
        secret = generate_secret()
        timestamp = 1000000000.0  # Fixed timestamp
        code1 = generate_totp(secret, timestamp)
        code2 = generate_totp(secret, timestamp)
        assert code1 == code2
    
    def test_generate_totp_changes_with_time(self):
        """TOTP should change between time periods."""
        secret = generate_secret()
        code1 = generate_totp(secret, 1000000000.0)
        code2 = generate_totp(secret, 1000000000.0 + TOTP_PERIOD)
        # Codes should be different (extremely unlikely to be same)
        assert code1 != code2


class TestTOTPVerification:
    """Test TOTP code verification."""
    
    def test_verify_current_code(self):
        """Current code should verify successfully."""
        secret = generate_secret()
        code = generate_totp(secret)
        assert verify_totp(secret, code) is True
    
    def test_verify_rejects_wrong_code(self):
        """Wrong code should be rejected."""
        secret = generate_secret()
        assert verify_totp(secret, "000000") is False
    
    def test_verify_accepts_window(self):
        """Codes within time window should be accepted."""
        secret = generate_secret()
        now = time.time()
        # Generate code for 30 seconds ago
        old_code = generate_totp(secret, now - TOTP_PERIOD)
        # Should still verify (within window=1)
        assert verify_totp(secret, old_code, window=1, timestamp=now) is True
    
    def test_verify_rejects_outside_window(self):
        """Codes outside time window should be rejected."""
        secret = generate_secret()
        now = time.time()
        # Generate code for 5 periods ago
        very_old_code = generate_totp(secret, now - (TOTP_PERIOD * 5))
        # Should not verify (window=1)
        assert verify_totp(secret, very_old_code, window=1, timestamp=now) is False
    
    def test_verify_handles_spaces(self):
        """Verification should handle formatted codes."""
        secret = generate_secret()
        code = generate_totp(secret)
        # Add spaces
        formatted_code = f"{code[:3]} {code[3:]}"
        assert verify_totp(secret, formatted_code) is True
    
    def test_verify_rejects_non_numeric(self):
        """Non-numeric codes should be rejected."""
        secret = generate_secret()
        assert verify_totp(secret, "abcdef") is False
        assert verify_totp(secret, "12345") is False  # Wrong length


class TestBackupCodes:
    """Test backup code generation and hashing."""
    
    def test_generate_backup_codes_count(self):
        """Should generate correct number of backup codes."""
        codes = generate_backup_codes(10)
        assert len(codes) == 10
    
    def test_generate_backup_codes_format(self):
        """Backup codes should be in XXXX-XXXX format."""
        codes = generate_backup_codes(1)
        assert len(codes[0]) == 9  # 4 + 1 + 4
        assert codes[0][4] == "-"
    
    def test_backup_codes_are_unique(self):
        """Each backup code should be unique."""
        codes = generate_backup_codes(100)
        assert len(set(codes)) == 100
    
    def test_hash_backup_code_consistent(self):
        """Same code should produce same hash."""
        code = "ABCD-1234"
        hash1 = hash_backup_code(code)
        hash2 = hash_backup_code(code)
        assert hash1 == hash2
    
    def test_hash_backup_code_normalizes(self):
        """Hash should normalize case and remove dashes."""
        hash1 = hash_backup_code("ABCD-1234")
        hash2 = hash_backup_code("abcd1234")
        assert hash1 == hash2


class TestProvisioningURI:
    """Test QR code provisioning URI generation."""
    
    def test_uri_format(self):
        """URI should be valid otpauth format."""
        secret = generate_secret()
        uri = generate_provisioning_uri(secret, "testuser")
        assert uri.startswith("otpauth://totp/")
        assert secret in uri
        assert "testuser" in uri
    
    def test_uri_contains_issuer(self):
        """URI should contain issuer parameter."""
        secret = generate_secret()
        uri = generate_provisioning_uri(secret, "testuser", issuer="TestApp")
        assert "issuer=TestApp" in uri


class TestMFAManager:
    """Test MFA manager with database storage."""
    
    @pytest.fixture
    def mfa_manager(self):
        """Create MFA manager with temp database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_mfa.db")
            manager = MFAManager(storage_path=db_path)
            yield manager
    
    def test_setup_returns_secret_and_codes(self, mfa_manager):
        """Setup should return secret and backup codes."""
        setup = mfa_manager.setup_mfa("user1", "user1@example.com")
        assert isinstance(setup, MFASetup)
        assert len(setup.secret) >= 26
        assert len(setup.backup_codes) == 10
        assert "otpauth://" in setup.provisioning_uri
    
    def test_mfa_not_enabled_until_verified(self, mfa_manager):
        """MFA should not be enabled until verified."""
        mfa_manager.setup_mfa("user2", "user2@example.com")
        status = mfa_manager.get_status("user2")
        assert status.enabled is False
    
    def test_verify_and_enable(self, mfa_manager):
        """Correct code should enable MFA."""
        setup = mfa_manager.setup_mfa("user3", "user3@example.com")
        code = generate_totp(setup.secret)
        assert mfa_manager.verify_and_enable("user3", code) is True
        
        status = mfa_manager.get_status("user3")
        assert status.enabled is True
    
    def test_verify_rejects_wrong_code(self, mfa_manager):
        """Wrong code should not enable MFA."""
        mfa_manager.setup_mfa("user4", "user4@example.com")
        assert mfa_manager.verify_and_enable("user4", "000000") is False
        
        status = mfa_manager.get_status("user4")
        assert status.enabled is False
    
    def test_verify_accepts_backup_code(self, mfa_manager):
        """Backup code should verify successfully."""
        setup = mfa_manager.setup_mfa("user5", "user5@example.com")
        # Enable MFA first
        code = generate_totp(setup.secret)
        mfa_manager.verify_and_enable("user5", code)
        
        # Use a backup code
        backup_code = setup.backup_codes[0]
        assert mfa_manager.verify("user5", backup_code) is True
    
    def test_backup_code_consumed_after_use(self, mfa_manager):
        """Backup code should only work once."""
        setup = mfa_manager.setup_mfa("user6", "user6@example.com")
        code = generate_totp(setup.secret)
        mfa_manager.verify_and_enable("user6", code)
        
        backup_code = setup.backup_codes[0]
        # First use
        assert mfa_manager.verify("user6", backup_code) is True
        # Second use - should fail
        assert mfa_manager.verify("user6", backup_code) is False
    
    def test_disable_removes_mfa(self, mfa_manager):
        """Disable should remove MFA from account."""
        setup = mfa_manager.setup_mfa("user7", "user7@example.com")
        code = generate_totp(setup.secret)
        mfa_manager.verify_and_enable("user7", code)
        
        mfa_manager.disable("user7")
        status = mfa_manager.get_status("user7")
        assert status.enabled is False
    
    def test_is_required_when_enabled(self, mfa_manager):
        """is_required should return True when MFA is enabled."""
        setup = mfa_manager.setup_mfa("user8", "user8@example.com")
        code = generate_totp(setup.secret)
        mfa_manager.verify_and_enable("user8", code)
        
        assert mfa_manager.is_required("user8") is True
    
    def test_is_required_when_disabled(self, mfa_manager):
        """is_required should return False when MFA is not enabled."""
        assert mfa_manager.is_required("nonexistent_user") is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
