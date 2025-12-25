"""
Multi-Factor Authentication (MFA) Module - TOTP Implementation.

Provides TOTP-based MFA:
- Secret generation with QR code support
- TOTP verification against RFC 6238
- Backup code generation and usage
- Secure storage of MFA secrets

ISO 27002 Control: A.9.4.2 - Secure Log-on Procedures
"""

import base64
import hashlib
import hmac
import io
import logging
import os
import secrets
import struct
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# Configuration
TOTP_DIGITS = 6
TOTP_PERIOD = 30  # seconds
TOTP_WINDOW = 1  # Allow +/- 1 period for clock drift
BACKUP_CODE_COUNT = 10
BACKUP_CODE_LENGTH = 8

# MFA issuer name for authenticator apps
MFA_ISSUER = os.getenv("MFA_ISSUER", "NemoServer")


@dataclass
class MFASetup:
    """MFA setup result with secret and provisioning URI."""
    secret: str  # Base32 encoded secret
    provisioning_uri: str  # otpauth:// URI for QR codes
    backup_codes: list[str]  # One-time backup codes


@dataclass
class MFAStatus:
    """User's MFA status."""
    enabled: bool
    backup_codes_remaining: int = 0


def generate_secret(length: int = 20) -> str:
    """
    Generate a cryptographically secure TOTP secret.
    
    Args:
        length: Number of random bytes (20 = 160 bits, RFC 4226 minimum)
        
    Returns:
        Base32 encoded secret string
    """
    secret_bytes = secrets.token_bytes(length)
    return base64.b32encode(secret_bytes).decode("utf-8").rstrip("=")


def generate_backup_codes(count: int = BACKUP_CODE_COUNT) -> list[str]:
    """
    Generate single-use backup codes.
    
    Args:
        count: Number of codes to generate
        
    Returns:
        List of backup code strings
    """
    codes = []
    for _ in range(count):
        # Format: XXXX-XXXX for readability
        code = secrets.token_hex(BACKUP_CODE_LENGTH // 2).upper()
        codes.append(f"{code[:4]}-{code[4:]}")
    return codes


def hash_backup_code(code: str) -> str:
    """
    Hash a backup code for secure storage.
    
    Args:
        code: Raw backup code
        
    Returns:
        SHA-256 hash of normalized code
    """
    normalized = code.upper().replace("-", "")
    return hashlib.sha256(normalized.encode()).hexdigest()


def generate_provisioning_uri(
    secret: str,
    username: str,
    issuer: str = MFA_ISSUER,
) -> str:
    """
    Generate otpauth:// URI for authenticator apps.
    
    Args:
        secret: Base32 encoded secret
        username: User's account name
        issuer: Service name (displayed in authenticator)
        
    Returns:
        otpauth:// URI string
    """
    from urllib.parse import quote
    
    label = f"{issuer}:{username}"
    params = f"secret={secret}&issuer={quote(issuer)}&algorithm=SHA1&digits={TOTP_DIGITS}&period={TOTP_PERIOD}"
    return f"otpauth://totp/{quote(label)}?{params}"


def generate_totp(secret: str, timestamp: Optional[float] = None) -> str:
    """
    Generate TOTP code for current time period.
    
    Implementation follows RFC 6238 (TOTP) and RFC 4226 (HOTP).
    
    Args:
        secret: Base32 encoded secret
        timestamp: Unix timestamp (defaults to current time)
        
    Returns:
        6-digit TOTP code
    """
    if timestamp is None:
        timestamp = time.time()
    
    # Calculate time counter
    counter = int(timestamp) // TOTP_PERIOD
    
    # Decode secret (add padding if needed)
    secret_padded = secret + "=" * (8 - len(secret) % 8) if len(secret) % 8 else secret
    try:
        key = base64.b32decode(secret_padded.upper())
    except Exception:
        raise ValueError("Invalid TOTP secret")
    
    # HOTP algorithm (RFC 4226)
    counter_bytes = struct.pack(">Q", counter)
    hmac_hash = hmac.new(key, counter_bytes, hashlib.sha1).digest()
    
    # Dynamic truncation
    offset = hmac_hash[-1] & 0x0F
    code = struct.unpack(">I", hmac_hash[offset:offset + 4])[0]
    code = (code & 0x7FFFFFFF) % (10 ** TOTP_DIGITS)
    
    return str(code).zfill(TOTP_DIGITS)


def verify_totp(
    secret: str,
    code: str,
    window: int = TOTP_WINDOW,
    timestamp: Optional[float] = None,
) -> bool:
    """
    Verify TOTP code with time window for clock drift.
    
    Args:
        secret: Base32 encoded secret
        code: User-provided TOTP code
        window: Number of periods to check (+/-)
        timestamp: Unix timestamp (defaults to current time)
        
    Returns:
        True if code is valid
    """
    if timestamp is None:
        timestamp = time.time()
    
    # Normalize code
    code = code.strip().replace(" ", "")
    if len(code) != TOTP_DIGITS or not code.isdigit():
        return False
    
    # Check current and adjacent time periods
    for offset in range(-window, window + 1):
        check_time = timestamp + (offset * TOTP_PERIOD)
        expected = generate_totp(secret, check_time)
        if hmac.compare_digest(code, expected):
            return True
    
    return False


class MFAManager:
    """
    Manages MFA enrollment and verification.
    
    Stores MFA secrets and backup codes associated with user accounts.
    In production, these should be encrypted at rest.
    
    Usage:
        mfa = MFAManager(storage_backend)
        
        # Setup MFA
        setup = mfa.setup_mfa(user_id, username)
        # Show setup.provisioning_uri as QR code
        # Store setup.backup_codes securely
        
        # Verify and enable
        if mfa.verify_and_enable(user_id, totp_code):
            print("MFA enabled!")
        
        # Verify on login
        if mfa.verify(user_id, code):
            print("MFA verified!")
    """
    
    def __init__(self, storage_path: str = "/instance/mfa_secrets.db"):
        """
        Initialize MFA manager.
        
        Args:
            storage_path: Path to SQLite database for MFA secrets
        """
        import sqlite3
        from pathlib import Path
        
        self.storage_path = storage_path
        Path(storage_path).parent.mkdir(parents=True, exist_ok=True)
        
        self._init_database()
        logger.info("MFAManager initialized", storage_path=storage_path)
    
    def _init_database(self):
        """Create MFA tables if they don't exist."""
        import sqlite3
        
        conn = sqlite3.connect(self.storage_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS mfa_secrets (
                user_id TEXT PRIMARY KEY,
                secret TEXT NOT NULL,
                enabled INTEGER DEFAULT 0,
                created_at REAL NOT NULL,
                enabled_at REAL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS mfa_backup_codes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                code_hash TEXT NOT NULL,
                used INTEGER DEFAULT 0,
                used_at REAL,
                FOREIGN KEY (user_id) REFERENCES mfa_secrets(user_id)
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_backup_user ON mfa_backup_codes(user_id)")
        conn.commit()
        conn.close()
        
        # Secure file permissions
        try:
            from pathlib import Path
            Path(self.storage_path).chmod(0o600)
        except Exception:
            pass
    
    def _connect(self):
        """Get database connection."""
        import sqlite3
        conn = sqlite3.connect(self.storage_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def setup_mfa(self, user_id: str, username: str) -> MFASetup:
        """
        Generate MFA secret and backup codes for user.
        
        Does NOT enable MFA - user must verify first.
        
        Args:
            user_id: User's unique identifier
            username: User's display name (for authenticator)
            
        Returns:
            MFASetup with secret, URI, and backup codes
        """
        # Generate new secret
        secret = generate_secret()
        backup_codes = generate_backup_codes()
        provisioning_uri = generate_provisioning_uri(secret, username)
        
        # Store (not yet enabled)
        conn = self._connect()
        now = time.time()
        
        # Delete existing setup if any
        conn.execute("DELETE FROM mfa_backup_codes WHERE user_id = ?", (user_id,))
        conn.execute("DELETE FROM mfa_secrets WHERE user_id = ?", (user_id,))
        
        # Insert new secret
        conn.execute(
            "INSERT INTO mfa_secrets (user_id, secret, enabled, created_at) VALUES (?, ?, 0, ?)",
            (user_id, secret, now)
        )
        
        # Insert backup codes (hashed)
        for code in backup_codes:
            code_hash = hash_backup_code(code)
            conn.execute(
                "INSERT INTO mfa_backup_codes (user_id, code_hash, used) VALUES (?, ?, 0)",
                (user_id, code_hash)
            )
        
        conn.commit()
        conn.close()
        
        logger.info("mfa_setup_created", user_id=user_id)
        
        return MFASetup(
            secret=secret,
            provisioning_uri=provisioning_uri,
            backup_codes=backup_codes,
        )
    
    def verify_and_enable(self, user_id: str, code: str) -> bool:
        """
        Verify TOTP code and enable MFA if valid.
        
        Args:
            user_id: User's unique identifier
            code: TOTP code from authenticator
            
        Returns:
            True if verified and enabled
        """
        conn = self._connect()
        row = conn.execute(
            "SELECT secret, enabled FROM mfa_secrets WHERE user_id = ?",
            (user_id,)
        ).fetchone()
        
        if not row:
            conn.close()
            return False
        
        secret = row["secret"]
        
        if not verify_totp(secret, code):
            conn.close()
            logger.warning("mfa_enable_failed", user_id=user_id, reason="invalid_code")
            return False
        
        # Enable MFA
        conn.execute(
            "UPDATE mfa_secrets SET enabled = 1, enabled_at = ? WHERE user_id = ?",
            (time.time(), user_id)
        )
        conn.commit()
        conn.close()
        
        logger.info("mfa_enabled", user_id=user_id)
        return True
    
    def verify(self, user_id: str, code: str) -> bool:
        """
        Verify TOTP or backup code.
        
        Args:
            user_id: User's unique identifier
            code: TOTP code or backup code
            
        Returns:
            True if valid
        """
        conn = self._connect()
        row = conn.execute(
            "SELECT secret, enabled FROM mfa_secrets WHERE user_id = ?",
            (user_id,)
        ).fetchone()
        
        if not row or not row["enabled"]:
            conn.close()
            return True  # MFA not enabled = pass
        
        secret = row["secret"]
        
        # Try TOTP first
        if verify_totp(secret, code):
            conn.close()
            return True
        
        # Try backup code
        code_hash = hash_backup_code(code)
        backup_row = conn.execute(
            "SELECT id FROM mfa_backup_codes WHERE user_id = ? AND code_hash = ? AND used = 0",
            (user_id, code_hash)
        ).fetchone()
        
        if backup_row:
            # Mark as used
            conn.execute(
                "UPDATE mfa_backup_codes SET used = 1, used_at = ? WHERE id = ?",
                (time.time(), backup_row["id"])
            )
            conn.commit()
            conn.close()
            logger.warning("mfa_backup_code_used", user_id=user_id)
            return True
        
        conn.close()
        logger.warning("mfa_verification_failed", user_id=user_id)
        return False
    
    def get_status(self, user_id: str) -> MFAStatus:
        """
        Get user's MFA status.
        
        Args:
            user_id: User's unique identifier
            
        Returns:
            MFAStatus
        """
        conn = self._connect()
        row = conn.execute(
            "SELECT enabled FROM mfa_secrets WHERE user_id = ?",
            (user_id,)
        ).fetchone()
        
        if not row:
            conn.close()
            return MFAStatus(enabled=False)
        
        backup_count = conn.execute(
            "SELECT COUNT(*) as cnt FROM mfa_backup_codes WHERE user_id = ? AND used = 0",
            (user_id,)
        ).fetchone()["cnt"]
        
        conn.close()
        return MFAStatus(
            enabled=bool(row["enabled"]),
            backup_codes_remaining=backup_count,
        )
    
    def disable(self, user_id: str) -> bool:
        """
        Disable MFA for user.
        
        Args:
            user_id: User's unique identifier
            
        Returns:
            True if disabled
        """
        conn = self._connect()
        conn.execute("DELETE FROM mfa_backup_codes WHERE user_id = ?", (user_id,))
        conn.execute("DELETE FROM mfa_secrets WHERE user_id = ?", (user_id,))
        conn.commit()
        conn.close()
        
        logger.info("mfa_disabled", user_id=user_id)
        return True
    
    def is_required(self, user_id: str) -> bool:
        """
        Check if MFA verification is required for user.
        
        Args:
            user_id: User's unique identifier
            
        Returns:
            True if MFA is enabled and required
        """
        return self.get_status(user_id).enabled


# Singleton instance
_mfa_manager: Optional[MFAManager] = None


def get_mfa_manager() -> MFAManager:
    """Get or create MFA manager singleton."""
    global _mfa_manager
    if _mfa_manager is None:
        _mfa_manager = MFAManager()
    return _mfa_manager
