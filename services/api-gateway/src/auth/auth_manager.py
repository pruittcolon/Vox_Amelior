"""
Enhanced Authentication and Authorization System
Role-based access control with speaker-based data isolation, encrypted sessions, and persistent storage
"""

import base64
import json
import os
import secrets
import sqlite3
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import bcrypt
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding as sym_padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from shared.crypto.db_encryption import SQLCIPHER_AVAILABLE, create_encrypted_db
from shared.security.lockout import get_lockout_manager, LockoutStatus

# 2025 Best Practice: Structured logging instead of print statements
try:
    import structlog

    logger = structlog.get_logger("auth")
except ImportError:
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    _base_logger = logging.getLogger("auth")

    class _StructlogCompat:
        """Minimal structlog-like wrapper for stdlib logging."""

        def __init__(self, base: logging.Logger):
            self._base = base

        @staticmethod
        def _format(event: str, kv: dict[str, object]) -> str:
            if not kv:
                return event
            parts = " ".join(f"{k}={v}" for k, v in kv.items())
            return f"{event} {parts}"

        def debug(self, event: str, *args, **kv):
            self._base.debug(self._format(event, kv), *args)

        def info(self, event: str, *args, **kv):
            self._base.info(self._format(event, kv), *args)

        def warning(self, event: str, *args, **kv):
            self._base.warning(self._format(event, kv), *args)

        def error(self, event: str, *args, **kv):
            self._base.error(self._format(event, kv), *args)

        def critical(self, event: str, *args, **kv):
            self._base.critical(self._format(event, kv), *args)

        def exception(self, event: str, *args, **kv):
            self._base.exception(self._format(event, kv), *args)

    logger = _StructlogCompat(_base_logger)


class UserRole(str, Enum):
    ADMIN = "admin"  # Full system access, sees all transcripts
    USER = "user"  # Limited access, sees only own transcripts (speaker-based isolation)
    MSR = "msr"  # Member Service Representative
    LOAN_OFFICER = "loan_officer"
    EXECUTIVE = "executive"
    FRAUD_ANALYST = "fraud_analyst"


@dataclass
class User:
    user_id: str
    username: str
    password_hash: str
    role: UserRole
    speaker_id: str | None = None  # Maps to speaker identity for data isolation
    email: str | None = None
    created_at: float | None = None
    modified_at: float | None = None


@dataclass
class Session:
    session_token: str
    user_id: str
    role: UserRole
    speaker_id: str | None
    created_at: float
    expires_at: float
    ip_address: str | None = None
    last_refresh: float | None = None
    csrf_token: str | None = None


def _env_flag(var_name: str, default: bool) -> bool:
    value = os.getenv(var_name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes"}


SESSION_TOKEN_ENABLE_V2 = _env_flag("SESSION_TOKEN_ENABLE_V2", True)
SESSION_TOKEN_EMIT_V2 = _env_flag("SESSION_TOKEN_EMIT_V2", True)
SESSION_TOKEN_ACCEPT_V1 = _env_flag("SESSION_TOKEN_ACCEPT_V1", True)


class SessionEncryption:
    """Handles AES-256 encryption/decryption of session tokens"""

    def __init__(self, secret_key: bytes):
        """Initialize with 32-byte secret key"""
        if len(secret_key) != 32:
            raise ValueError("Secret key must be exactly 32 bytes")
        self.key = secret_key
        self.backend = default_backend()

    def encrypt(self, data: dict) -> str:
        """Encrypt session data and return base64-encoded token"""
        plaintext = json.dumps(data).encode("utf-8")
        padder = sym_padding.PKCS7(128).padder()
        padded_data = padder.update(plaintext) + padder.finalize()
        iv = secrets.token_bytes(16)
        cipher = Cipher(algorithms.AES(self.key), modes.CBC(iv), backend=self.backend)
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(padded_data) + encryptor.finalize()
        combined = iv + ciphertext
        return base64.urlsafe_b64encode(combined).decode("utf-8")

    def decrypt(self, token: str) -> dict | None:
        """Decrypt base64-encoded token and return session data"""
        try:
            combined = base64.urlsafe_b64decode(token.encode("utf-8"))
            iv = combined[:16]
            ciphertext = combined[16:]
            cipher = Cipher(algorithms.AES(self.key), modes.CBC(iv), backend=self.backend)
            decryptor = cipher.decryptor()
            padded_data = decryptor.update(ciphertext) + decryptor.finalize()
            unpadder = sym_padding.PKCS7(128).unpadder()
            plaintext = unpadder.update(padded_data) + unpadder.finalize()
            return json.loads(plaintext.decode("utf-8"))
        except Exception as e:
            logger.warning("token_decryption_failed", error=str(e))
            return None


class SessionEncryptionV2:
    """AES-GCM (AEAD) session tokens with explicit v2 prefix."""

    PREFIX = "v2."

    def __init__(self, secret_key: bytes):
        if len(secret_key) != 32:
            raise ValueError("Secret key must be exactly 32 bytes")
        self.aead = AESGCM(secret_key)

    def encrypt(self, data: dict) -> str:
        plaintext = json.dumps(data).encode("utf-8")
        nonce = secrets.token_bytes(12)
        ciphertext = self.aead.encrypt(nonce, plaintext, None)
        combined = nonce + ciphertext
        token = base64.urlsafe_b64encode(combined).decode("utf-8").rstrip("=")
        return f"{self.PREFIX}{token}"

    def decrypt(self, token: str) -> dict | None:
        if not token.startswith(self.PREFIX):
            return None
        body = token[len(self.PREFIX) :]
        padding = "=" * (-len(body) % 4)
        try:
            raw = base64.urlsafe_b64decode(body + padding)
            nonce, ciphertext = raw[:12], raw[12:]
            plaintext = self.aead.decrypt(nonce, ciphertext, None)
            return json.loads(plaintext.decode("utf-8"))
        except Exception as exc:
            logger.debug("v2_token_rejected", error=str(exc))
            return None


class SessionTokenCodec:
    """Encodes/decodes session tokens with compatibility flags."""

    def __init__(self, secret_key: bytes):
        self.v1 = SessionEncryption(secret_key)
        self.enable_v2 = SESSION_TOKEN_ENABLE_V2
        self.emit_v2 = self.enable_v2 and SESSION_TOKEN_EMIT_V2
        self.accept_v1 = SESSION_TOKEN_ACCEPT_V1 or not self.enable_v2
        self.v2 = SessionEncryptionV2(secret_key) if self.enable_v2 else None
        logger.info("session_token_config", emit_v2=self.emit_v2, accept_v1=self.accept_v1)

    def encode(self, data: dict) -> str:
        if self.emit_v2 and self.v2:
            return self.v2.encrypt(data)
        return self.v1.encrypt(data)

    def decode(self, token: str) -> dict | None:
        if token.startswith(SessionEncryptionV2.PREFIX):
            if not self.enable_v2 or not self.v2:
                return None
            return self.v2.decrypt(token)
        if self.accept_v1:
            return self.v1.decrypt(token)
        return None


# Initial admin user is always created on startup if it doesn't exist
# This ensures production systems always have a working authentication


class AuthManager:
    """Handles authentication, sessions, and authorization with persistent storage"""

    def __init__(
        self,
        db_path: str = "/instance/users.db",
        secret_key: bytes | None = None,
        session_duration_hours: int = 24,
        refresh_interval_hours: int = 1,
        create_default_users: bool | None = None,
        db_encryption_key: str | None = None,
    ):
        """
        Initialize auth manager with persistent database

        Args:
            db_path: Path to SQLite database for user storage
            secret_key: 32-byte key for session encryption (generated if not provided)
            session_duration_hours: Session validity duration
            refresh_interval_hours: Token refresh interval
        """
        self.db_path = db_path
        self.session_duration = session_duration_hours * 3600
        self.refresh_interval = refresh_interval_hours * 3600
        requested_encryption = db_encryption_key is not None
        enforce_encryption = os.getenv("ENFORCE_USERS_DB_ENCRYPTION", "false").lower() in {"1", "true", "yes"}
        use_encryption = requested_encryption and SQLCIPHER_AVAILABLE
        if requested_encryption and not SQLCIPHER_AVAILABLE:
            if enforce_encryption:
                raise RuntimeError(
                    "SECURITY BLOCK: users DB encryption requested but SQLCipher driver is missing. "
                    "Install pysqlcipher3/sqlcipher3 or unset users_db_key."
                )
            logger.warning(
                "Users DB encryption disabled: SQLCipher driver missing (enforce=%s)",
                enforce_encryption,
            )
        self._db = create_encrypted_db(
            db_path=db_path,
            encryption_key=db_encryption_key,
            use_encryption=use_encryption,
        )
        if self._db.use_encryption:
            logger.info("Users DB encryption enabled (provider=SQLCipher)")
        else:
            logger.info("Users DB encryption disabled")

        # Initialize encryption
        if secret_key is None:
            # Generate and print warning - in production, load from env
            secret_key = secrets.token_bytes(32)
            logger.warning(
                "ephemeral_secret_key",
                message="Generated ephemeral secret key - set SECRET_KEY in environment for persistence",
            )
        self.token_codec = SessionTokenCodec(secret_key)

        # In-memory sessions (could be moved to Redis for distributed systems)
        self.sessions: dict[str, Session] = {}

        # Initialize database
        self._init_database()
        # Always create the initial admin user if it doesn't exist
        self._create_initial_admin()

        logger.info("auth_manager_initialized", db_path=db_path)

    def _init_database(self):
        """Create users table if it doesn't exist"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        conn = self._db.connect()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL,
                speaker_id TEXT,
                email TEXT,
                created_at REAL NOT NULL,
                modified_at REAL NOT NULL
            )
        """)
        conn.commit()
        # Close and reset the shared connection so subsequent calls reopen cleanly
        try:
            conn.close()
        finally:
            try:
                self._db.close()
            except Exception:
                pass
        try:
            Path(self.db_path).chmod(0o600)
        except Exception:
            pass
        logger.info("database_initialized", path=self.db_path)

    def _create_initial_admin(self):
        """Create initial admin user if none exists.

        This ensures production systems always have a working authentication.
        The admin password should be changed immediately after first login.

        In production, set ADMIN_PASSWORD environment variable to use a custom
        initial password instead of the default.
        """
        # Check if any admin user already exists
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM users WHERE role = ?", (UserRole.ADMIN.value,))
            admin_count = cur.fetchone()[0]
        finally:
            conn.close()

        if admin_count > 0:
            logger.info("admin_user_exists", count=admin_count)
            return

        # Create initial admin user
        now = time.time()
        # Use ADMIN_PASSWORD env var if set, otherwise default (MUST be changed!)
        admin_password = os.environ.get("ADMIN_PASSWORD", "admin123")
        if admin_password == "admin123":
            logger.warning(
                "DEFAULT_ADMIN_PASSWORD",
                message="Using default admin password - CHANGE THIS IMMEDIATELY in production!",
            )

        admin = User(
            user_id="admin",
            username="admin",
            password_hash=self._hash_password(admin_password),
            role=UserRole.ADMIN,
            speaker_id=None,  # Admin sees all speakers
            email="admin@nemoserver.local",
            created_at=now,
            modified_at=now,
        )

        self._save_user(admin)
        logger.info(
            "initial_admin_created",
            username="admin",
            message="Initial admin user created. Change password immediately!",
        )

    def _save_user(self, user: User):
        """Save or update user in database"""
        conn = self._db.connect()
        conn.execute(
            """
            INSERT OR REPLACE INTO users 
            (user_id, username, password_hash, role, speaker_id, email, created_at, modified_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                user.user_id,
                user.username,
                user.password_hash,
                user.role.value,
                user.speaker_id,
                user.email,
                user.created_at,
                user.modified_at,
            ),
        )
        conn.commit()
        conn.close()

    def create_user(
        self,
        username: str,
        password: str,
        role: UserRole = UserRole.USER,
        speaker_id: str | None = None,
        email: str | None = None,
    ) -> User:
        """Create a new user with a hashed password"""
        if self.get_user(username):
            raise ValueError(f"User '{username}' already exists")
        now = time.time()
        user = User(
            user_id=username,
            username=username,
            password_hash=self._hash_password(password),
            role=role,
            speaker_id=speaker_id,
            email=email,
            created_at=now,
            modified_at=now,
        )
        self._save_user(user)
        return user

    def _connect(self):
        conn = self._db.connect()
        conn.row_factory = sqlite3.Row
        return conn

    def get_user(self, username: str) -> User | None:
        """Load user from database"""
        conn = self._connect()
        try:
            cur = conn.cursor()
            # Case-insensitive match to avoid login failures from keyboard auto-casing.
            cur.execute("SELECT * FROM users WHERE username = ? COLLATE NOCASE", (username,))
            row = cur.fetchone()
        finally:
            conn.close()

        if not row:
            return None

        return User(
            user_id=row["user_id"],
            username=row["username"],
            password_hash=row["password_hash"],
            role=UserRole(row["role"]),
            speaker_id=row["speaker_id"],
            email=row["email"],
            created_at=row["created_at"],
            modified_at=row["modified_at"],
        )

    def get_user_by_id(self, user_id: str) -> User | None:
        """Load user by ID from database"""
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
            row = cur.fetchone()
        finally:
            conn.close()

        if not row:
            return None

        return User(
            user_id=row["user_id"],
            username=row["username"],
            password_hash=row["password_hash"],
            role=UserRole(row["role"]),
            speaker_id=row["speaker_id"],
            email=row["email"],
            created_at=row["created_at"],
            modified_at=row["modified_at"],
        )

    def list_users(self) -> list[dict]:
        """List all users (without password hashes)"""
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute("SELECT user_id, username, role, speaker_id, email, created_at, modified_at FROM users")
            rows = cur.fetchall()
        finally:
            conn.close()

        return [dict(row) for row in rows]

    def _hash_password(self, password: str) -> str:
        """Hash password using bcrypt with cost factor 12"""
        return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=12)).decode("utf-8")

    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against bcrypt hash"""
        try:
            return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
        except Exception:
            return False

    def authenticate(
        self, username: str, password: str, ip_address: str | None = None
    ) -> tuple[str | None, LockoutStatus | None]:
        """
        Authenticate user and create encrypted session.
        
        Returns:
            Tuple of (session_token, lockout_status)
            - (token, None) on success
            - (None, LockoutStatus) when locked
            - (None, None) on auth failure
        """
        # Check lockout status first
        lockout = get_lockout_manager()
        identifier = username.lower()  # Normalize for lockout tracking
        status = lockout.check_lockout(identifier)
        
        if status.is_locked:
            logger.warning("login_blocked_lockout", username=username, remaining=status.remaining_seconds)
            return None, status
        
        user = self.get_user(username)
        if not user:
            # Prevent timing attacks
            bcrypt.hashpw(b"dummy", bcrypt.gensalt())
            # Record failed attempt to prevent enumeration
            lockout.record_failed_attempt(identifier)
            return None, None

        if not self._verify_password(password, user.password_hash):
            # Record failed attempt
            failed_status = lockout.record_failed_attempt(identifier)
            logger.warning("login_failed", username=username, attempts=failed_status.failed_attempts)
            if failed_status.is_locked:
                return None, failed_status
            return None, None
        
        # Success - clear failed attempts
        lockout.record_success(identifier)

        # Create session data
        now = time.time()
        csrf_token = secrets.token_hex(32)
        session_data = {
            "user_id": user.user_id,
            "role": user.role.value,
            "speaker_id": user.speaker_id,
            "created_at": now,
            "expires_at": now + self.session_duration,
            "ip": ip_address,
            "last_refresh": now,
            "csrf_token": csrf_token,
        }

        # Encrypt session data to create token
        session_token = self.token_codec.encode(session_data)

        # Store session in memory
        session = Session(
            session_token=session_token,
            user_id=user.user_id,
            role=user.role,
            speaker_id=user.speaker_id,
            created_at=now,
            expires_at=session_data["expires_at"],
            ip_address=ip_address,
            last_refresh=now,
            csrf_token=csrf_token,
        )

        self.sessions[session_token] = session

        logger.info("user_authenticated", username=username, role=user.role.value, speaker_id=user.speaker_id)
        return session_token, None

    def validate_session(self, session_token: str) -> Session | None:
        """Validate encrypted session token and check expiration"""
        # Check in-memory cache first
        session = self.sessions.get(session_token)
        if session:
            # Check expiration
            if time.time() > session.expires_at:
                del self.sessions[session_token]
                return None
            return session

        # Decrypt and validate token
        session_data = self.token_codec.decode(session_token)
        if not session_data:
            return None

        # Check expiration
        if time.time() > session_data["expires_at"]:
            return None

        # Reconstruct session object
        session = Session(
            session_token=session_token,
            user_id=session_data["user_id"],
            role=UserRole(session_data["role"]),
            speaker_id=session_data.get("speaker_id"),
            created_at=session_data["created_at"],
            expires_at=session_data["expires_at"],
            ip_address=session_data.get("ip"),
            last_refresh=session_data.get("last_refresh", session_data["created_at"]),
            csrf_token=session_data.get("csrf_token"),
        )

        # Cache in memory
        self.sessions[session_token] = session
        return session

    def refresh_token(self, session_token: str, ip_address: str | None = None) -> str | None:
        """
        Refresh session token if needed (rotate every refresh_interval)
        Returns new token if refreshed, same token if not needed, None if invalid
        """
        session = self.validate_session(session_token)
        if not session:
            return None

        now = time.time()

        # Check if refresh is needed
        if now - session.last_refresh < self.refresh_interval:
            return session_token  # No refresh needed

        # Create new token with extended expiration
        user = self.get_user_by_id(session.user_id)
        if not user:
            return None

        # Invalidate old token
        self.logout(session_token)

        # Create new session
        return self.authenticate(user.username, user.password_hash, ip_address or session.ip_address)

    def logout(self, session_token: str) -> bool:
        """End session and invalidate token"""
        if session_token in self.sessions:
            del self.sessions[session_token]
            return True
        return False

    def change_password(self, username: str, old_password: str, new_password: str) -> bool:
        """Change user password with verification"""
        user = self.get_user(username)
        if not user:
            return False

        # Verify old password
        if not self._verify_password(old_password, user.password_hash):
            return False

        # Hash new password
        user.password_hash = self._hash_password(new_password)
        user.modified_at = time.time()

        # Save to database
        self._save_user(user)

        logger.info("password_changed", username=username)
        return True

    def check_permission(self, session_token: str, required_role: UserRole) -> bool:
        """Check if session has required role or higher"""
        session = self.validate_session(session_token)
        if not session:
            return False

        # Role hierarchy: admin > user
        role_levels = {UserRole.USER: 1, UserRole.ADMIN: 2}

        user_level = role_levels.get(session.role, 0)
        required_level = role_levels.get(required_role, 999)

        return user_level >= required_level

    def get_user_info(self, session_token: str) -> dict | None:
        """Get user info from session"""
        session = self.validate_session(session_token)
        if not session:
            return None

        user = self.get_user_by_id(session.user_id)
        if not user:
            return None

        return {
            "user_id": user.user_id,
            "username": user.username,
            "role": user.role.value,
            "speaker_id": user.speaker_id,
            "email": user.email,
        }

    def cleanup_expired_sessions(self):
        """Remove expired sessions (call periodically)"""
        now = time.time()
        expired = [token for token, session in self.sessions.items() if session.expires_at < now]
        for token in expired:
            del self.sessions[token]

        if expired:
            logger.info("expired_sessions_cleaned", count=len(expired))
        return len(expired)


# Global auth manager instance
# Secret key should be loaded from environment in production
auth_manager = None


def init_auth_manager(
    secret_key: bytes | None = None,
    db_path: str = "/instance/users.db",
    create_default_users: bool | None = None,
    db_encryption_key: str | None = None,
):
    """Initialize global auth manager"""
    global auth_manager
    auth_manager = AuthManager(
        secret_key=secret_key,
        db_path=db_path,
        create_default_users=create_default_users,
        db_encryption_key=db_encryption_key,
    )
    return auth_manager


def get_auth_manager() -> AuthManager:
    """Get global auth manager instance"""
    if auth_manager is None:
        raise RuntimeError("Auth manager not initialized. Call init_auth_manager() first.")
    return auth_manager
