"""
Fail-Closed Startup Security Checks

This module provides security checks that block application startup if critical
security requirements are not met. Following 2024 best practices:
- Never hardcode secrets
- Block startup without required secrets
- Validate secret strength (length, not weak defaults)
- Block unsafe development flags in production (SECURE_MODE)

Usage:
    from shared.security.startup_checks import assert_strong_secret, assert_secure_mode

    # At startup, before any routes are registered:
    session_key = assert_strong_secret("session_key", min_bytes=32)
    jwt_secret = assert_strong_secret("jwt_secret_primary", min_bytes=32)
    assert_secure_mode()
"""

import base64
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Known weak default values that should never be used in production
WEAK_DEFAULTS = frozenset(
    {
        "dev_secret",
        "changeme",
        "password",
        "secret",
        "test",
        "admin",
        "default",
        "development",
        "12345",
        "123456",
        "qwerty",
        "letmein",
        "password123",
    }
)


def assert_strong_secret(
    name: str,
    min_bytes: int = 32,
    allow_env_fallback: bool = True,
) -> bytes:
    """
    Ensure a secret exists and meets minimum entropy requirements.

    Checks in order:
    1. Docker secret at /run/secrets/{name}
    2. Environment variable {NAME} (uppercase)

    Args:
        name: Name of the secret (e.g., "session_key", "jwt_secret_primary")
        min_bytes: Minimum required length in bytes (default: 32 for AES-256)
        allow_env_fallback: Whether to check environment variable if Docker secret missing

    Returns:
        The decoded secret as bytes

    Raises:
        RuntimeError: If secret is missing, weak, or too short
    """
    value: str | None = None
    source: str = "unknown"

    # Priority 1: Docker secret
    secret_path = Path(f"/run/secrets/{name}")
    if secret_path.exists():
        try:
            value = secret_path.read_text().strip()
            source = f"docker_secret:{secret_path}"
        except Exception as e:
            logger.warning("Failed to read Docker secret %s: %s", secret_path, e)

    # Priority 2: Environment variable (uppercase)
    if not value and allow_env_fallback:
        env_name = name.upper()
        value = os.environ.get(env_name)
        if value:
            source = f"env:{env_name}"

    # Fail if not found
    if not value:
        raise RuntimeError(
            f"SECURITY BLOCK: Required secret '{name}' not found. "
            f"Expected at /run/secrets/{name} or as {name.upper()} env var. "
            "Run: ./scripts/setup_secrets.sh to generate secrets."
        )

    # Check for weak defaults
    if value.lower() in WEAK_DEFAULTS:
        raise RuntimeError(
            f"SECURITY BLOCK: Secret '{name}' uses a known weak default value. "
            "Generate a strong random secret using: "
            'python3 -c "import secrets, base64; print(base64.urlsafe_b64encode(secrets.token_bytes(32)).decode())"'
        )

    # Decode and validate length
    key_bytes: bytes
    try:
        # Try URL-safe base64 first (what our scripts generate)
        padded = value + "=" * (-len(value) % 4)
        key_bytes = base64.urlsafe_b64decode(padded)
    except Exception:
        try:
            # Try standard base64
            key_bytes = base64.b64decode(value + "==")
        except Exception:
            # Fall back to raw UTF-8 bytes
            key_bytes = value.encode("utf-8")

    if len(key_bytes) < min_bytes:
        raise RuntimeError(
            f"SECURITY BLOCK: Secret '{name}' must be at least {min_bytes} bytes "
            f"(got {len(key_bytes)} bytes from {source}). "
            "Generate a stronger secret."
        )

    logger.info("✅ Loaded secret '%s' from %s (%d bytes)", name, source, len(key_bytes))
    return key_bytes


def assert_secure_mode() -> None:
    """
    Block unsafe development flags when SECURE_MODE=true.

    This function should be called at startup in production environments.
    When SECURE_MODE is enabled, the following flags must NOT be set:
    - ENABLE_DEMO_USERS: Creates accounts with weak/known credentials
    - TEST_MODE: Disables authentication checks
    - ANALYZE_FALLBACK: Allows unauthenticated analysis

    Raises:
        RuntimeError: If SECURE_MODE=true but unsafe flags are enabled
    """
    secure_mode = os.environ.get("SECURE_MODE", "").lower() in {"1", "true", "yes"}

    if not secure_mode:
        logger.debug("SECURE_MODE not enabled; skipping unsafe flag checks")
        return

    # Map of unsafe flags to descriptions
    unsafe_flags = {
        "ENABLE_DEMO_USERS": "Creates accounts with weak/known credentials",
        "TEST_MODE": "Disables authentication checks",
        "ANALYZE_FALLBACK": "Allows unauthenticated analysis",
        "ALLOW_FRAMING": "Disables clickjacking protection (X-Frame-Options)",
    }

    violations = []
    for flag, description in unsafe_flags.items():
        if os.environ.get(flag, "").lower() in {"1", "true", "yes"}:
            violations.append(f"  - {flag}: {description}")

    if violations:
        raise RuntimeError(
            "SECURITY BLOCK: SECURE_MODE=true but unsafe development flags are enabled:\n"
            + "\n".join(violations)
            + "\n\nDisable these flags for production deployment."
        )

    logger.info("✅ SECURE_MODE enabled; all unsafe flags are disabled")


def validate_secret_permissions(path: Path) -> bool:
    """
    Validate that a secret file has appropriate permissions (600 or more restrictive).

    Args:
        path: Path to the secret file

    Returns:
        True if permissions are acceptable, False otherwise
    """
    if not path.exists():
        return False

    try:
        mode = path.stat().st_mode & 0o777
        if mode > 0o600:
            logger.warning("Secret file %s has overly permissive mode %o (should be 600)", path, mode)
            return False
        return True
    except Exception as e:
        logger.warning("Failed to check permissions for %s: %s", path, e)
        return False
