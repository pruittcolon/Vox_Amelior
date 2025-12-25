"""
Input Validation Utilities - Secure Input Processing.

This module provides Pydantic validators and sanitization functions
for common input types. All user input should be validated using
these utilities before processing.

Phase 14 of ultimateseniordevplan.md.
"""

import html
import re

# ============================================================
# Regex Patterns for Validation
# ============================================================

# Email: RFC 5322 simplified
EMAIL_PATTERN = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")

# Username: alphanumeric, underscore, hyphen, 3-64 chars
USERNAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{3,64}$")

# UUID: standard format
UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)

# Speaker ID: lowercase alphanumeric, underscore, hyphen
SPEAKER_ID_PATTERN = re.compile(r"^[a-z0-9_-]{1,64}$")

# File path: prevent path traversal
SAFE_PATH_PATTERN = re.compile(r"^[a-zA-Z0-9_\-./]+$")

# URL slug
SLUG_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


# ============================================================
# Sanitization Functions
# ============================================================


def sanitize_html(text: str) -> str:
    """Escape HTML entities to prevent XSS.

    Args:
        text: User-provided text that may contain HTML.

    Returns:
        str: Text with HTML entities escaped.

    Example:
        >>> sanitize_html("<script>alert('xss')</script>")
        '&lt;script&gt;alert(&#x27;xss&#x27;)&lt;/script&gt;'
    """
    return html.escape(text)


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal.

    Removes path separators and dangerous characters.

    Args:
        filename: User-provided filename.

    Returns:
        str: Safe filename.

    Example:
        >>> sanitize_filename("../../../etc/passwd")
        'etcpasswd'
    """
    # Remove path components
    filename = filename.replace("/", "").replace("\\", "")
    filename = filename.replace("..", "")

    # Keep only safe characters
    safe_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
    return "".join(c for c in filename if c in safe_chars)


def truncate_string(text: str, max_length: int) -> str:
    """Truncate string to maximum length.

    Args:
        text: Input text.
        max_length: Maximum allowed length.

    Returns:
        str: Truncated text.
    """
    if len(text) <= max_length:
        return text
    return text[:max_length]


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text.

    Collapses multiple spaces, removes leading/trailing whitespace.

    Args:
        text: Input text.

    Returns:
        str: Normalized text.
    """
    return " ".join(text.split())


# ============================================================
# Pydantic Validators (reusable)
# ============================================================


def validate_email(email: str) -> str:
    """Validate email format.

    Args:
        email: Email address to validate.

    Returns:
        str: Validated and normalized email.

    Raises:
        ValueError: If email format is invalid.
    """
    email = email.strip().lower()
    if not EMAIL_PATTERN.match(email):
        raise ValueError("Invalid email format")
    return email


def validate_username(username: str) -> str:
    """Validate username format.

    Args:
        username: Username to validate.

    Returns:
        str: Validated username.

    Raises:
        ValueError: If username format is invalid.
    """
    username = username.strip()
    if not USERNAME_PATTERN.match(username):
        raise ValueError("Username must be 3-64 characters, alphanumeric, underscore, or hyphen")
    return username


def validate_uuid(value: str) -> str:
    """Validate UUID format.

    Args:
        value: UUID string to validate.

    Returns:
        str: Validated UUID.

    Raises:
        ValueError: If UUID format is invalid.
    """
    value = value.strip().lower()
    if not UUID_PATTERN.match(value):
        raise ValueError("Invalid UUID format")
    return value


def validate_speaker_id(speaker_id: str) -> str:
    """Validate speaker identifier.

    Args:
        speaker_id: Speaker ID to validate.

    Returns:
        str: Validated speaker ID.

    Raises:
        ValueError: If speaker ID format is invalid.
    """
    speaker_id = speaker_id.strip().lower()
    if not SPEAKER_ID_PATTERN.match(speaker_id):
        raise ValueError("Speaker ID must be 1-64 lowercase alphanumeric characters, underscore, or hyphen")
    return speaker_id


def validate_safe_path(path: str) -> str:
    """Validate path is safe (no traversal).

    Args:
        path: File path to validate.

    Returns:
        str: Validated path.

    Raises:
        ValueError: If path contains traversal or dangerous characters.
    """
    if ".." in path or path.startswith("/"):
        raise ValueError("Path traversal not allowed")
    if not SAFE_PATH_PATTERN.match(path):
        raise ValueError("Path contains invalid characters")
    return path


# ============================================================
# Password Validation (NIST SP 800-63B)
# ============================================================


def validate_password_strength(password: str) -> str:
    """Validate password meets NIST 2024 guidelines.

    NIST SP 800-63B prioritizes length over complexity:
    - Minimum 8 characters
    - Maximum 128 characters
    - No composition rules (uppercase, numbers, etc.)
    - Check against common passwords (implementation dependent)

    Args:
        password: Password to validate.

    Returns:
        str: Validated password.

    Raises:
        ValueError: If password doesn't meet requirements.
    """
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters")
    if len(password) > 128:
        raise ValueError("Password must be at most 128 characters")

    # Check for common weak passwords
    weak_passwords = {
        "password",
        "12345678",
        "qwerty123",
        "password123",
        "admin123",
        "letmein",
        "welcome1",
        "monkey123",
    }
    if password.lower() in weak_passwords:
        raise ValueError("Password is too common")

    return password


# ============================================================
# Input Length Limits
# ============================================================

MAX_INPUT_LENGTHS = {
    "username": 64,
    "email": 254,
    "password": 128,
    "message": 10_000,
    "prompt": 6_000,
    "filename": 255,
    "path": 4096,
    "url": 2048,
    "speaker_id": 64,
    "title": 200,
    "description": 2000,
}


def get_max_length(field_name: str, default: int = 1000) -> int:
    """Get maximum length for a field.

    Args:
        field_name: Name of field.
        default: Default max length if not defined.

    Returns:
        int: Maximum allowed length.
    """
    return MAX_INPUT_LENGTHS.get(field_name, default)
