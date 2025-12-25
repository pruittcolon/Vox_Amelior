"""
Shared Constants Module - Documented Magic Numbers and Defaults.

This module eliminates magic numbers throughout the codebase by providing
named, documented constants. All values include units in comments and
are organized by functional area.

Following NIST 2024 guidelines for security-related values.
"""

# =============================================================================
# Size Limits (Bytes)
# =============================================================================

# File upload limits
MAX_UPLOAD_BYTES: int = 100 * 1024 * 1024
"""Maximum file upload size: 100 MB (megabytes)."""

MAX_AUDIO_SIZE_BYTES: int = 50 * 1024 * 1024
"""Maximum audio file size for transcription: 50 MB."""

# Text limits
MAX_PROMPT_CHARS: int = 6_000
"""Maximum prompt length for Gemma: 6,000 characters."""

MAX_CONTEXT_CHARS: int = 12_000
"""Maximum context window for analysis: 12,000 characters."""

MAX_TRANSCRIPT_LENGTH: int = 50_000
"""Maximum transcript length: 50,000 characters."""

# =============================================================================
# Time Constants (Seconds)
# =============================================================================

# Session management
SESSION_MAX_AGE_SECONDS: int = 86_400
"""Session cookie max-age: 24 hours (86,400 seconds)."""

SESSION_DURATION_SECONDS: int = 86_400
"""Session duration: 24 hours (86,400 seconds)."""

# Security headers
HSTS_MAX_AGE_SECONDS: int = 31_536_000
"""Strict-Transport-Security max-age: 1 year (31,536,000 seconds)."""

# Rate limiting
DEFAULT_RATE_LIMIT_WINDOW_SECONDS: int = 60
"""Rate limit sliding window: 60 seconds."""

DEFAULT_RATE_LIMIT_REQUESTS: int = 120
"""Default rate limit: 120 requests per window."""

AUTH_RATE_LIMIT_REQUESTS: int = 20
"""Authentication endpoint rate limit: 20 requests per window."""

LOGIN_RATE_LIMIT_WINDOW_SECONDS: int = 60
"""Login rate limit window: 60 seconds."""

LOGIN_RATE_LIMIT_MAX_ATTEMPTS: int = 5
"""Maximum login attempts per window: 5."""

REGISTER_RATE_LIMIT_WINDOW_SECONDS: int = 60
"""Registration rate limit window: 60 seconds."""

REGISTER_RATE_LIMIT_MAX_ATTEMPTS: int = 5
"""Maximum registration attempts per window: 5."""

# Timeouts
DEFAULT_TIMEOUT_SECONDS: int = 30
"""Default HTTP request timeout: 30 seconds."""

CACHE_MAX_AGE_SECONDS: int = 3_600
"""Static asset cache max-age: 1 hour (3,600 seconds)."""

# =============================================================================
# Audio Processing
# =============================================================================

AUDIO_SAMPLE_RATE_HZ: int = 16_000
"""Target audio sample rate: 16,000 Hz (16 kHz)."""

# =============================================================================
# WebSocket Codes
# =============================================================================

WS_CLOSE_NORMAL: int = 1000
"""Normal WebSocket closure."""

WS_CLOSE_GOING_AWAY: int = 1001
"""WebSocket endpoint going away."""

WS_CLOSE_PROTOCOL_ERROR: int = 1002
"""WebSocket protocol error."""

WS_CLOSE_INTERNAL_ERROR: int = 1011
"""Internal server error."""

WS_CLOSE_AUTH_REQUIRED: int = 4001
"""Custom: Authentication required."""

WS_CLOSE_INVALID_SESSION: int = 4002
"""Custom: Invalid session token."""

WS_CLOSE_ORIGIN_NOT_ALLOWED: int = 4003
"""Custom: Origin not allowed."""

# =============================================================================
# Password Policy (NIST 2024)
# =============================================================================

PASSWORD_MIN_LENGTH: int = 8
"""Minimum password length: 8 characters (NIST SP 800-63B)."""

PASSWORD_MAX_LENGTH: int = 128
"""Maximum password length: 128 characters."""

# =============================================================================
# Analysis Limits
# =============================================================================

ANALYSIS_FALLBACK_MAX_ARTIFACTS: int = 200
"""Maximum artifacts to store per user in fallback storage."""

# =============================================================================
# Content Truncation Limits
# =============================================================================

PROMPT_HEAD_CHARS: int = 2_000
"""First N characters to keep when truncating prompt: 2,000."""

SUMMARY_CONTEXT_MAX_CHARS: int = 12_000
"""Maximum summary context: 12,000 characters."""

CONVERSATION_SAMPLE_CHARS: int = 6_000
"""Conversation sample limit: 6,000 characters."""

# =============================================================================
# Network Ports (Default Docker)
# =============================================================================

DEFAULT_API_GATEWAY_PORT: int = 8_000
"""API Gateway default port: 8000."""

DEFAULT_GEMMA_PORT: int = 8_001
"""Gemma service default port: 8001."""

DEFAULT_GPU_COORDINATOR_PORT: int = 8_002
"""GPU Coordinator default port: 8002."""

DEFAULT_TRANSCRIPTION_PORT: int = 8_003
"""Transcription service default port: 8003."""

DEFAULT_RAG_PORT: int = 8_004
"""RAG service default port: 8004."""

DEFAULT_EMOTION_PORT: int = 8_005
"""Emotion service default port: 8005."""

DEFAULT_ML_PORT: int = 8_006
"""ML service default port: 8006."""

DEFAULT_INSIGHTS_PORT: int = 8_010
"""Insights service default port: 8010."""

DEFAULT_FISERV_PORT: int = 8_015
"""Fiserv service default port: 8015."""
