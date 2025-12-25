"""
Centralized Configuration Settings.

All environment variables are defined here using Pydantic Settings.
This replaces scattered os.getenv() calls throughout the codebase.

Phase 1 of API Restructure.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable loading.

    All settings have sensible defaults for development.
    Production deployments should override via environment variables.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # =========================================================================
    # Server Configuration
    # =========================================================================
    DEBUG: bool = False
    STRUCTURED_LOGGING: bool = True

    # =========================================================================
    # Service URLs (Microservices)
    # =========================================================================
    GEMMA_URL: str = "http://gemma-service:8001"
    RAG_URL: str = "http://rag-service:8004"
    EMOTION_URL: str = "http://emotion-service:8005"
    TRANSCRIPTION_URL: str = "http://transcription-service:8003"
    INSIGHTS_URL: str = "http://insights-service:8010"
    ML_SERVICE_URL: str = "http://localhost:8006"

    # =========================================================================
    # Feature Flags
    # =========================================================================
    EMAIL_ANALYZER_ENABLED: bool = True
    ANALYZE_FALLBACK_ENABLED: bool = True
    ALLOW_SELF_REGISTRATION: bool = True
    SECURE_MODE: bool = False

    # =========================================================================
    # CORS & Origins
    # =========================================================================
    ALLOWED_ORIGINS: str = "http://127.0.0.1,http://localhost"

    @property
    def allowed_origins_list(self) -> list[str]:
        """Parse ALLOWED_ORIGINS into list."""
        return [o.strip() for o in self.ALLOWED_ORIGINS.split(",") if o.strip()]

    # =========================================================================
    # File Upload
    # =========================================================================
    MAX_UPLOAD_MB: int = 100

    @property
    def max_upload_bytes(self) -> int:
        return self.MAX_UPLOAD_MB * 1024 * 1024

    # =========================================================================
    # Application Paths
    # =========================================================================
    APP_INSTANCE_DIR: str = "/app/instance"
    ANALYSIS_FALLBACK_DIR: str = ""  # Defaults to APP_INSTANCE_DIR/analysis_fallback
    ANALYSIS_FALLBACK_FILE: str = ""  # Defaults to APP_INSTANCE_DIR/analysis_fallback.json
    ANALYSIS_FALLBACK_MAX_ARTIFACTS: int = 200

    @property
    def instance_path(self) -> Path:
        return Path(self.APP_INSTANCE_DIR)

    @property
    def analysis_fallback_dir_path(self) -> Path:
        if self.ANALYSIS_FALLBACK_DIR:
            return Path(self.ANALYSIS_FALLBACK_DIR)
        return self.instance_path / "analysis_fallback"

    @property
    def analysis_fallback_file_path(self) -> Path:
        if self.ANALYSIS_FALLBACK_FILE:
            return Path(self.ANALYSIS_FALLBACK_FILE)
        return self.instance_path / "analysis_fallback.json"

    # =========================================================================
    # Canonical Host (for redirects)
    # =========================================================================
    CANONICAL_HOST: str = "localhost"
    CANONICAL_PORT: str = ""

    # =========================================================================
    # Authentication Rate Limiting
    # =========================================================================
    LOGIN_RATE_LIMIT_WINDOW: int = 60
    LOGIN_RATE_LIMIT_LIMIT: int = 5
    REGISTER_RATE_LIMIT_WINDOW: int = 60
    REGISTER_RATE_LIMIT_LIMIT: int = 5

    # =========================================================================
    # Session Keys (loaded from secrets)
    # =========================================================================
    SESSION_KEY_B64: str = ""
    SESSION_KEY: str = ""

    # =========================================================================
    # General Rate Limiting
    # =========================================================================
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_WINDOW_SEC: int = 60
    RATE_LIMIT_DEFAULT: int = 120
    RATE_LIMIT_AUTH: int = 20
    RATE_LIMIT_SKIP_PREFIXES: str = "/ui/,/ui,/assets/,/static/,/docs/"
    RATE_LIMIT_SKIP_PATHS: str = "/health,/,/api/gemma/warmup,/api/gemma/stats,/upload,/api/upload"

    @property
    def rate_limit_skip_prefixes_tuple(self) -> tuple[str, ...]:
        return tuple(filter(None, self.RATE_LIMIT_SKIP_PREFIXES.split(",")))

    @property
    def rate_limit_skip_paths_set(self) -> set[str]:
        return set(filter(None, self.RATE_LIMIT_SKIP_PATHS.split(",")))

    # =========================================================================
    # Security Headers
    # =========================================================================
    FORCE_HSTS: bool = False
    ALLOW_FRAMING: bool = False

    # =========================================================================
    # Audit Logging
    # =========================================================================
    AUDIT_ENABLED: bool = True

    # =========================================================================
    # WebSocket
    # =========================================================================
    WS_ALLOWED_ORIGINS: str = ""

    @property
    def ws_allowed_origins_set(self) -> set[str]:
        if not self.WS_ALLOWED_ORIGINS:
            return set()
        return set(filter(None, self.WS_ALLOWED_ORIGINS.split(",")))


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance.

    Use this function instead of creating Settings() directly
    to benefit from caching and ensure a single source of truth.
    """
    return Settings()


# Convenience alias for quick access
settings = get_settings()
