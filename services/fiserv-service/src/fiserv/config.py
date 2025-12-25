"""
Fiserv Banking Hub Configuration

Credentials and settings for Banking Hub API access.
Supports multiple providers (Premier, Signature, DNA, Finxact, etc.)
"""

import os
from pathlib import Path

from pydantic_settings import BaseSettings


def _read_secret_file(env_var: str) -> str | None:
    """Read secret from Docker secrets file path specified in env var."""
    file_path = os.getenv(env_var)
    if file_path:
        path = Path(file_path)
        if path.exists():
            return path.read_text().strip()
    return None


class FiservConfig(BaseSettings):
    """Fiserv Banking Hub configuration.

    Credentials are loaded from Docker secrets (recommended) or environment variables.
    No hardcoded defaults - secrets must be provided.
    """

    # API Credentials - Load from secrets first, then env vars, no hardcoded defaults
    api_key: str = _read_secret_file("FISERV_API_KEY_FILE") or os.getenv("FISERV_API_KEY", "")
    api_secret: str = _read_secret_file("FISERV_API_SECRET_FILE") or os.getenv("FISERV_API_SECRET", "")

    # URLs
    host_url: str = os.getenv("FISERV_HOST_URL", "https://bankinghub-cert.fiservapis.com/banking/efx/v1")
    token_url: str = os.getenv("FISERV_TOKEN_URL", "https://bankinghub-cert.fiservapis.com/fts-apim/oauth2/v2")

    # Default Organization ID (DNABanking)
    default_org_id: str = os.getenv("FISERV_ORG_ID", "999950001")

    # Token settings
    token_expiry_buffer_seconds: int = 60  # Refresh 60s before expiry

    class Config:
        env_prefix = "FISERV_"


# Provider Organization IDs
PROVIDER_ORG_IDS: dict[str, str] = {
    "Premier": "999990301",
    "Signature": "999980101",
    "Finxact": "999990601",
    "Precision": "999960001",
    "Cleartouch": "9999700001",
    "Portico": "999940001",
    "Identity": "999901101",
    "DNABanking": "999950001",
    "DNACU": "999930001",
}

# Debit card logo ID for EPOC-Finxact
EPOC_DEBIT_CARD_LOGO = "97200112"


def get_config() -> FiservConfig:
    """Get Fiserv configuration singleton."""
    return FiservConfig()


def get_org_id(provider: str = "DNABanking") -> str:
    """Get organization ID for a specific provider."""
    return PROVIDER_ORG_IDS.get(provider, PROVIDER_ORG_IDS["DNABanking"])
