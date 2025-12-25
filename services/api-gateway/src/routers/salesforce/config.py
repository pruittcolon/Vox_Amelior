"""
Salesforce Configuration Module

Handles all configuration loading from environment variables.
No hardcoded credentials permitted - security first.
"""

import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class AuthFlow(str, Enum):
    """Supported OAuth flows."""

    CLIENT_CREDENTIALS = "client_credentials"
    JWT_BEARER = "jwt_bearer"


@dataclass
class SalesforceConfig:
    """
    Salesforce connection configuration from environment.

    Required Environment Variables:
        SALESFORCE_CLIENT_ID: OAuth app client ID
        SALESFORCE_CLIENT_SECRET: OAuth app client secret
        SALESFORCE_DOMAIN: Your Salesforce My Domain (e.g., mycompany.my.salesforce.com)

    Optional Environment Variables:
        SALESFORCE_API_VERSION: API version (default: v59.0)
        SALESFORCE_MAX_RETRIES: Max retry attempts (default: 3)
        SALESFORCE_RETRY_DELAY: Base delay between retries in seconds (default: 1.0)
        SALESFORCE_ENABLED: Enable/disable integration (default: true)
    """

    client_id: str
    client_secret: str
    domain: str
    api_version: str = "v59.0"
    auth_flow: AuthFlow = AuthFlow.CLIENT_CREDENTIALS
    max_retries: int = 3
    retry_base_delay: float = 1.0

    @staticmethod
    def _read_secret(name: str) -> Optional[str]:
        """
        Read a secret from Docker /run/secrets/ or environment variable.
        Docker secrets take precedence over env vars for security.
        """
        # Try Docker secrets first (mounted at /run/secrets/)
        secret_path = f"/run/secrets/{name}"
        if os.path.exists(secret_path):
            try:
                with open(secret_path, "r") as f:
                    return f.read().strip()
            except Exception as e:
                logger.warning(f"Failed to read secret {name}: {e}")
        
        # Fallback to environment variable (uppercase with SALESFORCE_ prefix)
        env_name = f"SALESFORCE_{name.upper()}"
        return os.getenv(env_name)

    @classmethod
    def from_env(cls) -> Optional["SalesforceConfig"]:
        """
        Load configuration from Docker secrets or environment variables.
        Docker secrets take precedence for security.
        Returns None if required variables are missing.
        """
        client_id = cls._read_secret("salesforce_client_id") or os.getenv("SALESFORCE_CLIENT_ID")
        client_secret = cls._read_secret("salesforce_client_secret") or os.getenv("SALESFORCE_CLIENT_SECRET")
        domain = cls._read_secret("salesforce_domain") or os.getenv("SALESFORCE_DOMAIN")

        # All three are required - no fallbacks for security
        if not all([client_id, client_secret, domain]):
            logger.warning("Salesforce credentials not fully configured")
            return None
        
        logger.info(f"Salesforce configured for domain: {domain}")

        return cls(
            client_id=client_id,
            client_secret=client_secret,
            domain=domain,
            api_version=os.getenv("SALESFORCE_API_VERSION", "v59.0"),
            max_retries=int(os.getenv("SALESFORCE_MAX_RETRIES", "3")),
            retry_base_delay=float(os.getenv("SALESFORCE_RETRY_DELAY", "1.0")),
        )

    def __repr__(self) -> str:
        """Safe repr that doesn't expose credentials."""
        return f"SalesforceConfig(domain={self.domain!r}, api_version={self.api_version!r})"


# Global configuration - lazy loaded
_sf_config: SalesforceConfig | None = None
SALESFORCE_ENABLED = os.getenv("SALESFORCE_ENABLED", "true").lower() == "true"


def get_config() -> SalesforceConfig | None:
    """
    Get Salesforce configuration (lazy loaded).
    Returns None if not configured.
    """
    global _sf_config
    if _sf_config is None:
        _sf_config = SalesforceConfig.from_env()
    return _sf_config


def require_config() -> SalesforceConfig:
    """
    Get Salesforce configuration, raising error if not configured.
    Use in endpoints that require Salesforce to be configured.
    """
    from fastapi import HTTPException

    config = get_config()
    if config is None:
        raise HTTPException(
            status_code=503,
            detail="Salesforce not configured. Set SALESFORCE_CLIENT_ID, "
            "SALESFORCE_CLIENT_SECRET, and SALESFORCE_DOMAIN environment variables.",
        )
    return config
