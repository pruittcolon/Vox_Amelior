"""
Fiserv OAuth2 Client

Handles authentication with Banking Hub OAuth2 endpoint.
Implements automatic token refresh.
"""

import base64
import logging
import time
from dataclasses import dataclass

import httpx

from .config import get_config
from .tracker import get_tracker

logger = logging.getLogger(__name__)


@dataclass
class TokenInfo:
    """OAuth2 token information."""

    access_token: str
    token_type: str
    expires_at: float  # Unix timestamp

    @property
    def is_expired(self) -> bool:
        """Check if token is expired (with buffer)."""
        config = get_config()
        return time.time() >= (self.expires_at - config.token_expiry_buffer_seconds)


class FiservAuthClient:
    """
    OAuth2 client for Fiserv Banking Hub.

    Generates Bearer tokens using client credentials flow.
    Tokens expire after ~15 minutes.
    """

    def __init__(self):
        self.config = get_config()
        self._token: TokenInfo | None = None
        self._http_client: httpx.AsyncClient | None = None

    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=30.0)
        return self._http_client

    def _get_basic_auth_header(self) -> str:
        """Generate Basic auth header from API key and secret."""
        credentials = f"{self.config.api_key}:{self.config.api_secret}"
        encoded = base64.b64encode(credentials.encode()).decode()
        return f"Basic {encoded}"

    async def get_token(self, force_refresh: bool = False) -> str:
        """
        Get valid access token, refreshing if needed.

        Args:
            force_refresh: Force token refresh even if not expired

        Returns:
            Valid access token string
        """
        if not force_refresh and self._token and not self._token.is_expired:
            return self._token.access_token

        await self._refresh_token()
        return self._token.access_token

    async def _refresh_token(self) -> None:
        """Request new token from OAuth2 endpoint."""
        client = await self._get_http_client()

        headers = {
            "Authorization": self._get_basic_auth_header(),
            "Content-Type": "application/x-www-form-urlencoded",
        }

        data = {"grant_type": "client_credentials"}

        # Track this API call
        tracker = get_tracker()
        if not tracker.can_make_call():
            raise Exception("API call limit (1000) reached! Cannot refresh token.")

        logger.info(f"Requesting OAuth2 token... (API calls: {tracker.get_stats()['total_calls']}/1000)")

        try:
            response = await client.post(self.config.token_url, headers=headers, data=data)
            response.raise_for_status()

            token_data = response.json()

            # Calculate expiry timestamp
            # expires_in is in milliseconds per Fiserv docs
            expires_in_ms = int(token_data.get("expires_in", 899000))
            expires_in_seconds = expires_in_ms / 1000
            expires_at = time.time() + expires_in_seconds

            self._token = TokenInfo(
                access_token=token_data["access_token"],
                token_type=token_data.get("token_type", "Bearer"),
                expires_at=expires_at,
            )

            # Record successful API call
            tracker.record_call("token")

            logger.info(
                f"Token obtained, expires in {expires_in_seconds:.0f}s "
                f"(at {time.strftime('%H:%M:%S', time.localtime(expires_at))})"
            )

        except httpx.HTTPStatusError as e:
            logger.error(f"Token request failed: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Token request error: {e}")
            raise

    async def get_auth_headers(self, org_id: str | None = None) -> dict:
        """
        Get complete headers for API request.

        Args:
            org_id: Override organization ID (defaults to config)

        Returns:
            Dict with Authorization and EFXHeader
        """
        token = await self.get_token()

        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "EFXHeader": f'{{"OrganizationId": "{org_id or self.config.default_org_id}"}}',
        }

    @property
    def token_status(self) -> dict:
        """Get current token status for debugging."""
        if not self._token:
            return {"status": "no_token"}

        return {
            "status": "valid" if not self._token.is_expired else "expired",
            "token_type": self._token.token_type,
            "expires_at": self._token.expires_at,
            "expires_in_seconds": max(0, self._token.expires_at - time.time()),
            "token_preview": f"{self._token.access_token[:20]}..." if self._token.access_token else None,
        }

    async def close(self) -> None:
        """Close HTTP client."""
        if self._http_client:
            await self._http_client.aclose()


# Singleton instance
_auth_client: FiservAuthClient | None = None


def get_auth_client() -> FiservAuthClient:
    """Get auth client singleton."""
    global _auth_client
    if _auth_client is None:
        _auth_client = FiservAuthClient()
    return _auth_client
