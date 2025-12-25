"""
Fiserv API Client

Base client for making authenticated requests to Banking Hub APIs.
"""

import json
import logging
from typing import Any

import httpx

from .auth import get_auth_client
from .config import get_config, get_org_id
from .tracker import get_tracker

logger = logging.getLogger(__name__)


class FiservAPIClient:
    """
    Base API client for Fiserv Banking Hub.

    Handles request construction with proper headers and error handling.
    """

    def __init__(self, provider: str = "DNABanking"):
        """
        Initialize API client.

        Args:
            provider: Banking provider name (DNABanking, Premier, etc.)
        """
        self.config = get_config()
        self.auth = get_auth_client()
        self.org_id = get_org_id(provider)
        self.provider = provider
        self._http_client: httpx.AsyncClient | None = None

    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=30.0)
        return self._http_client

    async def _build_headers(self) -> dict[str, str]:
        """Build request headers with auth and EFXHeader."""
        token = await self.auth.get_token()

        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "EFXHeader": json.dumps({"OrganizationId": self.org_id}),
        }

    async def post(self, endpoint: str, payload: dict[str, Any], org_id_override: str | None = None) -> dict[str, Any]:
        """
        Make POST request to Banking Hub API.

        Args:
            endpoint: API endpoint path (e.g., "/party/parties/search")
            payload: Request body as dict
            org_id_override: Optional org ID override

        Returns:
            Response JSON as dict
        """
        client = await self._get_http_client()
        headers = await self._build_headers()

        # Override org ID if specified
        if org_id_override:
            headers["EFXHeader"] = json.dumps({"OrganizationId": org_id_override})

        url = f"{self.config.host_url}{endpoint}"

        logger.info(f"POST {url}")
        logger.debug(f"Payload: {json.dumps(payload, indent=2)}")

        # Check API limit
        tracker = get_tracker()
        if not tracker.can_make_call():
            return {"error": True, "message": "API limit (1000) reached!"}

        try:
            response = await client.post(url, headers=headers, json=payload)

            # Record the call (detect type from endpoint)
            call_type = self._detect_call_type(endpoint)
            tracker.record_call(call_type)

            # Log response status
            logger.info(f"Response: {response.status_code}")

            # Handle error responses
            if response.status_code >= 400:
                error_body = response.text
                logger.error(f"API error: {response.status_code} - {error_body}")
                return {"error": True, "status_code": response.status_code, "message": error_body}

            return response.json()

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error: {e}")
            return {"error": True, "message": str(e)}
        except Exception as e:
            logger.error(f"Request error: {e}")
            return {"error": True, "message": str(e)}

    async def put(self, endpoint: str, payload: dict[str, Any], org_id_override: str | None = None) -> dict[str, Any]:
        """
        Make PUT request to Banking Hub API.

        Note: Fiserv uses PUT for some operations that are semantically updates.
        """
        client = await self._get_http_client()
        headers = await self._build_headers()

        if org_id_override:
            headers["EFXHeader"] = json.dumps({"OrganizationId": org_id_override})

        url = f"{self.config.host_url}{endpoint}"

        logger.info(f"PUT {url}")

        try:
            response = await client.put(url, headers=headers, json=payload)

            if response.status_code >= 400:
                return {"error": True, "status_code": response.status_code, "message": response.text}

            return response.json()

        except Exception as e:
            logger.error(f"Request error: {e}")
            return {"error": True, "message": str(e)}

    def _detect_call_type(self, endpoint: str) -> str:
        """Detect call type from endpoint path."""
        if "party" in endpoint.lower():
            return "party"
        elif "acct" in endpoint.lower() or "account" in endpoint.lower():
            return "account"
        elif "tx" in endpoint.lower() or "transaction" in endpoint.lower():
            return "transaction"
        return "other"

    async def close(self) -> None:
        """Close HTTP client."""
        if self._http_client:
            await self._http_client.aclose()


# Singleton clients per provider
_clients: dict[str, FiservAPIClient] = {}


def get_api_client(provider: str = "DNABanking") -> FiservAPIClient:
    """Get API client singleton for a provider."""
    if provider not in _clients:
        _clients[provider] = FiservAPIClient(provider)
    return _clients[provider]
