"""
Salesforce Enterprise API Client

Handles all HTTP communication with Salesforce REST API.
Features:
- OAuth 2.0 Client Credentials authentication
- Automatic token refresh
- Retry with exponential backoff
- Rate limit handling
- Structured logging
"""

import asyncio
import logging
from typing import Any

import httpx

from .config import SALESFORCE_ENABLED, SalesforceConfig, require_config
from .errors import (
    SalesforceAuthError,
    SalesforceConnectionError,
    SalesforceError,
    SalesforceNotFoundError,
)

logger = logging.getLogger(__name__)


class SalesforceClient:
    """
    Enterprise Salesforce REST API client.

    Thread-safe, async-first client with built-in reliability patterns.

    Usage:
        client = SalesforceClient(config)
        await client.connect()
        records = await client.query("SELECT Id, Name FROM Account")
        await client.create("Account", {"Name": "New Corp"})
    """

    def __init__(self, config: SalesforceConfig):
        self.config = config
        self.access_token: str | None = None
        self.instance_url: str | None = None
        self._connected = False
        self._request_count = 0

    async def connect(self) -> bool:
        """
        Authenticate using OAuth 2.0 Client Credentials Flow.

        Returns:
            True if authentication successful.

        Raises:
            SalesforceAuthError: If authentication fails.
        """
        try:
            token_url = f"https://{self.config.domain}/services/oauth2/token"

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    token_url,
                    data={
                        "grant_type": "client_credentials",
                        "client_id": self.config.client_id,
                        "client_secret": self.config.client_secret,
                    },
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )

                if response.status_code != 200:
                    error_detail = response.text
                    logger.error(f"Salesforce auth failed: {response.status_code}")
                    raise SalesforceAuthError(f"Authentication failed: {error_detail}")

                data = response.json()
                self.access_token = data.get("access_token")
                self.instance_url = data.get("instance_url")
                self._connected = True

                logger.info(f"Salesforce connected: {self.instance_url}")
                return True

        except SalesforceAuthError:
            raise
        except Exception as e:
            logger.error(f"Salesforce connection error: {e}")
            raise SalesforceConnectionError(f"Connection error: {str(e)}", e)

    @property
    def is_connected(self) -> bool:
        return self._connected and self.access_token is not None

    async def _ensure_connected(self):
        """Ensure we have a valid connection."""
        if not self.is_connected:
            await self.connect()

    def _get_headers(self) -> dict[str, str]:
        """Get request headers with auth token."""
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
        }

    async def _request(self, method: str, url: str, **kwargs) -> httpx.Response:
        """
        Execute HTTP request with retry logic and exponential backoff.

        Handles:
        - Token expiry (401) -> auto-reconnect
        - Rate limits (429) -> wait and retry
        - Server errors (5xx) -> exponential backoff
        - Timeouts -> retry with backoff
        """
        await self._ensure_connected()

        last_error = None

        for attempt in range(self.config.max_retries):
            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.request(method, url, headers=self._get_headers(), **kwargs)

                    self._request_count += 1

                    logger.debug(f"SF API: {method} {url} -> {response.status_code} (attempt {attempt + 1})")

                    # Token expired
                    if response.status_code == 401:
                        self._connected = False
                        await self.connect()
                        continue

                    # Rate limited
                    if response.status_code == 429:
                        retry_after = int(response.headers.get("Retry-After", 60))
                        logger.warning(f"Rate limited, waiting {retry_after}s")
                        await asyncio.sleep(retry_after)
                        continue

                    # Server error
                    if response.status_code >= 500:
                        delay = self.config.retry_base_delay * (2**attempt)
                        logger.warning(f"Server error, retrying in {delay}s")
                        await asyncio.sleep(delay)
                        continue

                    return response

            except httpx.TimeoutException as e:
                last_error = e
                delay = self.config.retry_base_delay * (2**attempt)
                logger.warning(f"Timeout, retrying in {delay}s")
                await asyncio.sleep(delay)

            except Exception as e:
                last_error = e
                logger.error(f"Request error: {e}")
                raise SalesforceError(f"Request failed: {str(e)}")

        raise SalesforceError(f"Max retries exceeded: {last_error}")

    # =========================================================================
    # SOQL Query
    # =========================================================================

    async def query(self, soql: str) -> list[dict[str, Any]]:
        """
        Execute a SOQL query with automatic pagination.

        Args:
            soql: SOQL query string

        Returns:
            List of all matching records
        """
        url = f"{self.instance_url}/services/data/{self.config.api_version}/query"

        all_records = []
        params = {"q": soql}

        while True:
            response = await self._request("GET", url, params=params)

            if response.status_code != 200:
                raise SalesforceError(f"Query failed: {response.text}")

            data = response.json()
            records = data.get("records", [])
            all_records.extend(records)

            # Handle pagination
            next_url = data.get("nextRecordsUrl")
            if next_url:
                url = f"{self.instance_url}{next_url}"
                params = {}
            else:
                break

        return all_records

    # =========================================================================
    # CRUD Operations
    # =========================================================================

    async def get(self, object_name: str, record_id: str, fields: list[str] = None) -> dict[str, Any]:
        """Get a single record by ID."""
        url = f"{self.instance_url}/services/data/{self.config.api_version}/sobjects/{object_name}/{record_id}"

        params = {}
        if fields:
            params["fields"] = ",".join(fields)

        response = await self._request("GET", url, params=params if params else None)

        if response.status_code == 404:
            raise SalesforceNotFoundError(object_name, record_id)

        if response.status_code != 200:
            raise SalesforceError(f"Get failed: {response.text}")

        return response.json()

    async def create(self, object_name: str, data: dict[str, Any]) -> dict[str, Any]:
        """Create a new record."""
        url = f"{self.instance_url}/services/data/{self.config.api_version}/sobjects/{object_name}"

        response = await self._request("POST", url, json=data)

        if response.status_code not in (200, 201):
            raise SalesforceError(f"Create failed: {response.text}")

        result = response.json()
        logger.info(f"Created {object_name}: {result.get('id')}")
        return result

    async def update(self, object_name: str, record_id: str, data: dict[str, Any]) -> bool:
        """Update an existing record."""
        url = f"{self.instance_url}/services/data/{self.config.api_version}/sobjects/{object_name}/{record_id}"

        response = await self._request("PATCH", url, json=data)

        if response.status_code not in (200, 204):
            raise SalesforceError(f"Update failed: {response.text}")

        logger.info(f"Updated {object_name}/{record_id}")
        return True

    async def delete(self, object_name: str, record_id: str) -> bool:
        """Delete a record."""
        url = f"{self.instance_url}/services/data/{self.config.api_version}/sobjects/{object_name}/{record_id}"

        response = await self._request("DELETE", url)

        if response.status_code not in (200, 204):
            raise SalesforceError(f"Delete failed: {response.text}")

        logger.info(f"Deleted {object_name}/{record_id}")
        return True

    # =========================================================================
    # Composite API
    # =========================================================================

    async def composite(self, subrequests: list[dict[str, Any]], all_or_none: bool = True) -> dict[str, Any]:
        """
        Execute multiple API requests in a single call.

        Args:
            subrequests: List of subrequest objects (max 25)
            all_or_none: If True, all succeed or all fail

        Returns:
            Composite response with results for each subrequest
        """
        if len(subrequests) > 25:
            raise SalesforceError("Composite API supports max 25 subrequests")

        url = f"{self.instance_url}/services/data/{self.config.api_version}/composite"

        payload = {"allOrNone": all_or_none, "compositeRequest": subrequests}

        response = await self._request("POST", url, json=payload)

        if response.status_code != 200:
            raise SalesforceError(f"Composite request failed: {response.text}")

        return response.json()

    # =========================================================================
    # Bulk API 2.0
    # =========================================================================

    async def create_bulk_query_job(self, query: str) -> dict[str, Any]:
        """Create a Bulk API 2.0 query job."""
        url = f"{self.instance_url}/services/data/{self.config.api_version}/jobs/query"

        payload = {
            "operation": "query",
            "query": query,
            "contentType": "CSV",
            "columnDelimiter": "COMMA",
            "lineEnding": "LF",
        }

        response = await self._request("POST", url, json=payload)

        if response.status_code not in (200, 201):
            raise SalesforceError(f"Bulk query job failed: {response.text}")

        return response.json()

    async def create_bulk_ingest_job(
        self, operation: str, object_name: str, external_id_field: str = None
    ) -> dict[str, Any]:
        """Create a Bulk API 2.0 ingest job."""
        url = f"{self.instance_url}/services/data/{self.config.api_version}/jobs/ingest"

        payload = {"operation": operation, "object": object_name, "contentType": "CSV", "lineEnding": "LF"}

        if external_id_field:
            payload["externalIdFieldName"] = external_id_field

        response = await self._request("POST", url, json=payload)

        if response.status_code not in (200, 201):
            raise SalesforceError(f"Bulk ingest job failed: {response.text}")

        return response.json()

    async def get_bulk_job_status(self, job_id: str, job_type: str = "query") -> dict[str, Any]:
        """Get the status of a bulk job."""
        endpoint = "query" if job_type == "query" else "ingest"
        url = f"{self.instance_url}/services/data/{self.config.api_version}/jobs/{endpoint}/{job_id}"

        response = await self._request("GET", url)

        if response.status_code != 200:
            raise SalesforceError(f"Bulk job status failed: {response.text}")

        return response.json()

    # =========================================================================
    # Metadata
    # =========================================================================

    async def describe(self, object_name: str) -> dict[str, Any]:
        """Get metadata about a Salesforce object."""
        url = f"{self.instance_url}/services/data/{self.config.api_version}/sobjects/{object_name}/describe"

        response = await self._request("GET", url)

        if response.status_code != 200:
            raise SalesforceError(f"Describe failed: {response.text}")

        return response.json()

    async def list_objects(self) -> list[str]:
        """List all available Salesforce objects."""
        url = f"{self.instance_url}/services/data/{self.config.api_version}/sobjects"

        response = await self._request("GET", url)

        if response.status_code != 200:
            raise SalesforceError(f"List objects failed: {response.text}")

        data = response.json()
        return [obj["name"] for obj in data.get("sobjects", [])]


# =============================================================================
# Global Client Instance
# =============================================================================

_sf_client: SalesforceClient | None = None


async def get_client() -> SalesforceClient:
    """
    Get or create the global Salesforce client instance.

    Raises:
        HTTPException: If Salesforce is disabled or not configured.
    """
    from fastapi import HTTPException

    global _sf_client

    if not SALESFORCE_ENABLED:
        raise HTTPException(status_code=503, detail="Salesforce integration is disabled")

    config = require_config()

    if _sf_client is None:
        _sf_client = SalesforceClient(config)

    if not _sf_client.is_connected:
        try:
            await _sf_client.connect()
        except SalesforceAuthError as e:
            raise HTTPException(status_code=503, detail=str(e.message))

    return _sf_client


def reset_client():
    """Reset the global client (for testing)."""
    global _sf_client
    _sf_client = None
