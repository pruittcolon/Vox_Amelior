"""
Salesforce Enterprise Connector for Nemo Server
Enterprise CRM Data Integration

Authenticates via OAuth 2.0 Client Credentials Flow and provides
CRUD operations for Accounts, Contacts, Leads, Opportunities, and Cases.

SECURITY: All credentials must be provided via environment variables.
No hardcoded credentials are permitted.
"""

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)


class SalesforceError(Exception):
    """Base Salesforce error."""

    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class SalesforceAuthError(SalesforceError):
    """Authentication failed."""

    pass


class SalesforceRateLimitError(SalesforceError):
    """Rate limit exceeded."""

    pass


@dataclass
class SalesforceConfig:
    """
    Salesforce connection configuration.

    All values must be provided via environment variables.
    No default/fallback credentials are permitted for security.
    """

    client_id: str
    client_secret: str
    domain: str
    api_version: str = "v59.0"
    max_retries: int = 3
    retry_base_delay: float = 1.0

    @classmethod
    def from_env(cls) -> Optional["SalesforceConfig"]:
        """
        Load configuration from environment variables.
        Returns None if required variables are missing.

        Required environment variables:
        - SALESFORCE_CLIENT_ID
        - SALESFORCE_CLIENT_SECRET
        - SALESFORCE_DOMAIN

        Optional:
        - SALESFORCE_API_VERSION (default: v59.0)
        - SALESFORCE_MAX_RETRIES (default: 3)
        - SALESFORCE_RETRY_DELAY (default: 1.0)
        """
        client_id = os.getenv("SALESFORCE_CLIENT_ID")
        client_secret = os.getenv("SALESFORCE_CLIENT_SECRET")
        domain = os.getenv("SALESFORCE_DOMAIN")

        # All three are required - no fallbacks for security
        if not all([client_id, client_secret, domain]):
            logger.warning(
                "Salesforce configuration incomplete. "
                "Set SALESFORCE_CLIENT_ID, SALESFORCE_CLIENT_SECRET, and SALESFORCE_DOMAIN."
            )
            return None

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
        return (
            f"SalesforceConfig(domain={self.domain!r}, "
            f"api_version={self.api_version!r}, "
            f"client_id={self.client_id[:10]}...)"
        )


class SalesforceLoader:
    """
    Enterprise Salesforce data loader using REST API.

    Features:
    - OAuth 2.0 Client Credentials flow (production-ready)
    - Automatic token refresh on 401
    - Retry with exponential backoff
    - Pagination support for large queries
    - CRUD operations

    Example:
        loader = SalesforceLoader()
        if await loader.connect():
            accounts = await loader.fetch_accounts()
            await loader.create("Account", {"Name": "New Corp"})
            await loader.close()
    """

    def __init__(self, config: SalesforceConfig | None = None):
        self.config = config or SalesforceConfig.from_env()
        self.access_token: str | None = None
        self.instance_url: str | None = None
        self._client: httpx.AsyncClient | None = None
        self._request_count = 0

    @property
    def is_configured(self) -> bool:
        """Check if loader has valid configuration."""
        return self.config is not None

    async def connect(self) -> bool:
        """
        Authenticate with Salesforce using Client Credentials OAuth flow.

        Returns:
            True if authentication successful, False otherwise.
        """
        if not self.is_configured:
            logger.error("Cannot connect: Salesforce not configured")
            return False

        token_url = f"https://{self.config.domain}/services/oauth2/token"

        payload = {
            "grant_type": "client_credentials",
            "client_id": self.config.client_id,
            "client_secret": self.config.client_secret,
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(token_url, data=payload)

                if response.status_code == 200:
                    data = response.json()
                    self.access_token = data["access_token"]
                    self.instance_url = data["instance_url"]
                    logger.info(f"✅ Salesforce connected: {self.instance_url}")
                    return True
                else:
                    error = response.json() if response.text else {"error": "Unknown"}
                    logger.error(f"❌ Salesforce auth failed: {error}")
                    return False

        except Exception as e:
            logger.error(f"❌ Salesforce connection error: {e}")
            return False

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client with auth headers."""
        if not self.access_token:
            raise RuntimeError("Not authenticated. Call connect() first.")

        if self._client is None:
            self._client = httpx.AsyncClient(
                headers={
                    "Authorization": f"Bearer {self.access_token}",
                    "Content-Type": "application/json",
                },
                timeout=60.0,
            )
        return self._client

    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _request_with_retry(self, method: str, url: str, **kwargs) -> httpx.Response:
        """Execute HTTP request with retry logic."""
        client = await self._get_client()
        last_error = None

        for attempt in range(self.config.max_retries):
            try:
                response = await client.request(method, url, **kwargs)
                self._request_count += 1

                if response.status_code == 401:
                    # Token expired, reconnect
                    await self.close()
                    await self.connect()
                    client = await self._get_client()
                    continue

                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", 60))
                    logger.warning(f"Rate limited, waiting {retry_after}s")
                    await asyncio.sleep(retry_after)
                    continue

                if response.status_code >= 500:
                    delay = self.config.retry_base_delay * (2**attempt)
                    logger.warning(f"Server error, retrying in {delay}s")
                    await asyncio.sleep(delay)
                    continue

                return response

            except httpx.TimeoutException as e:
                last_error = e
                delay = self.config.retry_base_delay * (2**attempt)
                await asyncio.sleep(delay)

        raise SalesforceError(f"Max retries exceeded: {last_error}")

    async def query(self, soql: str) -> list[dict[str, Any]]:
        """
        Execute a SOQL query and return all results with pagination.

        Args:
            soql: Salesforce Object Query Language query string

        Returns:
            List of records matching the query
        """
        if not self.instance_url:
            raise RuntimeError("Not connected. Call connect() first.")

        url = f"{self.instance_url}/services/data/{self.config.api_version}/query"

        all_records = []
        params = {"q": soql}

        while True:
            response = await self._request_with_retry("GET", url, params=params)

            if response.status_code != 200:
                error = response.text
                logger.error(f"Query failed: {error}")
                raise SalesforceError(f"SOQL query failed: {error}")

            data = response.json()
            records = data.get("records", [])
            all_records.extend(records)

            next_url = data.get("nextRecordsUrl")
            if next_url:
                url = f"{self.instance_url}{next_url}"
                params = {}
            else:
                break

        logger.info(f"Fetched {len(all_records)} records")
        return all_records

    # =========================================================================
    # CRUD Operations
    # =========================================================================

    async def create(self, object_name: str, data: dict[str, Any]) -> dict[str, Any]:
        """Create a new record."""
        url = f"{self.instance_url}/services/data/{self.config.api_version}/sobjects/{object_name}"
        response = await self._request_with_retry("POST", url, json=data)

        if response.status_code not in (200, 201):
            raise SalesforceError(f"Create failed: {response.text}")

        result = response.json()
        logger.info(f"Created {object_name}: {result.get('id')}")
        return result

    async def update(self, object_name: str, record_id: str, data: dict[str, Any]) -> bool:
        """Update an existing record."""
        url = f"{self.instance_url}/services/data/{self.config.api_version}/sobjects/{object_name}/{record_id}"
        response = await self._request_with_retry("PATCH", url, json=data)

        if response.status_code not in (200, 204):
            raise SalesforceError(f"Update failed: {response.text}")

        logger.info(f"Updated {object_name}/{record_id}")
        return True

    async def delete(self, object_name: str, record_id: str) -> bool:
        """Delete a record."""
        url = f"{self.instance_url}/services/data/{self.config.api_version}/sobjects/{object_name}/{record_id}"
        response = await self._request_with_retry("DELETE", url)

        if response.status_code not in (200, 204):
            raise SalesforceError(f"Delete failed: {response.text}")

        logger.info(f"Deleted {object_name}/{record_id}")
        return True

    async def get(self, object_name: str, record_id: str) -> dict[str, Any]:
        """Get a single record by ID."""
        url = f"{self.instance_url}/services/data/{self.config.api_version}/sobjects/{object_name}/{record_id}"
        response = await self._request_with_retry("GET", url)

        if response.status_code == 404:
            raise SalesforceError(f"{object_name} with ID {record_id} not found", 404)

        if response.status_code != 200:
            raise SalesforceError(f"Get failed: {response.text}")

        return response.json()

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    async def fetch_accounts(self, limit: int = 100) -> list[dict[str, Any]]:
        """Fetch Account records."""
        soql = f"""
            SELECT Id, Name, Industry, Type, Phone, Website, 
                   BillingCity, BillingState, BillingCountry,
                   NumberOfEmployees, AnnualRevenue, CreatedDate
            FROM Account
            LIMIT {limit}
        """
        return await self.query(soql)

    async def fetch_contacts(self, limit: int = 100) -> list[dict[str, Any]]:
        """Fetch Contact records."""
        soql = f"""
            SELECT Id, FirstName, LastName, Email, Phone, Title,
                   AccountId, Department, MailingCity, MailingState,
                   CreatedDate
            FROM Contact
            LIMIT {limit}
        """
        return await self.query(soql)

    async def fetch_leads(self, limit: int = 100) -> list[dict[str, Any]]:
        """Fetch Lead records."""
        soql = f"""
            SELECT Id, FirstName, LastName, Email, Phone, Company,
                   Title, Status, LeadSource, Industry, CreatedDate
            FROM Lead
            LIMIT {limit}
        """
        return await self.query(soql)

    async def fetch_opportunities(self, limit: int = 100) -> list[dict[str, Any]]:
        """Fetch Opportunity records."""
        soql = f"""
            SELECT Id, Name, StageName, Amount, CloseDate,
                   AccountId, Type, LeadSource, Probability, CreatedDate
            FROM Opportunity
            LIMIT {limit}
        """
        return await self.query(soql)

    async def fetch_cases(self, limit: int = 100) -> list[dict[str, Any]]:
        """Fetch Case records."""
        soql = f"""
            SELECT Id, CaseNumber, Subject, Status, Priority,
                   Origin, Type, AccountId, ContactId, CreatedDate
            FROM Case
            LIMIT {limit}
        """
        return await self.query(soql)

    async def describe_object(self, object_name: str) -> dict[str, Any]:
        """Get metadata about a Salesforce object."""
        if not self.instance_url:
            raise RuntimeError("Not connected. Call connect() first.")

        url = f"{self.instance_url}/services/data/{self.config.api_version}/sobjects/{object_name}/describe"
        response = await self._request_with_retry("GET", url)

        if response.status_code != 200:
            raise SalesforceError(f"Describe failed: {response.text}")

        return response.json()

    async def list_objects(self) -> list[str]:
        """List all available Salesforce objects."""
        if not self.instance_url:
            raise RuntimeError("Not connected. Call connect() first.")

        url = f"{self.instance_url}/services/data/{self.config.api_version}/sobjects"
        response = await self._request_with_retry("GET", url)

        if response.status_code != 200:
            raise SalesforceError(f"List objects failed: {response.text}")

        data = response.json()
        return [obj["name"] for obj in data.get("sobjects", [])]

    def get_schema(self) -> dict[str, Any]:
        """Return the connector schema."""
        return {
            "name": "Salesforce",
            "objects": ["Account", "Contact", "Lead", "Opportunity", "Case"],
            "auth_type": "oauth2_client_credentials",
            "supports_pagination": True,
            "supports_custom_query": True,
            "supports_crud": True,
        }


# =============================================================================
# CLI Test Function
# =============================================================================


async def test_connection():
    """Test Salesforce connection with current environment config."""
    loader = SalesforceLoader()

    if not loader.is_configured:
        print("❌ Salesforce not configured. Set environment variables:")
        print("   - SALESFORCE_CLIENT_ID")
        print("   - SALESFORCE_CLIENT_SECRET")
        print("   - SALESFORCE_DOMAIN")
        return False

    print("🔐 Connecting to Salesforce...")
    print(f"   Domain: {loader.config.domain}")
    print(f"   Client ID: {loader.config.client_id[:20]}...")

    if await loader.connect():
        print(f"✅ Connected to: {loader.instance_url}")

        print("\n📊 Fetching sample data...")

        accounts = await loader.fetch_accounts(limit=5)
        print(f"   Accounts: {len(accounts)}")
        for acc in accounts[:3]:
            print(f"      - {acc.get('Name')}")

        contacts = await loader.fetch_contacts(limit=5)
        print(f"   Contacts: {len(contacts)}")

        leads = await loader.fetch_leads(limit=5)
        print(f"   Leads: {len(leads)}")

        await loader.close()
        print("\n✅ Salesforce connector test PASSED")
        return True
    else:
        print("❌ Connection failed")
        return False


if __name__ == "__main__":
    asyncio.run(test_connection())
