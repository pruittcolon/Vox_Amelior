"""
Salesforce Connector Comprehensive Test Suite
Tests all connector functionality with real Salesforce data.

SECURITY: All credentials must be provided via environment variables.
No hardcoded credentials in this file.
"""

import asyncio
import sys
import time
from pathlib import Path

import pytest

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_sources.salesforce_loader import SalesforceConfig, SalesforceLoader

# ============================================================================
# FIXTURES - Credentials from Environment
# ============================================================================


@pytest.fixture(scope="module")
def salesforce_config() -> SalesforceConfig:
    """
    Create Salesforce configuration from environment.
    Skip tests if not configured.
    """
    config = SalesforceConfig.from_env()

    if config is None:
        pytest.skip(
            "Salesforce not configured. Set SALESFORCE_CLIENT_ID, "
            "SALESFORCE_CLIENT_SECRET, and SALESFORCE_DOMAIN environment variables."
        )

    return config


@pytest.fixture(scope="module")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
async def connected_loader(salesforce_config) -> SalesforceLoader:
    """Create and connect a Salesforce loader."""
    loader = SalesforceLoader(salesforce_config)
    connected = await loader.connect()
    assert connected, "Failed to connect to Salesforce"
    yield loader
    await loader.close()


# ============================================================================
# UNIT TESTS - Connection & Authentication
# ============================================================================


class TestConnection:
    """Test connection and authentication."""

    @pytest.mark.asyncio
    async def test_config_from_env(self, salesforce_config):
        """Test configuration loading from environment."""
        assert salesforce_config.client_id is not None
        assert len(salesforce_config.client_id) > 10
        assert salesforce_config.domain is not None

    @pytest.mark.asyncio
    async def test_connect_success(self, salesforce_config):
        """Test successful connection."""
        loader = SalesforceLoader(salesforce_config)
        result = await loader.connect()

        assert result is True
        assert loader.access_token is not None
        assert loader.instance_url is not None
        assert "salesforce.com" in loader.instance_url

        await loader.close()

    @pytest.mark.asyncio
    async def test_connect_invalid_credentials(self):
        """Test connection with invalid credentials fails gracefully."""
        bad_config = SalesforceConfig(
            client_id="invalid_client_id", client_secret="invalid_secret", domain="test.salesforce.com"
        )
        loader = SalesforceLoader(bad_config)
        result = await loader.connect()

        assert result is False
        assert loader.access_token is None


# ============================================================================
# UNIT TESTS - Data Fetching
# ============================================================================


class TestDataFetching:
    """Test data fetching methods."""

    @pytest.mark.asyncio
    async def test_fetch_accounts(self, connected_loader):
        """Test fetching Account records."""
        accounts = await connected_loader.fetch_accounts(limit=10)

        assert isinstance(accounts, list)
        assert len(accounts) > 0

        # Verify account structure
        account = accounts[0]
        assert "Id" in account
        assert "Name" in account
        assert account["Id"].startswith("001")  # Account IDs start with 001

    @pytest.mark.asyncio
    async def test_fetch_contacts(self, connected_loader):
        """Test fetching Contact records."""
        contacts = await connected_loader.fetch_contacts(limit=10)

        assert isinstance(contacts, list)
        assert len(contacts) > 0

        contact = contacts[0]
        assert "Id" in contact
        assert "LastName" in contact
        assert contact["Id"].startswith("003")

    @pytest.mark.asyncio
    async def test_fetch_opportunities(self, connected_loader):
        """Test fetching Opportunity records."""
        opportunities = await connected_loader.fetch_opportunities(limit=10)

        assert isinstance(opportunities, list)
        assert len(opportunities) > 0

        opp = opportunities[0]
        assert "Id" in opp
        assert "Name" in opp
        assert "StageName" in opp

    @pytest.mark.asyncio
    async def test_fetch_leads(self, connected_loader):
        """Test fetching Lead records."""
        leads = await connected_loader.fetch_leads(limit=10)
        assert isinstance(leads, list)

    @pytest.mark.asyncio
    async def test_fetch_cases(self, connected_loader):
        """Test fetching Case records."""
        cases = await connected_loader.fetch_cases(limit=10)
        assert isinstance(cases, list)


# ============================================================================
# UNIT TESTS - SOQL Queries
# ============================================================================


class TestSOQLQueries:
    """Test custom SOQL query execution."""

    @pytest.mark.asyncio
    async def test_simple_query(self, connected_loader):
        """Test simple SOQL query."""
        query = "SELECT Id, Name FROM Account LIMIT 5"
        results = await connected_loader.query(query)

        assert isinstance(results, list)
        assert len(results) <= 5

    @pytest.mark.asyncio
    async def test_query_with_where_clause(self, connected_loader):
        """Test SOQL query with WHERE clause."""
        query = "SELECT Id, Name, Industry FROM Account WHERE Industry != null LIMIT 5"
        results = await connected_loader.query(query)

        assert isinstance(results, list)
        for result in results:
            assert result.get("Industry") is not None

    @pytest.mark.asyncio
    async def test_query_with_order_by(self, connected_loader):
        """Test SOQL query with ORDER BY."""
        query = "SELECT Id, Name, CreatedDate FROM Account ORDER BY CreatedDate DESC LIMIT 5"
        results = await connected_loader.query(query)

        assert isinstance(results, list)
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_query_aggregate(self, connected_loader):
        """Test aggregate SOQL query."""
        query = "SELECT COUNT(Id) total FROM Account"
        results = await connected_loader.query(query)

        assert isinstance(results, list)
        assert len(results) == 1
        assert "total" in results[0]


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================


class TestPerformance:
    """Test connector performance."""

    @pytest.mark.asyncio
    async def test_connection_speed(self, salesforce_config):
        """Test connection time is under 5 seconds."""
        loader = SalesforceLoader(salesforce_config)

        start = time.time()
        await loader.connect()
        elapsed = time.time() - start

        await loader.close()

        assert elapsed < 5.0, f"Connection took {elapsed:.2f}s, expected < 5s"

    @pytest.mark.asyncio
    async def test_query_speed(self, connected_loader):
        """Test query time for 100 records is under 5 seconds."""
        start = time.time()
        accounts = await connected_loader.fetch_accounts(limit=100)
        elapsed = time.time() - start

        assert elapsed < 5.0, f"Query took {elapsed:.2f}s, expected < 5s"


# ============================================================================
# SCHEMA TESTS
# ============================================================================


class TestSchema:
    """Test schema and metadata operations."""

    @pytest.mark.asyncio
    async def test_get_schema(self, connected_loader):
        """Test schema method returns valid structure."""
        schema = connected_loader.get_schema()

        assert "name" in schema
        assert schema["name"] == "Salesforce"
        assert "objects" in schema
        assert "Account" in schema["objects"]
        assert schema["supports_crud"] is True

    @pytest.mark.asyncio
    async def test_list_objects(self, connected_loader):
        """Test listing available objects."""
        objects = await connected_loader.list_objects()

        assert isinstance(objects, list)
        assert len(objects) > 100
        assert "Account" in objects
        assert "Contact" in objects
        assert "Opportunity" in objects


# ============================================================================
# RUN ALL TESTS
# ============================================================================


async def run_all_tests():
    """Run all tests and print summary."""
    print("=" * 70)
    print("Salesforce Connector Comprehensive Test Suite")
    print("=" * 70)

    # Check configuration
    config = SalesforceConfig.from_env()

    if config is None:
        print("\n❌ Salesforce not configured!")
        print("\nRequired environment variables:")
        print("   - SALESFORCE_CLIENT_ID")
        print("   - SALESFORCE_CLIENT_SECRET")
        print("   - SALESFORCE_DOMAIN")
        return {"passed": 0, "failed": 1, "errors": [("Configuration", "Not configured")]}

    loader = SalesforceLoader(config)
    results = {"passed": 0, "failed": 0, "errors": []}

    def test_result(name, success, error=None):
        if success:
            results["passed"] += 1
            print(f"   ✅ {name}")
        else:
            results["failed"] += 1
            results["errors"].append((name, error))
            print(f"   ❌ {name}: {error}")

    # Connection Tests
    print("\n📡 Connection Tests")
    print("-" * 50)

    try:
        connected = await loader.connect()
        test_result("Connect to Salesforce", connected)
    except Exception as e:
        test_result("Connect to Salesforce", False, str(e))
        print("\n⚠️ Cannot continue tests without connection")
        return results

    # Data Fetching Tests
    print("\n📊 Data Fetching Tests")
    print("-" * 50)

    for name, fetch_fn in [
        ("Accounts", loader.fetch_accounts),
        ("Contacts", loader.fetch_contacts),
        ("Opportunities", loader.fetch_opportunities),
        ("Leads", loader.fetch_leads),
        ("Cases", loader.fetch_cases),
    ]:
        try:
            records = await fetch_fn(limit=5)
            test_result(f"Fetch {name} ({len(records)} records)", True)
        except Exception as e:
            test_result(f"Fetch {name}", False, str(e))

    # Query Tests
    print("\n🔍 SOQL Query Tests")
    print("-" * 50)

    try:
        result = await loader.query("SELECT Id, Name FROM Account LIMIT 3")
        test_result("Simple SOQL Query", len(result) > 0)
    except Exception as e:
        test_result("Simple SOQL Query", False, str(e))

    try:
        result = await loader.query("SELECT COUNT(Id) total FROM Account")
        test_result(f"Aggregate Query (total: {result[0]['total']})", True)
    except Exception as e:
        test_result("Aggregate Query", False, str(e))

    # Schema Tests
    print("\n📋 Schema Tests")
    print("-" * 50)

    try:
        schema = loader.get_schema()
        test_result("Get connector schema", "Account" in schema["objects"])
    except Exception as e:
        test_result("Get connector schema", False, str(e))

    try:
        objects = await loader.list_objects()
        test_result(f"List objects ({len(objects)} found)", len(objects) > 50)
    except Exception as e:
        test_result("List objects", False, str(e))

    # Cleanup
    await loader.close()

    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    print(f"   Passed: {results['passed']}")
    print(f"   Failed: {results['failed']}")
    print(f"   Total:  {results['passed'] + results['failed']}")

    if results["errors"]:
        print("\n❌ Failed Tests:")
        for name, error in results["errors"]:
            print(f"   • {name}: {error}")

    return results


if __name__ == "__main__":
    results = asyncio.run(run_all_tests())
    exit(0 if results["failed"] == 0 else 1)
