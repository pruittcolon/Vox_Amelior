#!/usr/bin/env python3
"""
Salesforce Connector Test Script
Tests connection and data fetching from Salesforce.

SECURITY: Credentials must be provided via environment variables.
No credentials are stored in this file.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Load environment from .env.salesforce if exists
env_file = Path(__file__).parent.parent / ".env.salesforce"
if env_file.exists():
    print(f"📄 Loading config from {env_file}")
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ[key] = value
else:
    print("ℹ️  Using environment variables (no .env.salesforce found)")


async def main():
    from data_sources.salesforce_loader import SalesforceConfig, SalesforceLoader

    print("=" * 60)
    print("Salesforce Connector Test")
    print("=" * 60)

    # Check configuration
    config = SalesforceConfig.from_env()

    if config is None:
        print("\n❌ Salesforce not configured!")
        print("\nRequired environment variables:")
        print("   - SALESFORCE_CLIENT_ID")
        print("   - SALESFORCE_CLIENT_SECRET")
        print("   - SALESFORCE_DOMAIN")
        print("\nSet these via environment or create .env.salesforce file.")
        return False

    # Show config (masked)
    print("\n📋 Configuration:")
    print(f"   Domain: {config.domain}")
    print(
        f"   Client ID: {config.client_id[:20]}..."
        if len(config.client_id) > 20
        else f"   Client ID: {config.client_id}"
    )
    print(f"   API Version: {config.api_version}")

    # Test connection
    print("\n🔐 Testing connection...")
    loader = SalesforceLoader(config)

    if not await loader.connect():
        print("❌ Connection FAILED")
        return False

    print(f"✅ Connected to: {loader.instance_url}")

    # Test data fetching
    print("\n" + "=" * 60)
    print("📊 Fetching Sample Data")
    print("=" * 60)

    try:
        # Accounts
        print("\n🏢 Accounts:")
        accounts = await loader.fetch_accounts(limit=10)
        print(f"   Found: {len(accounts)} accounts")
        for acc in accounts[:5]:
            name = acc.get("Name", "N/A")
            industry = acc.get("Industry", "N/A")
            print(f"   • {name} ({industry})")

        # Contacts
        print("\n👤 Contacts:")
        contacts = await loader.fetch_contacts(limit=10)
        print(f"   Found: {len(contacts)} contacts")
        for contact in contacts[:5]:
            first = contact.get("FirstName", "")
            last = contact.get("LastName", "")
            email = contact.get("Email", "N/A")
            print(f"   • {first} {last} - {email}")

        # Opportunities
        print("\n💰 Opportunities:")
        opportunities = await loader.fetch_opportunities(limit=10)
        print(f"   Found: {len(opportunities)} opportunities")
        for opp in opportunities[:5]:
            name = opp.get("Name", "N/A")
            stage = opp.get("StageName", "N/A")
            amount = opp.get("Amount", 0) or 0
            print(f"   • {name} - {stage} (${amount:,.0f})")

        # Leads
        print("\n🎯 Leads:")
        leads = await loader.fetch_leads(limit=10)
        print(f"   Found: {len(leads)} leads")

        # Cases
        print("\n📋 Cases:")
        cases = await loader.fetch_cases(limit=10)
        print(f"   Found: {len(cases)} cases")

        # Custom SOQL query
        print("\n🔍 Custom SOQL Query:")
        query = "SELECT Id, Name, CreatedDate FROM Account ORDER BY CreatedDate DESC LIMIT 3"
        print(f"   Query: {query}")
        results = await loader.query(query)
        print(f"   Results: {len(results)}")
        for r in results:
            created = r.get("CreatedDate", "N/A")
            print(f"   • {r.get('Name')} (Created: {created[:10] if created else 'N/A'})")

    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        await loader.close()
        return False

    await loader.close()

    # Summary
    print("\n" + "=" * 60)
    print("✅ Salesforce Connector Test PASSED")
    print("=" * 60)
    print("\nData Summary:")
    print(f"   Accounts:      {len(accounts)}")
    print(f"   Contacts:      {len(contacts)}")
    print(f"   Opportunities: {len(opportunities)}")
    print(f"   Leads:         {len(leads)}")
    print(f"   Cases:         {len(cases)}")

    return True


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
