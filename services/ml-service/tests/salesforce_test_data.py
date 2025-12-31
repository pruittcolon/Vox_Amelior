"""
Salesforce Test Data Creator
Creates custom test data in Salesforce for comprehensive connector testing.
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_sources.salesforce_loader import SalesforceConfig, SalesforceLoader

logger = logging.getLogger(__name__)

# ============================================================================
# TEST DATA DEFINITIONS
# ============================================================================

TEST_ACCOUNTS = [
    {
        "Name": "Quantum AI Labs",
        "Industry": "Technology",
        "Type": "Prospect",
        "AnnualRevenue": 5000000,
        "NumberOfEmployees": 50,
        "Description": "AI research company specializing in quantum computing applications",
        "Website": "https://quantumailabs.example.com",
        "BillingCity": "San Francisco",
        "BillingState": "CA",
        "BillingCountry": "United States",
    },
    {
        "Name": "Neural Finance Corp",
        "Industry": "Financial Services",
        "Type": "Customer",
        "AnnualRevenue": 12000000,
        "NumberOfEmployees": 150,
        "Description": "Financial technology company using ML for trading algorithms",
        "Website": "https://neuralfinance.example.com",
        "BillingCity": "New York",
        "BillingState": "NY",
        "BillingCountry": "United States",
    },
    {
        "Name": "Voice Analytics Inc",
        "Industry": "Technology",
        "Type": "Customer",
        "AnnualRevenue": 3500000,
        "NumberOfEmployees": 35,
        "Description": "Speech recognition and voice analytics platform",
        "Website": "https://voiceanalytics.example.com",
        "BillingCity": "Austin",
        "BillingState": "TX",
        "BillingCountry": "United States",
    },
    {
        "Name": "Smart Retail Solutions",
        "Industry": "Retail",
        "Type": "Prospect",
        "AnnualRevenue": 8000000,
        "NumberOfEmployees": 120,
        "Description": "AI-powered retail optimization and inventory management",
        "Website": "https://smartretail.example.com",
        "BillingCity": "Seattle",
        "BillingState": "WA",
        "BillingCountry": "United States",
    },
    {
        "Name": "HealthTech Dynamics",
        "Industry": "Healthcare",
        "Type": "Prospect",
        "AnnualRevenue": 15000000,
        "NumberOfEmployees": 200,
        "Description": "Healthcare AI for diagnostics and patient care optimization",
        "Website": "https://healthtechdynamics.example.com",
        "BillingCity": "Boston",
        "BillingState": "MA",
        "BillingCountry": "United States",
    },
    {
        "Name": "AutoML Manufacturing",
        "Industry": "Manufacturing",
        "Type": "Customer",
        "AnnualRevenue": 25000000,
        "NumberOfEmployees": 500,
        "Description": "Automated machine learning for manufacturing optimization",
        "Website": "https://automlmfg.example.com",
        "BillingCity": "Detroit",
        "BillingState": "MI",
        "BillingCountry": "United States",
    },
    {
        "Name": "DataFlow Logistics",
        "Industry": "Transportation",
        "Type": "Prospect",
        "AnnualRevenue": 7000000,
        "NumberOfEmployees": 80,
        "Description": "AI-powered logistics and supply chain optimization",
        "Website": "https://dataflowlogistics.example.com",
        "BillingCity": "Chicago",
        "BillingState": "IL",
        "BillingCountry": "United States",
    },
    {
        "Name": "Cognitive Insurance Co",
        "Industry": "Insurance",
        "Type": "Customer",
        "AnnualRevenue": 45000000,
        "NumberOfEmployees": 350,
        "Description": "AI-driven insurance underwriting and claims processing",
        "Website": "https://cognitiveinsurance.example.com",
        "BillingCity": "Hartford",
        "BillingState": "CT",
        "BillingCountry": "United States",
    },
    {
        "Name": "AI-First Marketing",
        "Industry": "Media",
        "Type": "Prospect",
        "AnnualRevenue": 2000000,
        "NumberOfEmployees": 25,
        "Description": "AI-powered marketing automation and analytics",
        "Website": "https://aifirstmarketing.example.com",
        "BillingCity": "Los Angeles",
        "BillingState": "CA",
        "BillingCountry": "United States",
    },
    {
        "Name": "SecureAI Systems",
        "Industry": "Technology",
        "Type": "Customer",
        "AnnualRevenue": 10000000,
        "NumberOfEmployees": 100,
        "Description": "AI-powered cybersecurity and threat detection",
        "Website": "https://secureaisystems.example.com",
        "BillingCity": "Washington",
        "BillingState": "DC",
        "BillingCountry": "United States",
    },
]

TEST_CONTACTS = [
    {
        "FirstName": "Sarah",
        "LastName": "Chen",
        "Title": "CTO",
        "Department": "Engineering",
        "Email": "sarah.chen@quantumailabs.example.com",
        "Account": "Quantum AI Labs",
    },
    {
        "FirstName": "Marcus",
        "LastName": "Johnson",
        "Title": "VP Engineering",
        "Department": "Engineering",
        "Email": "marcus.johnson@quantumailabs.example.com",
        "Account": "Quantum AI Labs",
    },
    {
        "FirstName": "Emily",
        "LastName": "Rodriguez",
        "Title": "Chief Data Officer",
        "Department": "Data Science",
        "Email": "emily.rodriguez@neuralfinance.example.com",
        "Account": "Neural Finance Corp",
    },
    {
        "FirstName": "David",
        "LastName": "Kim",
        "Title": "Head of AI",
        "Department": "Technology",
        "Email": "david.kim@neuralfinance.example.com",
        "Account": "Neural Finance Corp",
    },
    {
        "FirstName": "Alex",
        "LastName": "Thompson",
        "Title": "CEO",
        "Department": "Executive",
        "Email": "alex.thompson@voiceanalytics.example.com",
        "Account": "Voice Analytics Inc",
    },
    {
        "FirstName": "Jennifer",
        "LastName": "Lee",
        "Title": "VP Product",
        "Department": "Product",
        "Email": "jennifer.lee@voiceanalytics.example.com",
        "Account": "Voice Analytics Inc",
    },
    {
        "FirstName": "Michael",
        "LastName": "Brown",
        "Title": "Director of IT",
        "Department": "IT",
        "Email": "michael.brown@smartretail.example.com",
        "Account": "Smart Retail Solutions",
    },
    {
        "FirstName": "Rachel",
        "LastName": "Davis",
        "Title": "Chief Medical Officer",
        "Department": "Medical",
        "Email": "rachel.davis@healthtechdynamics.example.com",
        "Account": "HealthTech Dynamics",
    },
    {
        "FirstName": "James",
        "LastName": "Wilson",
        "Title": "VP Operations",
        "Department": "Operations",
        "Email": "james.wilson@automlmfg.example.com",
        "Account": "AutoML Manufacturing",
    },
    {
        "FirstName": "Amanda",
        "LastName": "Garcia",
        "Title": "Chief Security Officer",
        "Department": "Security",
        "Email": "amanda.garcia@secureaisystems.example.com",
        "Account": "SecureAI Systems",
    },
]

TEST_OPPORTUNITIES = [
    {
        "Name": "Nemo Enterprise License - Quantum AI",
        "StageName": "Prospecting",
        "Amount": 50000,
        "CloseDate": "2025-03-15",
        "Account": "Quantum AI Labs",
    },
    {
        "Name": "AI Platform Expansion - Neural Finance",
        "StageName": "Negotiation/Review",
        "Amount": 150000,
        "CloseDate": "2025-02-01",
        "Account": "Neural Finance Corp",
    },
    {
        "Name": "Voice Transcription POC",
        "StageName": "Closed Won",
        "Amount": 25000,
        "CloseDate": "2025-01-15",
        "Account": "Voice Analytics Inc",
    },
    {
        "Name": "ML Engine Integration",
        "StageName": "Proposal/Price Quote",
        "Amount": 200000,
        "CloseDate": "2025-04-01",
        "Account": "AutoML Manufacturing",
    },
    {
        "Name": "Healthcare AI Suite",
        "StageName": "Qualification",
        "Amount": 75000,
        "CloseDate": "2025-05-01",
        "Account": "HealthTech Dynamics",
    },
    {
        "Name": "Retail Analytics Platform",
        "StageName": "Needs Analysis",
        "Amount": 45000,
        "CloseDate": "2025-03-30",
        "Account": "Smart Retail Solutions",
    },
    {
        "Name": "Logistics Optimization Engine",
        "StageName": "Prospecting",
        "Amount": 35000,
        "CloseDate": "2025-06-01",
        "Account": "DataFlow Logistics",
    },
    {
        "Name": "Insurance Claims Automation",
        "StageName": "Closed Won",
        "Amount": 250000,
        "CloseDate": "2024-12-01",
        "Account": "Cognitive Insurance Co",
    },
    {
        "Name": "Marketing AI Pilot",
        "StageName": "Qualification",
        "Amount": 15000,
        "CloseDate": "2025-02-28",
        "Account": "AI-First Marketing",
    },
    {
        "Name": "Security Threat Detection Upgrade",
        "StageName": "Negotiation/Review",
        "Amount": 100000,
        "CloseDate": "2025-01-31",
        "Account": "SecureAI Systems",
    },
]

TEST_LEADS = [
    {
        "FirstName": "Jennifer",
        "LastName": "Walsh",
        "Company": "FutureTech Innovations",
        "Title": "VP Technology",
        "Status": "Open - Not Contacted",
        "LeadSource": "Web",
        "Email": "jwalsh@futuretech.example.com",
    },
    {
        "FirstName": "Robert",
        "LastName": "Martinez",
        "Company": "CloudAI Solutions",
        "Title": "Director of Engineering",
        "Status": "Working - Contacted",
        "LeadSource": "Trade Show",
        "Email": "rmartinez@cloudai.example.com",
    },
    {
        "FirstName": "Lisa",
        "LastName": "Chang",
        "Company": "DataDriven Enterprises",
        "Title": "Chief Analytics Officer",
        "Status": "Working - Contacted",
        "LeadSource": "Partner Referral",
        "Email": "lchang@datadriven.example.com",
    },
    {
        "FirstName": "Christopher",
        "LastName": "Taylor",
        "Company": "SmartOps Inc",
        "Title": "CTO",
        "Status": "Open - Not Contacted",
        "LeadSource": "Web",
        "Email": "ctaylor@smartops.example.com",
    },
    {
        "FirstName": "Michelle",
        "LastName": "Anderson",
        "Company": "AI Ventures",
        "Title": "Managing Partner",
        "Status": "Open - Not Contacted",
        "LeadSource": "Trade Show",
        "Email": "manderson@aiventures.example.com",
    },
]

TEST_CASES = [
    {
        "Subject": "API Rate Limiting Issue",
        "Description": "Experiencing rate limits during peak usage",
        "Status": "New",
        "Priority": "High",
        "Origin": "Email",
        "Account": "Neural Finance Corp",
    },
    {
        "Subject": "Integration Documentation Request",
        "Description": "Need detailed API documentation for custom integration",
        "Status": "Closed",
        "Priority": "Medium",
        "Origin": "Web",
        "Account": "SecureAI Systems",
    },
    {
        "Subject": "Model Training Question",
        "Description": "Questions about fine-tuning ML models",
        "Status": "Working",
        "Priority": "Low",
        "Origin": "Phone",
        "Account": "Voice Analytics Inc",
    },
    {
        "Subject": "Performance Optimization",
        "Description": "Request for optimization recommendations",
        "Status": "New",
        "Priority": "Medium",
        "Origin": "Email",
        "Account": "AutoML Manufacturing",
    },
    {
        "Subject": "Data Privacy Compliance",
        "Description": "Questions about GDPR and data handling",
        "Status": "Working",
        "Priority": "High",
        "Origin": "Email",
        "Account": "HealthTech Dynamics",
    },
]


class SalesforceTestDataManager:
    """Manages test data creation and cleanup in Salesforce."""

    def __init__(self, loader: SalesforceLoader):
        self.loader = loader
        self.created_ids: dict[str, list[str]] = {
            "Account": [],
            "Contact": [],
            "Opportunity": [],
            "Lead": [],
            "Case": [],
        }
        self.account_name_to_id: dict[str, str] = {}

    async def create_record(self, object_type: str, data: dict[str, Any]) -> str:
        """Create a single record in Salesforce."""
        client = await self.loader._get_client()
        url = f"{self.loader.instance_url}/services/data/{self.loader.config.api_version}/sobjects/{object_type}"

        response = await client.post(url, json=data)

        if response.status_code in (200, 201):
            result = response.json()
            record_id = result["id"]
            self.created_ids[object_type].append(record_id)
            return record_id
        else:
            error = response.json()
            raise RuntimeError(f"Failed to create {object_type}: {error}")

    async def delete_record(self, object_type: str, record_id: str) -> bool:
        """Delete a single record from Salesforce."""
        client = await self.loader._get_client()
        url = f"{self.loader.instance_url}/services/data/{self.loader.config.api_version}/sobjects/{object_type}/{record_id}"

        response = await client.delete(url)
        return response.status_code == 204

    async def create_all_test_data(self) -> dict[str, int]:
        """Create all test data in Salesforce."""
        stats = {"Account": 0, "Contact": 0, "Opportunity": 0, "Lead": 0, "Case": 0}

        # 1. Create Accounts first
        print("📊 Creating Accounts...")
        for account in TEST_ACCOUNTS:
            try:
                account_id = await self.create_record("Account", account)
                self.account_name_to_id[account["Name"]] = account_id
                stats["Account"] += 1
                print(f"   ✅ {account['Name']}")
            except Exception as e:
                print(f"   ❌ {account['Name']}: {e}")

        # 2. Create Contacts (linked to Accounts)
        print("\n👤 Creating Contacts...")
        for contact in TEST_CONTACTS:
            try:
                account_name = contact.pop("Account")
                account_id = self.account_name_to_id.get(account_name)
                if account_id:
                    contact["AccountId"] = account_id
                contact_id = await self.create_record("Contact", contact)
                stats["Contact"] += 1
                print(f"   ✅ {contact['FirstName']} {contact['LastName']}")
            except Exception as e:
                print(f"   ❌ {contact.get('FirstName', 'Unknown')}: {e}")

        # 3. Create Opportunities (linked to Accounts)
        print("\n💰 Creating Opportunities...")
        for opp in TEST_OPPORTUNITIES:
            try:
                account_name = opp.pop("Account")
                account_id = self.account_name_to_id.get(account_name)
                if account_id:
                    opp["AccountId"] = account_id
                opp_id = await self.create_record("Opportunity", opp)
                stats["Opportunity"] += 1
                print(f"   ✅ {opp['Name']}")
            except Exception as e:
                print(f"   ❌ {opp.get('Name', 'Unknown')}: {e}")

        # 4. Create Leads
        print("\n🎯 Creating Leads...")
        for lead in TEST_LEADS:
            try:
                lead_id = await self.create_record("Lead", lead)
                stats["Lead"] += 1
                print(f"   ✅ {lead['FirstName']} {lead['LastName']}")
            except Exception as e:
                print(f"   ❌ {lead.get('FirstName', 'Unknown')}: {e}")

        # 5. Create Cases (linked to Accounts)
        print("\n📋 Creating Cases...")
        for case in TEST_CASES:
            try:
                account_name = case.pop("Account")
                account_id = self.account_name_to_id.get(account_name)
                if account_id:
                    case["AccountId"] = account_id
                case_id = await self.create_record("Case", case)
                stats["Case"] += 1
                print(f"   ✅ {case['Subject']}")
            except Exception as e:
                print(f"   ❌ {case.get('Subject', 'Unknown')}: {e}")

        return stats

    async def cleanup_test_data(self) -> dict[str, int]:
        """Delete all test data created by this manager."""
        stats = {"Account": 0, "Contact": 0, "Opportunity": 0, "Lead": 0, "Case": 0}

        # Delete in reverse order of creation (dependencies)
        for object_type in ["Case", "Opportunity", "Contact", "Lead", "Account"]:
            print(f"🗑️ Deleting {object_type}s...")
            for record_id in self.created_ids[object_type]:
                try:
                    if await self.delete_record(object_type, record_id):
                        stats[object_type] += 1
                except Exception as e:
                    print(f"   ❌ Failed to delete {record_id}: {e}")

        return stats

    async def cleanup_by_name_pattern(
        self,
        pattern: str = "Quantum AI|Neural Finance|Voice Analytics|Smart Retail|HealthTech|AutoML|DataFlow|Cognitive Insurance|AI-First|SecureAI",
    ) -> int:
        """Delete accounts matching a pattern (and cascading related records)."""
        import re

        # Find accounts matching pattern
        accounts = await self.loader.fetch_accounts(limit=100)
        deleted = 0

        for acc in accounts:
            if re.search(pattern, acc.get("Name", "")):
                try:
                    await self.delete_record("Account", acc["Id"])
                    deleted += 1
                    print(f"   🗑️ Deleted {acc['Name']}")
                except Exception as e:
                    print(f"   ❌ Failed to delete {acc['Name']}: {e}")

        return deleted


async def main():
    """Create test data in Salesforce."""
    print("=" * 60)
    print("Salesforce Test Data Creator")
    print("=" * 60)

    # Configure connection - credentials from environment
    import os
    config = SalesforceConfig(
        client_id=os.getenv("SALESFORCE_CLIENT_ID", ""),
        client_secret=os.getenv("SALESFORCE_CLIENT_SECRET", ""),
        username=os.getenv("SALESFORCE_USERNAME", ""),
        password=os.getenv("SALESFORCE_PASSWORD", ""),
        security_token=os.getenv("SALESFORCE_SECURITY_TOKEN", ""),
        domain=os.getenv("SALESFORCE_DOMAIN", "login.salesforce.com"),
    )

    loader = SalesforceLoader(config)

    print("\n🔐 Connecting to Salesforce...")
    if not await loader.connect():
        print("❌ Failed to connect")
        return False

    print(f"✅ Connected: {loader.instance_url}\n")

    manager = SalesforceTestDataManager(loader)

    # Create test data
    stats = await manager.create_all_test_data()

    print("\n" + "=" * 60)
    print("📊 Test Data Creation Summary")
    print("=" * 60)
    for obj, count in stats.items():
        print(f"   {obj}: {count} created")

    await loader.close()

    print("\n✅ Test data creation complete!")
    return True


if __name__ == "__main__":
    asyncio.run(main())
