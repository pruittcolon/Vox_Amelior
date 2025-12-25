"""
Party Service API Wrapper

Handles customer/party operations: search, create, update.
"""

import logging
from typing import Any

from .client import get_api_client

logger = logging.getLogger(__name__)


class PartyService:
    """
    Wrapper for Fiserv Party Service APIs.

    Manages party (customer) data: names, addresses, IDs, phones, emails.
    """

    def __init__(self, provider: str = "DNABanking"):
        self.client = get_api_client(provider)

    async def search(
        self,
        name: str | None = None,
        tax_ident: str | None = None,
        phone: str | None = None,
        email: str | None = None,
        account_id: str | None = None,
        max_records: int = 20,
    ) -> dict[str, Any]:
        """
        Search for parties (customers).

        Args:
            name: Customer name to search
            tax_ident: SSN/Tax ID to search
            phone: Phone number to search
            email: Email address to search
            account_id: Associated account ID
            max_records: Maximum records to return

        Returns:
            Party list response
        """
        # Build search criteria
        party_list_sel = {}

        if name:
            party_list_sel["Name"] = name
        if tax_ident:
            party_list_sel["TaxIdent"] = tax_ident
        if phone:
            party_list_sel["Phone"] = phone
        if email:
            party_list_sel["EmailAddr"] = email
        if account_id:
            party_list_sel["AcctId"] = account_id

        payload = {"RecCtrlIn": {"MaxRecLimit": max_records}, "PartyListSel": party_list_sel}

        logger.info(f"Searching parties: {party_list_sel}")

        return await self.client.post("/party/parties/search", payload)

    async def get_by_id(self, party_id: str) -> dict[str, Any]:
        """
        Get party details by ID.

        Args:
            party_id: Party identifier

        Returns:
            Party details
        """
        payload = {"PartyKeys": {"PartyId": party_id}}

        return await self.client.post("/party/parties", payload)

    async def create(
        self,
        given_name: str,
        family_name: str,
        tax_ident: str | None = None,
        birth_date: str | None = None,
        email: str | None = None,
        phone: str | None = None,
        address: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """
        Create a new party (customer).

        Args:
            given_name: First name
            family_name: Last name
            tax_ident: SSN/Tax ID
            birth_date: Date of birth (YYYY-MM-DD)
            email: Email address
            phone: Phone number
            address: Address dict with Addr1, City, StateProv, PostalCode

        Returns:
            Created party response
        """
        person_party_info = {"PersonName": {"GivenName": given_name, "FamilyName": family_name, "NameType": "Primary"}}

        if tax_ident:
            person_party_info["TaxIdent"] = tax_ident
            person_party_info["TaxIdentType"] = "SSN"

        if birth_date:
            person_party_info["BirthDt"] = birth_date

        # Build contact info
        contact = {}
        if email:
            contact["Email"] = {"EmailAddr": email}
        if phone:
            contact["PhoneNum"] = {"Phone": phone, "PhoneType": "Home"}
        if address:
            contact["PostAddr"] = {"AddrType": "Primary", "AddrFormatType": "Label", **address}

        if contact:
            person_party_info["Contact"] = [contact]

        payload = {"PersonPartyInfo": person_party_info}

        logger.info(f"Creating party: {given_name} {family_name}")

        return await self.client.post("/party/parties", payload)

    def parse_party_list(self, response: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Parse party list response into simplified format.

        Args:
            response: Raw API response

        Returns:
            List of simplified party dicts
        """
        parties = []

        if "error" in response:
            logger.warning(f"Error in response: {response.get('message')}")
            return parties

        for rec in response.get("PartyListRec", []):
            party = {
                "party_id": rec.get("PartyKeys", {}).get("PartyId"),
            }

            # Person info
            person_info = rec.get("PersonPartyListInfo", {})
            if person_info:
                name_info = person_info.get("PersonName", {})
                party["name"] = (
                    name_info.get("FullName")
                    or f"{name_info.get('GivenName', '')} {name_info.get('FamilyName', '')}".strip()
                )
                party["type"] = "Person"
                party["tax_ident"] = person_info.get("TaxIdent")
                party["birth_date"] = person_info.get("BirthDt")

            # Org info
            org_info = rec.get("OrgPartyListInfo", {})
            if org_info:
                party["name"] = org_info.get("OrgName", {}).get("Name")
                party["type"] = "Organization"
                party["tax_ident"] = org_info.get("TaxIdent")

            # Contact info
            contacts = (person_info or org_info).get("Contact", [])
            if contacts:
                contact = contacts[0]
                if "Email" in contact:
                    party["email"] = contact["Email"].get("EmailAddr")
                if "PhoneNum" in contact:
                    party["phone"] = contact["PhoneNum"].get("Phone")
                if "PostAddr" in contact:
                    addr = contact["PostAddr"]
                    party["address"] = (
                        f"{addr.get('Addr1', '')}, {addr.get('City', '')} {addr.get('StateProv', '')} {addr.get('PostalCode', '')}"
                    )

            # Status
            status = rec.get("PartyStatus", {})
            party["status"] = status.get("PartyStatusCode")

            parties.append(party)

        return parties


def get_party_service(provider: str = "DNABanking") -> PartyService:
    """Get party service instance."""
    return PartyService(provider)
