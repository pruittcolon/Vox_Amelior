"""
Account Service API Wrapper

Handles account operations: inquiry, listing, creation.
"""

import logging
from typing import Any

from .client import get_api_client

logger = logging.getLogger(__name__)


class AccountService:
    """
    Wrapper for Fiserv Account Service APIs.

    Manages deposit, savings, and loan accounts.
    """

    def __init__(self, provider: str = "DNABanking"):
        self.client = get_api_client(provider)

    async def get_by_id(self, account_id: str, account_type: str = "DDA") -> dict[str, Any]:
        """
        Get account details by ID.

        Args:
            account_id: Account identifier
            account_type: Account type (DDA, SDA, CDA, LOAN, etc.)

        Returns:
            Account details response
        """
        payload = {"AcctKeys": {"AcctId": account_id, "AcctType": account_type}}

        logger.info(f"Getting account: {account_id} ({account_type})")

        return await self.client.post("/acct/accounts", payload)

    async def get_balance(self, account_id: str, account_type: str = "DDA") -> dict[str, Any]:
        """
        Get account balance.

        Args:
            account_id: Account identifier
            account_type: Account type

        Returns:
            Balance information
        """
        payload = {"AcctKeys": {"AcctId": account_id, "AcctType": account_type}}

        return await self.client.post("/acct/accounts/balance", payload)

    async def list_by_party(
        self, party_id: str, account_type: str | None = None, max_records: int = 50
    ) -> dict[str, Any]:
        """
        List accounts for a party (customer).

        Args:
            party_id: Party identifier
            account_type: Optional filter by account type
            max_records: Maximum records to return

        Returns:
            Account list response
        """
        acct_list_sel = {"PartyKeys": {"PartyId": party_id}}

        if account_type:
            acct_list_sel["AcctType"] = account_type

        payload = {"RecCtrlIn": {"MaxRecLimit": max_records}, "AcctListSel": acct_list_sel}

        logger.info(f"Listing accounts for party: {party_id}")

        return await self.client.post("/acct/accts/list", payload)

    async def create(
        self, party_id: str, account_type: str = "DDA", product_id: str | None = None, nickname: str | None = None
    ) -> dict[str, Any]:
        """
        Create a new account.

        Args:
            party_id: Party (customer) ID
            account_type: Account type (DDA, SDA, CDA, LOAN)
            product_id: Product identifier
            nickname: Account nickname

        Returns:
            Created account response
        """
        acct_info = {"AcctType": account_type, "OwnerParty": {"PartyId": party_id}}

        if product_id:
            acct_info["ProductId"] = product_id
        if nickname:
            acct_info["Nickname"] = nickname

        payload = {"AcctInfo": acct_info}

        logger.info(f"Creating {account_type} account for party: {party_id}")

        return await self.client.post("/acct/accounts", payload)

    def parse_account_info(self, response: dict[str, Any]) -> dict[str, Any]:
        """
        Parse account response into simplified format.

        Args:
            response: Raw API response

        Returns:
            Simplified account dict
        """
        if "error" in response:
            return {"error": response.get("message")}

        acct_rec = response.get("AcctRec", {})
        acct_keys = acct_rec.get("AcctKeys", {})
        acct_info = acct_rec.get("AcctInfo", {})
        acct_status = acct_rec.get("AcctStatus", {})

        return {
            "account_id": acct_keys.get("AcctId"),
            "account_type": acct_keys.get("AcctType"),
            "product_id": acct_info.get("ProductId"),
            "nickname": acct_info.get("Nickname"),
            "status": acct_status.get("AcctStatusCode"),
            "open_date": acct_info.get("OpenDt"),
            "balance": self._extract_balance(acct_info),
        }

    def _extract_balance(self, acct_info: dict[str, Any]) -> dict[str, Any] | None:
        """Extract balance info from account info."""
        balance_info = acct_info.get("AcctBal", [])
        if not balance_info:
            return None

        balances = {}
        for bal in balance_info:
            bal_type = bal.get("BalType", "Unknown")
            balances[bal_type] = {
                "amount": bal.get("CurAmt", {}).get("Amt"),
                "currency": bal.get("CurAmt", {}).get("CurCode", "USD"),
            }

        return balances


def get_account_service(provider: str = "DNABanking") -> AccountService:
    """Get account service instance."""
    return AccountService(provider)
