"""
Transaction Service API Wrapper

Handles transaction operations: listing, searching by date range.
"""

import logging
from datetime import datetime, timedelta
from typing import Any

from .client import get_api_client

logger = logging.getLogger(__name__)


class TransactionService:
    """
    Wrapper for Fiserv Transaction Service APIs.

    Retrieves transaction history and balances.
    """

    def __init__(self, provider: str = "DNABanking"):
        self.client = get_api_client(provider)

    async def list_by_date_range(
        self,
        account_id: str,
        account_type: str = "DDA",
        start_date: str | None = None,
        end_date: str | None = None,
        max_records: int = 100,
    ) -> dict[str, Any]:
        """
        List transactions for date range.

        Args:
            account_id: Account identifier
            account_type: Account type (DDA, SDA, CDA, LOAN)
            start_date: Start date (YYYY-MM-DD), default 30 days ago
            end_date: End date (YYYY-MM-DD), default today
            max_records: Maximum records to return

        Returns:
            Transaction list response
        """
        # Default to last 30 days
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if not start_date:
            start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

        payload = {
            "RecCtrlIn": {"MaxRecLimit": max_records},
            "TrnListSel": {
                "AcctKeys": {"AcctId": account_id, "AcctType": account_type},
                "DateRange": {"StartDt": start_date, "EndDt": end_date},
            },
        }

        logger.info(f"Listing transactions for {account_id}: {start_date} to {end_date}")

        return await self.client.post("/tx/transactions/list", payload)

    async def get_by_id(self, transaction_id: str, account_id: str, account_type: str = "DDA") -> dict[str, Any]:
        """
        Get transaction details by ID.

        Args:
            transaction_id: Transaction identifier
            account_id: Account identifier
            account_type: Account type

        Returns:
            Transaction details
        """
        payload = {"TrnKeys": {"TrnId": transaction_id, "AcctKeys": {"AcctId": account_id, "AcctType": account_type}}}

        return await self.client.post("/tx/transactions", payload)

    def parse_transactions(self, response: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Parse transaction list into simplified format.

        Args:
            response: Raw API response

        Returns:
            List of simplified transaction dicts
        """
        transactions = []

        if "error" in response:
            logger.warning(f"Error in response: {response.get('message')}")
            return transactions

        for rec in response.get("TrnListRec", []):
            trn_keys = rec.get("TrnKeys", {})
            trn_info = rec.get("TrnInfo", {})

            txn = {
                "transaction_id": trn_keys.get("TrnId"),
                "account_id": trn_keys.get("AcctKeys", {}).get("AcctId"),
                "date": trn_info.get("PostedDt") or trn_info.get("TrnDt"),
                "amount": trn_info.get("TrnAmt", {}).get("Amt"),
                "currency": trn_info.get("TrnAmt", {}).get("CurCode", "USD"),
                "type": trn_info.get("TrnType"),
                "description": trn_info.get("Desc") or trn_info.get("TrnDesc"),
                "dr_cr": trn_info.get("DrCr"),  # Debit or Credit
                "running_balance": trn_info.get("RunBal", {}).get("Amt"),
                "category": trn_info.get("Category"),
                "merchant": trn_info.get("MerchantName"),
            }

            transactions.append(txn)

        return transactions

    def analyze_transactions(self, transactions: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Basic analysis of transaction list.

        Args:
            transactions: List of parsed transactions

        Returns:
            Analysis summary
        """
        if not transactions:
            return {"count": 0, "message": "No transactions"}

        amounts = [t.get("amount", 0) or 0 for t in transactions]
        debits = [t for t in transactions if t.get("dr_cr") == "Debit"]
        credits = [t for t in transactions if t.get("dr_cr") == "Credit"]

        analysis = {
            "count": len(transactions),
            "total_debits": sum(t.get("amount", 0) or 0 for t in debits),
            "total_credits": sum(t.get("amount", 0) or 0 for t in credits),
            "debit_count": len(debits),
            "credit_count": len(credits),
            "avg_transaction": sum(amounts) / len(amounts) if amounts else 0,
            "max_transaction": max(amounts) if amounts else 0,
            "min_transaction": min(amounts) if amounts else 0,
            "date_range": {
                "start": min(t.get("date", "") for t in transactions if t.get("date")),
                "end": max(t.get("date", "") for t in transactions if t.get("date")),
            },
        }

        return analysis


def get_transaction_service(provider: str = "DNABanking") -> TransactionService:
    """Get transaction service instance."""
    return TransactionService(provider)
