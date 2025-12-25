"""
Feature Collection Module

Collects and engineers features from Fiserv APIs for ML models.
Maps raw Fiserv data to standardized ML feature vectors.
"""

import logging
from datetime import datetime
from typing import Any

from ..fiserv.account_service import get_account_service
from ..fiserv.party_service import get_party_service
from ..fiserv.transaction_service import get_transaction_service

logger = logging.getLogger(__name__)


class FeatureCollector:
    """
    Collects member features from Fiserv APIs for ML models.

    Features are derived from:
    - Party Service: Demographics (age, tenure)
    - Account Service: Product holdings, balances
    - Transaction Service: Behavioral patterns
    """

    def __init__(self, provider: str = "DNABanking"):
        self.provider = provider

    async def collect_member_features(
        self, member_id: str, include_transactions: bool = True, transaction_days: int = 90
    ) -> dict[str, Any]:
        """
        Collect all features for a member.

        Args:
            member_id: Party/member ID
            include_transactions: Whether to fetch transaction history
            transaction_days: Days of transaction history to analyze

        Returns:
            Feature dictionary for ML models
        """
        party_svc = get_party_service(self.provider)
        acct_svc = get_account_service(self.provider)
        tx_svc = get_transaction_service(self.provider)

        features = {
            "member_id": member_id,
            "collected_at": datetime.now().isoformat(),
        }

        try:
            # Get party info
            party_resp = await party_svc.get_by_id(member_id)
            party_features = self._extract_party_features(party_resp)
            features.update(party_features)

            # Get accounts
            accounts_resp = await acct_svc.list_by_party(member_id)
            accounts = self._parse_accounts(accounts_resp)
            account_features = self._extract_account_features(accounts)
            features.update(account_features)

            # Get transactions if requested
            if include_transactions and accounts:
                transactions = []
                for acct in accounts[:5]:  # Limit to 5 accounts for performance
                    try:
                        tx_resp = await tx_svc.list_by_date_range(
                            acct["account_id"], acct["account_type"], max_records=200
                        )
                        parsed = tx_svc.parse_transactions(tx_resp)
                        transactions.extend(parsed)
                    except Exception as e:
                        logger.warning(f"Failed to get transactions for {acct['account_id']}: {e}")

                tx_features = self._extract_transaction_features(transactions, transaction_days)
                features.update(tx_features)

            features["collection_success"] = True

        except Exception as e:
            logger.error(f"Failed to collect features for {member_id}: {e}")
            features["collection_success"] = False
            features["error"] = str(e)

        return features

    def _extract_party_features(self, party_resp: dict[str, Any]) -> dict[str, Any]:
        """Extract features from party response."""
        features = {}

        party_rec = party_resp.get("PartyRec", {})
        person_info = party_rec.get("PersonPartyInfo", {})

        # Age from birth date
        birth_date = person_info.get("BirthDt")
        if birth_date:
            try:
                bd = datetime.strptime(birth_date, "%Y-%m-%d")
                features["age"] = (datetime.now() - bd).days // 365
                features["age_bracket"] = self._get_age_bracket(features["age"])
            except (ValueError, TypeError, AttributeError):
                logger.debug("Invalid birth date format: %s", birth_date)
                features["age"] = None
                features["age_bracket"] = "unknown"
        else:
            features["age"] = None
            features["age_bracket"] = "unknown"

        # Contact info presence
        contacts = person_info.get("Contact", [])
        features["has_email"] = any("Email" in c for c in contacts)
        features["has_phone"] = any("PhoneNum" in c for c in contacts)
        features["has_address"] = any("PostAddr" in c for c in contacts)

        # Status
        status = party_rec.get("PartyStatus", {})
        features["party_status"] = status.get("PartyStatusCode", "Unknown")

        return features

    def _parse_accounts(self, accounts_resp: dict[str, Any]) -> list[dict[str, Any]]:
        """Parse accounts list response."""
        accounts = []

        for rec in accounts_resp.get("AcctListRec", []):
            acct_keys = rec.get("AcctKeys", {})
            acct_info = rec.get("AcctInfo", {})

            accounts.append(
                {
                    "account_id": acct_keys.get("AcctId"),
                    "account_type": acct_keys.get("AcctType"),
                    "product_id": acct_info.get("ProductId", ""),
                    "open_date": acct_info.get("OpenDt"),
                    "status": rec.get("AcctStatus", {}).get("AcctStatusCode"),
                    "balance": self._get_balance_amount(acct_info.get("AcctBal", [])),
                }
            )

        return accounts

    def _get_balance_amount(self, bal_list: list[dict]) -> float:
        """Extract current balance from balance list."""
        for bal in bal_list:
            if bal.get("BalType") in ("Current", "Avail", "Ledger"):
                return bal.get("CurAmt", {}).get("Amt", 0.0)
        return 0.0

    def _extract_account_features(self, accounts: list[dict[str, Any]]) -> dict[str, Any]:
        """Extract features from account list."""
        features = {}

        # Product holdings
        features["has_checking"] = any(a["account_type"] == "DDA" for a in accounts)
        features["has_savings"] = any(a["account_type"] == "SDA" for a in accounts)
        features["has_certificate"] = any(a["account_type"] == "CDA" for a in accounts)
        features["has_loan"] = any(a["account_type"] == "LOAN" for a in accounts)

        # Product ID based detection
        product_ids = [a.get("product_id", "").upper() for a in accounts]
        features["has_auto_loan"] = any("AUTO" in p for p in product_ids)
        features["has_credit_card"] = any("CARD" in p or "CC" in p for p in product_ids)
        features["has_heloc"] = any("HELOC" in p or "HEL" in p for p in product_ids)
        features["has_mortgage"] = any("MORT" in p or "MTG" in p for p in product_ids)

        # Counts
        features["product_count"] = len(accounts)
        features["deposit_count"] = sum(1 for a in accounts if a["account_type"] in ("DDA", "SDA", "CDA"))
        features["loan_count"] = sum(1 for a in accounts if a["account_type"] == "LOAN")

        # Balances
        balances = [a.get("balance", 0) or 0 for a in accounts]
        features["total_balance"] = sum(balances)
        features["avg_balance"] = sum(balances) / len(balances) if balances else 0
        features["max_balance"] = max(balances) if balances else 0

        # Tenure (from oldest account)
        open_dates = []
        for a in accounts:
            od = a.get("open_date")
            if od:
                try:
                    open_dates.append(datetime.strptime(od, "%Y-%m-%d"))
                except (ValueError, TypeError):
                    logger.debug("Invalid open date format: %s", od)

        if open_dates:
            oldest = min(open_dates)
            features["tenure_months"] = (datetime.now() - oldest).days // 30
            features["tenure_years"] = features["tenure_months"] // 12
        else:
            features["tenure_months"] = 0
            features["tenure_years"] = 0

        return features

    def _extract_transaction_features(self, transactions: list[dict[str, Any]], days: int = 90) -> dict[str, Any]:
        """Extract behavioral features from transactions."""
        features = {}

        if not transactions:
            features["transaction_count_30d"] = 0
            features["transaction_count_60d"] = 0
            features["transaction_count_90d"] = 0
            features["has_direct_deposit"] = False
            features["has_recurring_credits"] = False
            features["external_transfer_activity"] = False
            return features

        now = datetime.now()

        def within_days(tx_date: str, d: int) -> bool:
            if not tx_date:
                return False
            try:
                dt = datetime.strptime(tx_date[:10], "%Y-%m-%d")
                return (now - dt).days <= d
            except (ValueError, TypeError):
                return False

        # Time-windowed counts
        tx_30d = [t for t in transactions if within_days(t.get("date"), 30)]
        tx_60d = [t for t in transactions if within_days(t.get("date"), 60)]
        tx_90d = [t for t in transactions if within_days(t.get("date"), 90)]

        features["transaction_count_30d"] = len(tx_30d)
        features["transaction_count_60d"] = len(tx_60d)
        features["transaction_count_90d"] = len(tx_90d)

        # Credit/Debit analysis (30 day)
        credits_30d = [t for t in tx_30d if t.get("dr_cr") == "Credit"]
        debits_30d = [t for t in tx_30d if t.get("dr_cr") == "Debit"]

        features["credit_count_30d"] = len(credits_30d)
        features["debit_count_30d"] = len(debits_30d)
        features["total_credits_30d"] = sum(t.get("amount", 0) or 0 for t in credits_30d)
        features["total_debits_30d"] = sum(t.get("amount", 0) or 0 for t in debits_30d)

        # Direct deposit detection (recurring ACH credits > $500)
        large_credits = [t for t in credits_30d if (t.get("amount") or 0) > 500]
        features["has_direct_deposit"] = len(large_credits) >= 2
        features["large_credit_count_30d"] = len([t for t in credits_30d if (t.get("amount") or 0) > 1000])

        # External transfer detection
        descriptions = [t.get("description", "").upper() for t in transactions]
        features["external_transfer_activity"] = any("TRANSFER" in d or "ACH" in d or "WIRE" in d for d in descriptions)

        # Spending patterns
        if features["transaction_count_30d"] > 0:
            avg_tx = features["total_debits_30d"] / max(features["debit_count_30d"], 1)
            features["avg_transaction_amount"] = avg_tx
        else:
            features["avg_transaction_amount"] = 0

        # Month-over-month change
        tx_prior_30d = [t for t in transactions if 30 < self._days_ago(t.get("date")) <= 60]
        if len(tx_prior_30d) > 0:
            features["tx_count_change_pct"] = (len(tx_30d) / len(tx_prior_30d)) - 1
        else:
            features["tx_count_change_pct"] = 0

        # Days since last transaction
        dates = [t.get("date") for t in transactions if t.get("date")]
        if dates:
            try:
                latest = max(datetime.strptime(d[:10], "%Y-%m-%d") for d in dates)
                features["days_since_last_tx"] = (now - latest).days
            except (ValueError, TypeError):
                features["days_since_last_tx"] = 999
        else:
            features["days_since_last_tx"] = 999

        return features

    def _days_ago(self, date_str: str) -> int:
        """Calculate days ago from date string."""
        if not date_str:
            return 999
        try:
            dt = datetime.strptime(date_str[:10], "%Y-%m-%d")
            return (datetime.now() - dt).days
        except (ValueError, TypeError):
            return 999

    def _get_age_bracket(self, age: int) -> str:
        """Convert age to bracket."""
        if age < 25:
            return "18-24"
        elif age < 35:
            return "25-34"
        elif age < 45:
            return "35-44"
        elif age < 55:
            return "45-54"
        elif age < 65:
            return "55-64"
        else:
            return "65+"


# Singleton instance
_collector = None


def get_feature_collector(provider: str = "DNABanking") -> FeatureCollector:
    """Get feature collector singleton."""
    global _collector
    if _collector is None:
        _collector = FeatureCollector(provider)
    return _collector
