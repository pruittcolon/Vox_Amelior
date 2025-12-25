from typing import Any


class Normalizer:
    """
    Normalizes raw Fiserv data (or Mock data) into analysis-ready Pandas DataFrames.
    Maps canonical fields to the generic schemas expected by ML engines.
    """

    @staticmethod
    def get_member_transactions_view(dataset: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Converts the canonical 'transactions' list into a flat list of dicts
        suitable for 'member_transactions_view' dataframe.
        """
        raw_txs = dataset.get("transactions", [])
        member_id = dataset.get("member", {}).get("member_id", "unknown")

        normalized = []
        for tx in raw_txs:
            # Determine signed amount
            amt = float(tx.get("amount", 0.0))
            if tx.get("dr_cr") == "DR":
                amt_signed = -abs(amt)
            else:
                amt_signed = abs(amt)

            row = {
                "date": tx.get("date"),
                "amount": amt,  # Absolute amount (often used by spend_pattern)
                "amount_signed": amt_signed,  # Signed amount (used by cash_flow)
                "direction": tx.get("dr_cr"),
                "merchant": tx.get("description", "Unknown"),  # Use desc as proxy for merchant
                "category": tx.get("category", "Uncategorized"),
                "description": tx.get("description"),
                "account_id": tx.get("account_id"),
                "member_id": member_id,
                "transaction_id": tx.get("tx_id"),
            }
            normalized.append(row)

        return normalized

    @staticmethod
    def get_member_features_view(dataset: dict[str, Any]) -> dict[str, Any]:
        """
        Aggregates data into a single-row feature vector for the member.
        Used by clustering/titan engines.
        """
        member = dataset.get("member", {})
        accounts = dataset.get("accounts", [])
        txs = dataset.get("transactions", [])

        total_balance = sum([float(a.get("balance", 0)) for a in accounts])

        # Simple aggregations
        inflow = sum([float(t["amount"]) for t in txs if t.get("dr_cr") == "CR"])
        outflow = sum([float(t["amount"]) for t in txs if t.get("dr_cr") == "DR"])

        row = {
            "member_id": member.get("member_id"),
            "tenure_months": member.get("tenure_months"),
            "credit_score": member.get("credit_score"),
            "total_balance": total_balance,
            "account_count": len(accounts),
            "transaction_count": len(txs),
            "total_inflow_90d": inflow,
            "total_outflow_90d": outflow,
            "net_cashflow": inflow - outflow,
        }
        return row
