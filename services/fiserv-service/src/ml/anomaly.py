"""
ML Anomaly Detection Module

Simple statistical anomaly detection for banking transactions.
Human-in-the-loop: flags anomalies for review, doesn't take action.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class AnomalyType(str, Enum):
    """Types of detected anomalies."""

    HIGH_VALUE = "high_value"
    UNUSUAL_FREQUENCY = "unusual_frequency"
    UNUSUAL_TIMING = "unusual_timing"
    DUPLICATE_SUSPECTED = "duplicate_suspected"
    VELOCITY_SPIKE = "velocity_spike"


@dataclass
class AnomalyFlag:
    """Represents a flagged anomaly for human review."""

    anomaly_type: AnomalyType
    severity: str  # "low", "medium", "high"
    description: str
    transaction_id: str | None = None
    account_id: str | None = None
    amount: float | None = None
    details: dict[str, Any] | None = None


class AnomalyDetector:
    """
    Statistical anomaly detector for transactions.

    Uses simple statistical methods (Z-score, IQR) to flag
    unusual patterns for human review.
    """

    def __init__(self, z_score_threshold: float = 3.0, iqr_multiplier: float = 1.5):
        """
        Initialize detector.

        Args:
            z_score_threshold: Z-score above which is anomalous
            iqr_multiplier: IQR multiplier for outlier detection
        """
        self.z_score_threshold = z_score_threshold
        self.iqr_multiplier = iqr_multiplier

    def detect_high_value_transactions(
        self, transactions: list[dict[str, Any]], method: str = "zscore"
    ) -> list[AnomalyFlag]:
        """
        Detect unusually high-value transactions.

        Args:
            transactions: List of transaction dicts
            method: Detection method ("zscore" or "iqr")

        Returns:
            List of flagged anomalies
        """
        flags = []

        amounts = [t.get("amount", 0) for t in transactions if t.get("amount")]
        if len(amounts) < 5:
            return flags  # Not enough data

        amounts = np.array(amounts)

        if method == "zscore":
            mean = np.mean(amounts)
            std = np.std(amounts)
            if std == 0:
                return flags

            for txn in transactions:
                amt = txn.get("amount", 0)
                if amt and std > 0:
                    z = (amt - mean) / std
                    if abs(z) > self.z_score_threshold:
                        severity = "high" if z > 4 else "medium" if z > 3.5 else "low"
                        flags.append(
                            AnomalyFlag(
                                anomaly_type=AnomalyType.HIGH_VALUE,
                                severity=severity,
                                description=f"Transaction amount ${amt:.2f} is {z:.1f} std devs from mean ${mean:.2f}",
                                transaction_id=txn.get("transaction_id"),
                                account_id=txn.get("account_id"),
                                amount=amt,
                                details={"z_score": z, "mean": mean, "std": std},
                            )
                        )

        elif method == "iqr":
            q1, q3 = np.percentile(amounts, [25, 75])
            iqr = q3 - q1
            upper_bound = q3 + (self.iqr_multiplier * iqr)

            for txn in transactions:
                amt = txn.get("amount", 0)
                if amt and amt > upper_bound:
                    flags.append(
                        AnomalyFlag(
                            anomaly_type=AnomalyType.HIGH_VALUE,
                            severity="medium",
                            description=f"Transaction ${amt:.2f} exceeds upper bound ${upper_bound:.2f}",
                            transaction_id=txn.get("transaction_id"),
                            account_id=txn.get("account_id"),
                            amount=amt,
                            details={"upper_bound": upper_bound, "iqr": iqr},
                        )
                    )

        return flags

    def detect_duplicates(
        self, transactions: list[dict[str, Any]], amount_tolerance: float = 0.01, time_window_hours: int = 24
    ) -> list[AnomalyFlag]:
        """
        Detect potential duplicate transactions.

        Args:
            transactions: List of transaction dicts
            amount_tolerance: Percentage tolerance for amount match
            time_window_hours: Hours within which to check

        Returns:
            List of flagged potential duplicates
        """
        flags = []

        # Group by similar amounts
        from collections import defaultdict

        amount_groups = defaultdict(list)

        for txn in transactions:
            amt = txn.get("amount", 0)
            if amt:
                # Round to nearest dollar for grouping
                key = round(amt)
                amount_groups[key].append(txn)

        # Check for duplicates within groups
        for key, group in amount_groups.items():
            if len(group) >= 2:
                # Multiple transactions with same amount
                for i, txn1 in enumerate(group):
                    for txn2 in group[i + 1 :]:
                        if txn1.get("description") == txn2.get("description"):
                            flags.append(
                                AnomalyFlag(
                                    anomaly_type=AnomalyType.DUPLICATE_SUSPECTED,
                                    severity="medium",
                                    description=f"Possible duplicate: ${key} - '{txn1.get('description', 'N/A')}'",
                                    transaction_id=txn1.get("transaction_id"),
                                    account_id=txn1.get("account_id"),
                                    amount=key,
                                    details={
                                        "txn1_id": txn1.get("transaction_id"),
                                        "txn2_id": txn2.get("transaction_id"),
                                        "txn1_date": txn1.get("date"),
                                        "txn2_date": txn2.get("date"),
                                    },
                                )
                            )

        return flags

    def detect_velocity_spike(
        self, transactions: list[dict[str, Any]], threshold_multiplier: float = 3.0
    ) -> list[AnomalyFlag]:
        """
        Detect unusual transaction velocity (frequency spikes).

        Args:
            transactions: List of transaction dicts
            threshold_multiplier: Multiplier over average for spike

        Returns:
            List of flagged velocity spikes
        """
        flags = []

        if len(transactions) < 10:
            return flags

        # Count transactions per day
        from collections import Counter

        daily_counts = Counter()

        for txn in transactions:
            date = txn.get("date", "")
            if date:
                day = date[:10]  # YYYY-MM-DD
                daily_counts[day] += 1

        if len(daily_counts) < 3:
            return flags

        counts = list(daily_counts.values())
        avg_daily = np.mean(counts)

        for day, count in daily_counts.items():
            if count > avg_daily * threshold_multiplier:
                flags.append(
                    AnomalyFlag(
                        anomaly_type=AnomalyType.VELOCITY_SPIKE,
                        severity="medium",
                        description=f"High transaction volume on {day}: {count} vs avg {avg_daily:.1f}",
                        details={"date": day, "count": count, "average": avg_daily, "multiplier": count / avg_daily},
                    )
                )

        return flags

    def analyze_all(self, transactions: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Run all anomaly detection methods.

        Args:
            transactions: List of transaction dicts

        Returns:
            Analysis results with all flags
        """
        high_value = self.detect_high_value_transactions(transactions)
        duplicates = self.detect_duplicates(transactions)
        velocity = self.detect_velocity_spike(transactions)

        all_flags = high_value + duplicates + velocity

        return {
            "transaction_count": len(transactions),
            "total_flags": len(all_flags),
            "high_value_flags": len(high_value),
            "duplicate_flags": len(duplicates),
            "velocity_flags": len(velocity),
            "flags": [
                {
                    "type": f.anomaly_type.value,
                    "severity": f.severity,
                    "description": f.description,
                    "transaction_id": f.transaction_id,
                    "account_id": f.account_id,
                    "amount": f.amount,
                    "details": f.details,
                }
                for f in all_flags
            ],
        }


def get_anomaly_detector() -> AnomalyDetector:
    """Get anomaly detector instance."""
    return AnomalyDetector()
