"""
Member Churn Prediction Model

Predicts probability of member leaving in next 90 days.
Provides explainable risk factors and recommended retention actions.
"""

import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

# Risk factor definitions
RISK_FACTORS = {
    "direct_deposit_stopped": {
        "weight": 0.25,
        "description": "Direct deposit has stopped or decreased",
        "severity": "HIGH",
    },
    "balance_decline": {
        "weight": 0.20,
        "description": "Account balances declining",
        "severity": "HIGH",
    },
    "transaction_decline": {
        "weight": 0.18,
        "description": "Transaction activity decreasing",
        "severity": "MEDIUM",
    },
    "inactivity": {
        "weight": 0.15,
        "description": "Extended period without activity",
        "severity": "HIGH",
    },
    "single_product": {
        "weight": 0.10,
        "description": "Only one product relationship",
        "severity": "MEDIUM",
    },
    "low_engagement": {
        "weight": 0.08,
        "description": "Low overall engagement",
        "severity": "LOW",
    },
    "external_transfers": {
        "weight": 0.12,
        "description": "Increased external transfers out",
        "severity": "MEDIUM",
    },
}

# Retention action templates
RETENTION_ACTIONS = {
    "HIGH": [
        "Immediate relationship manager outreach (within 24 hours)",
        "Offer rate match or bonus on deposits",
        "Schedule financial review appointment",
    ],
    "MEDIUM": [
        "Send personalized retention email",
        "Offer loyalty bonus or promotional rate",
        "Recommend additional products that add value",
    ],
    "LOW": [
        "Add to nurture campaign",
        "Send product awareness communications",
        "Monitor for further decline",
    ],
}


class ChurnPredictor:
    """
    Predicts member churn probability with explainable risk factors.

    Uses rule-based scoring (V1) that identifies key churn signals
    from Fiserv transaction and account data.
    """

    def __init__(self):
        self.risk_factors = RISK_FACTORS
        self.model_version = "rule_based_v1"

    def predict(self, member_features: dict[str, Any]) -> dict[str, Any]:
        """
        Predict churn probability for a member.

        Args:
            member_features: Feature dict from FeatureCollector

        Returns:
            Churn prediction with risk factors and actions
        """
        # Calculate individual risk signals
        signals = self._detect_risk_signals(member_features)

        # Aggregate into churn probability
        churn_prob = self._calculate_churn_probability(signals)

        # Determine risk level
        risk_level = self._get_risk_level(churn_prob)

        # Get top risk factors
        top_factors = self._get_top_factors(signals)

        # Generate recommended actions
        actions = self._get_recommended_actions(risk_level, signals)

        # Estimate days to churn
        days_to_churn = self._estimate_days_to_churn(churn_prob, signals)

        # Count detected signals
        detected_count = len([s for s in signals.values() if s["detected"]])
        no_signals_detected = detected_count == 0

        return {
            "member_id": member_features.get("member_id"),
            "model_version": self.model_version,
            "predicted_at": datetime.now().isoformat(),
            "churn_probability": round(churn_prob, 3),
            "risk_level": risk_level,
            "predicted_days_to_churn": days_to_churn,
            "risk_factor_count": detected_count,
            "top_risk_factors": top_factors,
            "recommended_actions": actions,
            "retention_priority": self._get_priority_score(churn_prob, member_features),
            "no_risk_signals_detected": no_signals_detected,
            "baseline_probability_applied": no_signals_detected,
        }

    def _detect_risk_signals(self, f: dict[str, Any]) -> dict[str, dict]:
        """Detect individual risk signals from features."""
        signals = {}

        # Direct deposit stopped
        signals["direct_deposit_stopped"] = {
            "detected": not f.get("has_direct_deposit", True) and f.get("tenure_months", 0) > 6,
            "severity": self.risk_factors["direct_deposit_stopped"]["severity"],
            "weight": self.risk_factors["direct_deposit_stopped"]["weight"],
            "detail": "No direct deposit detected for established member",
        }

        # Balance decline
        balance = f.get("total_balance", 0)
        signals["balance_decline"] = {
            "detected": balance < 500 and f.get("tenure_months", 0) > 3,
            "severity": self.risk_factors["balance_decline"]["severity"],
            "weight": self.risk_factors["balance_decline"]["weight"],
            "detail": f"Low balance: ${balance:,.2f}",
        }

        # Transaction decline
        tx_change = f.get("tx_count_change_pct", 0)
        signals["transaction_decline"] = {
            "detected": tx_change < -0.3,  # 30% decline
            "severity": self.risk_factors["transaction_decline"]["severity"],
            "weight": self.risk_factors["transaction_decline"]["weight"],
            "detail": f"Transaction activity down {abs(tx_change) * 100:.0f}%",
        }

        # Inactivity
        days_inactive = f.get("days_since_last_tx", 0)
        signals["inactivity"] = {
            "detected": days_inactive > 30,
            "severity": self.risk_factors["inactivity"]["severity"],
            "weight": self.risk_factors["inactivity"]["weight"],
            "detail": f"No transactions in {days_inactive} days",
        }

        # Single product relationship
        product_count = f.get("product_count", 1)
        signals["single_product"] = {
            "detected": product_count <= 1,
            "severity": self.risk_factors["single_product"]["severity"],
            "weight": self.risk_factors["single_product"]["weight"],
            "detail": f"Only {product_count} product(s) - weak relationship",
        }

        # Low engagement
        low_tx = f.get("transaction_count_30d", 0) < 5
        no_dd = not f.get("has_direct_deposit", False)
        signals["low_engagement"] = {
            "detected": low_tx and no_dd,
            "severity": self.risk_factors["low_engagement"]["severity"],
            "weight": self.risk_factors["low_engagement"]["weight"],
            "detail": "Low transaction activity and no direct deposit",
        }

        # External transfer activity
        signals["external_transfers"] = {
            "detected": f.get("external_transfer_activity", False) and balance < 1000,
            "severity": self.risk_factors["external_transfers"]["severity"],
            "weight": self.risk_factors["external_transfers"]["weight"],
            "detail": "External transfers with declining balance",
        }

        return signals

    def _calculate_churn_probability(self, signals: dict[str, dict]) -> float:
        """Calculate aggregate churn probability."""
        # Baseline churn probability - industry average for healthy members
        # No real-world member has exactly 0% churn risk
        BASELINE_CHURN = 0.02  # 2% baseline even for healthy members

        total_weight = 0.0
        detected_weight = 0.0

        for signal_name, signal_data in signals.items():
            weight = signal_data["weight"]
            total_weight += weight

            if signal_data["detected"]:
                # Severity multiplier
                severity = signal_data["severity"]
                multiplier = {"HIGH": 1.0, "MEDIUM": 0.7, "LOW": 0.4}.get(severity, 0.5)
                detected_weight += weight * multiplier

        # Base probability from detected signals
        if total_weight > 0:
            prob = detected_weight / total_weight
        else:
            prob = 0.0

        # Blend with baseline: always have at least baseline risk
        # This ensures no member ever shows exactly 0% churn
        final_prob = max(BASELINE_CHURN, prob) if prob < BASELINE_CHURN else prob

        return min(final_prob, 1.0)

    def _get_risk_level(self, prob: float) -> str:
        """Convert probability to risk level."""
        if prob >= 0.7:
            return "HIGH"
        elif prob >= 0.4:
            return "MEDIUM"
        else:
            return "LOW"

    def _get_top_factors(self, signals: dict[str, dict]) -> list[dict[str, Any]]:
        """Get top contributing risk factors."""
        detected = [
            {
                "factor": name,
                "description": self.risk_factors[name]["description"],
                "severity": data["severity"],
                "detail": data["detail"],
                "impact": data["weight"],
            }
            for name, data in signals.items()
            if data["detected"]
        ]

        # Sort by impact
        detected.sort(key=lambda x: x["impact"], reverse=True)

        return detected[:5]

    def _get_recommended_actions(self, risk_level: str, signals: dict[str, dict]) -> list[str]:
        """Get recommended retention actions."""
        actions = RETENTION_ACTIONS.get(risk_level, []).copy()

        # Add specific actions based on signals
        if signals.get("direct_deposit_stopped", {}).get("detected"):
            actions.insert(0, "Investigate direct deposit stoppage - competitive offer?")

        if signals.get("single_product", {}).get("detected"):
            actions.append("Cross-sell additional products to deepen relationship")

        return actions[:5]

    def _estimate_days_to_churn(self, prob: float, signals: dict[str, dict]) -> int | None:
        """Estimate days until likely churn."""
        if prob < 0.3:
            return None  # Low risk, no prediction

        # Higher prob = sooner churn
        if prob >= 0.8:
            base_days = 30
        elif prob >= 0.6:
            base_days = 60
        elif prob >= 0.4:
            base_days = 90
        else:
            base_days = 120

        # Adjust for specific signals
        if signals.get("inactivity", {}).get("detected"):
            base_days = min(base_days, 45)

        if signals.get("balance_decline", {}).get("detected"):
            base_days -= 10

        return max(base_days, 14)

    def _get_priority_score(self, churn_prob: float, features: dict[str, Any]) -> int:
        """
        Calculate retention priority (1-100).

        Combines churn risk with member value.
        """
        # Base priority from churn probability
        priority = int(churn_prob * 50)

        # Add value component
        balance = features.get("total_balance", 0)
        if balance > 50000:
            priority += 30
        elif balance > 25000:
            priority += 20
        elif balance > 10000:
            priority += 10

        # Tenure bonus
        tenure = features.get("tenure_years", 0)
        if tenure >= 10:
            priority += 15
        elif tenure >= 5:
            priority += 10

        # Product depth
        products = features.get("product_count", 1)
        if products >= 4:
            priority += 10
        elif products >= 2:
            priority += 5

        return min(priority, 100)

    def get_high_risk_threshold(self) -> float:
        """Get threshold for high-risk classification."""
        return 0.7

    def get_medium_risk_threshold(self) -> float:
        """Get threshold for medium-risk classification."""
        return 0.4


# Singleton
_predictor = None


def get_churn_predictor() -> ChurnPredictor:
    """Get churn predictor singleton."""
    global _predictor
    if _predictor is None:
        _predictor = ChurnPredictor()
    return _predictor
