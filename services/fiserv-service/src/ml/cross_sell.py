"""
Cross-Sell Propensity Engine

Predicts which products each member is most likely to adopt.
Uses XGBoost propensity models with explainable recommendations.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Product definitions with features
PRODUCTS = {
    "auto_loan": {
        "display_name": "Auto Loan",
        "description": "Vehicle financing with competitive rates",
        "feature_flag": "has_auto_loan",
        "target_features": ["has_direct_deposit", "tenure_months", "total_balance"],
    },
    "credit_card": {
        "display_name": "Credit Card",
        "description": "Rewards credit card for everyday purchases",
        "feature_flag": "has_credit_card",
        "target_features": ["debit_count_30d", "avg_transaction_amount", "has_checking"],
    },
    "heloc": {
        "display_name": "Home Equity Line of Credit",
        "description": "Flexible borrowing against home equity",
        "feature_flag": "has_heloc",
        "target_features": ["has_mortgage", "tenure_years", "total_balance"],
    },
    "certificate": {
        "display_name": "Share Certificate (CD)",
        "description": "Higher yield on term deposits",
        "feature_flag": "has_certificate",
        "target_features": ["total_balance", "avg_balance", "has_savings"],
    },
    "savings": {
        "display_name": "High-Yield Savings",
        "description": "Savings account with competitive APY",
        "feature_flag": "has_savings",
        "target_features": ["has_checking", "total_credits_30d", "large_credit_count_30d"],
    },
}


class CrossSellEngine:
    """
    Predicts product propensity and generates recommendations.

    Uses a rule-based scoring system (V1) that can be upgraded
    to trained XGBoost models when historical data is available.
    """

    def __init__(self):
        self.products = PRODUCTS
        self.models = {}  # Placeholder for trained models
        self.model_version = "rule_based_v1"

    def predict(self, member_features: dict[str, Any], top_n: int = 3, min_score: float = 0.3) -> dict[str, Any]:
        """
        Generate product recommendations for a member.

        Args:
            member_features: Feature dict from FeatureCollector
            top_n: Number of recommendations to return
            min_score: Minimum propensity score threshold

        Returns:
            Recommendations with scores and reasons
        """
        recommendations = []

        for product_id, product_info in self.products.items():
            # Skip if member already has product
            flag = product_info["feature_flag"]
            if member_features.get(flag, False):
                continue

            # Calculate propensity score
            score, reasons = self._calculate_propensity(product_id, product_info, member_features)

            if score >= min_score:
                recommendations.append(
                    {
                        "product_id": product_id,
                        "product_name": product_info["display_name"],
                        "description": product_info["description"],
                        "propensity_score": round(score, 3),
                        "confidence": self._score_to_confidence(score),
                        "reasons": reasons[:3],  # Top 3 reasons
                    }
                )

        # Sort by score descending
        recommendations.sort(key=lambda x: x["propensity_score"], reverse=True)

        return {
            "member_id": member_features.get("member_id"),
            "model_version": self.model_version,
            "recommendation_count": min(len(recommendations), top_n),
            "recommendations": recommendations[:top_n],
            "all_products_evaluated": len(self.products),
            "products_already_owned": self._count_owned(member_features),
        }

    def _calculate_propensity(
        self, product_id: str, product_info: dict, features: dict[str, Any]
    ) -> tuple[float, list[str]]:
        """
        Calculate propensity score using rule-based logic.

        Returns (score, reasons) tuple.
        """
        score = 0.0
        reasons = []

        # Base score from member profile strength
        profile_score = self._profile_score(features)
        score += profile_score * 0.2

        # Product-specific scoring
        if product_id == "auto_loan":
            score, reasons = self._score_auto_loan(features, score, reasons)
        elif product_id == "credit_card":
            score, reasons = self._score_credit_card(features, score, reasons)
        elif product_id == "heloc":
            score, reasons = self._score_heloc(features, score, reasons)
        elif product_id == "certificate":
            score, reasons = self._score_certificate(features, score, reasons)
        elif product_id == "savings":
            score, reasons = self._score_savings(features, score, reasons)

        # Tenure bonus (loyal members)
        tenure = features.get("tenure_years", 0)
        if tenure >= 5:
            score += 0.1
            reasons.append(f"Loyal member for {tenure}+ years")

        # Engagement bonus
        if features.get("has_direct_deposit"):
            score += 0.05

        return min(score, 1.0), reasons

    def _score_auto_loan(self, f: dict, score: float, reasons: list[str]) -> tuple:
        """Score auto loan propensity."""

        # Strong indicators
        if f.get("has_direct_deposit"):
            score += 0.25
            reasons.append("Active direct deposit indicates stable income")

        if f.get("tenure_months", 0) >= 24:
            score += 0.15
            reasons.append("Established relationship (2+ years)")

        if f.get("total_balance", 0) > 5000:
            score += 0.15
            reasons.append("Healthy account balances")

        # Transaction patterns suggesting car payment elsewhere
        if f.get("large_credit_count_30d", 0) >= 2 and not f.get("has_loan"):
            score += 0.1
            reasons.append("Regular income but no active loans")

        return score, reasons

    def _score_credit_card(self, f: dict, score: float, reasons: list[str]) -> tuple:
        """Score credit card propensity."""

        # High transaction volume = credit card value
        debit_count = f.get("debit_count_30d", 0)
        if debit_count >= 30:
            score += 0.25
            reasons.append(f"High transaction volume ({debit_count} purchases/month)")
        elif debit_count >= 15:
            score += 0.15
            reasons.append("Moderate transaction activity")

        # Travel/dining detection (if category data available)
        avg_tx = f.get("avg_transaction_amount", 0)
        if avg_tx > 50:
            score += 0.1
            reasons.append("Transaction patterns suggest rewards card value")

        # Has checking but not card
        if f.get("has_checking") and not f.get("has_credit_card"):
            score += 0.15
            reasons.append("Active checking member without credit card")

        return score, reasons

    def _score_heloc(self, f: dict, score: float, reasons: list[str]) -> tuple:
        """Score HELOC propensity."""

        # Mortgage is strong predictor
        if f.get("has_mortgage"):
            score += 0.3
            reasons.append("Existing mortgage indicates home equity available")

        # Long tenure suggests homeowner stability
        tenure = f.get("tenure_years", 0)
        if tenure >= 10:
            score += 0.2
            reasons.append(f"Long-term relationship ({tenure} years)")
        elif tenure >= 5:
            score += 0.1

        # High balances suggest financial capacity
        if f.get("total_balance", 0) > 25000:
            score += 0.15
            reasons.append("Strong deposit relationship")

        # Age bracket
        age = f.get("age") or 0
        if age and 35 <= age <= 55:
            score += 0.1
            reasons.append("Prime home equity utilization age")

        return score, reasons

    def _score_certificate(self, f: dict, score: float, reasons: list[str]) -> tuple:
        """Score certificate/CD propensity."""

        # High savings balance but no CDs
        if f.get("has_savings") and f.get("avg_balance", 0) > 10000:
            score += 0.3
            reasons.append("High savings balance could earn higher yield in CD")

        if f.get("total_balance", 0) > 25000:
            score += 0.2
            reasons.append("Significant deposits suitable for term investment")

        # Stable (low withdrawal) activity
        tx_count = f.get("transaction_count_30d", 0)
        if tx_count < 20 and f.get("total_balance", 0) > 5000:
            score += 0.1
            reasons.append("Low transaction activity suggests funds available for CD")

        # Older members more likely
        age = f.get("age") or 0
        if age and age >= 55:
            score += 0.1
            reasons.append("Age demographic aligned with CD preferences")

        return score, reasons

    def _score_savings(self, f: dict, score: float, reasons: list[str]) -> tuple:
        """Score high-yield savings propensity."""

        # Has checking but not savings
        if f.get("has_checking") and not f.get("has_savings"):
            score += 0.35
            reasons.append("Active checking member without savings account")

        # Good income indicators
        if f.get("large_credit_count_30d", 0) >= 2:
            score += 0.2
            reasons.append("Regular income deposits could build savings")

        total_credits = f.get("total_credits_30d", 0)
        total_debits = f.get("total_debits_30d", 0)
        if total_credits > total_debits + 500:
            score += 0.15
            reasons.append("Positive cash flow could fund savings")

        return score, reasons

    def _profile_score(self, f: dict) -> float:
        """Calculate profile strength score (0-1)."""
        score = 0.0

        if f.get("has_email"):
            score += 0.2
        if f.get("has_phone"):
            score += 0.2
        if f.get("has_direct_deposit"):
            score += 0.3
        if f.get("product_count", 0) >= 2:
            score += 0.3

        return min(score, 1.0)

    def _score_to_confidence(self, score: float) -> str:
        """Convert score to confidence level."""
        if score >= 0.7:
            return "HIGH"
        elif score >= 0.5:
            return "MEDIUM"
        else:
            return "LOW"

    def _count_owned(self, features: dict) -> int:
        """Count products already owned."""
        count = 0
        for product_info in self.products.values():
            if features.get(product_info["feature_flag"], False):
                count += 1
        return count

    def get_available_products(self) -> list[dict[str, Any]]:
        """Get list of products available for cross-sell."""
        return [
            {
                "product_id": pid,
                "name": info["display_name"],
                "description": info["description"],
            }
            for pid, info in self.products.items()
        ]


# Singleton
_engine = None


def get_cross_sell_engine() -> CrossSellEngine:
    """Get cross-sell engine singleton."""
    global _engine
    if _engine is None:
        _engine = CrossSellEngine()
    return _engine
