"""
Dynamic Loan Pricing Optimizer

Recommends optimal loan interest rates based on:
- Risk tier (from Zest AI or credit score)
- Member relationship value
- Market competitiveness
- Product type and term
"""

import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

# Base cost of funds (adjustable)
DEFAULT_COST_OF_FUNDS = 4.5

# Risk tier premiums (basis points above cost of funds)
RISK_PREMIUMS = {
    "A+": 1.50,  # Excellent credit
    "A": 2.00,
    "B": 3.00,
    "C": 4.50,
    "D": 6.00,
    "E": 8.00,  # High risk
}

# Credit score to risk tier mapping
CREDIT_SCORE_TIERS = {
    (760, 850): "A+",
    (720, 759): "A",
    (680, 719): "B",
    (640, 679): "C",
    (580, 639): "D",
    (300, 579): "E",
}

# Loan type base adjustments
LOAN_TYPE_ADJUSTMENTS = {
    "auto_new": {"adjustment": -0.25, "min_term": 24, "max_term": 84},
    "auto_used": {"adjustment": 0.50, "min_term": 24, "max_term": 72},
    "personal": {"adjustment": 1.50, "min_term": 12, "max_term": 60},
    "heloc": {"adjustment": 0.00, "min_term": 60, "max_term": 240},
    "signature": {"adjustment": 2.00, "min_term": 12, "max_term": 36},
}

# Market rate references (would be updated dynamically)
MARKET_RATES = {
    "auto_new_60": 6.49,
    "auto_used_60": 7.49,
    "personal_36": 11.99,
    "heloc": 8.50,
    "signature_24": 12.99,
}


class PricingOptimizer:
    """
    Optimizes loan pricing based on risk, relationship, and market factors.

    Provides rate recommendations with adjustments breakdown.
    """

    def __init__(self, cost_of_funds: float = DEFAULT_COST_OF_FUNDS):
        self.cost_of_funds = cost_of_funds
        self.model_version = "rule_based_v1"

    def optimize(
        self,
        member_features: dict[str, Any],
        loan_type: str,
        amount: float,
        term_months: int,
        credit_score: int | None = None,
        risk_tier: str | None = None,
    ) -> dict[str, Any]:
        """
        Calculate optimal loan rate.

        Args:
            member_features: Member features from FeatureCollector
            loan_type: Type of loan (auto_new, auto_used, personal, heloc, signature)
            amount: Requested loan amount
            term_months: Loan term in months
            credit_score: Credit score (optional if risk_tier provided)
            risk_tier: Risk tier from Zest AI (optional, derived from score if not provided)

        Returns:
            Rate recommendation with breakdown
        """
        # Determine risk tier
        if risk_tier is None:
            risk_tier = self._score_to_tier(credit_score or 700)

        # Get base rate components
        risk_premium = RISK_PREMIUMS.get(risk_tier, 4.0)
        base_rate = self.cost_of_funds + risk_premium

        # Loan type adjustment
        loan_config = LOAN_TYPE_ADJUSTMENTS.get(loan_type, {"adjustment": 0})
        type_adjustment = loan_config.get("adjustment", 0)

        # Calculate relationship adjustments
        adjustments = self._calculate_adjustments(member_features, loan_type, amount, term_months)

        # Total adjustment
        total_adjustment = type_adjustment + sum(adj["amount"] for adj in adjustments)

        # Final rate calculation
        recommended_rate = base_rate + total_adjustment

        # Floor and ceiling
        min_rate = self.cost_of_funds + 0.5  # At least 50bp margin
        max_rate = 18.0  # Regulatory/usury ceiling
        recommended_rate = max(min_rate, min(recommended_rate, max_rate))

        # Calculate competitive positioning
        market_key = f"{loan_type}_{term_months}" if term_months else loan_type
        market_rate = MARKET_RATES.get(market_key, MARKET_RATES.get(loan_type, 8.0))

        win_probability = self._estimate_win_probability(recommended_rate, market_rate)

        return {
            "member_id": member_features.get("member_id"),
            "model_version": self.model_version,
            "calculated_at": datetime.now().isoformat(),
            # Rate recommendation
            "recommended_rate": round(recommended_rate, 2),
            "rate_range": {
                "floor": round(recommended_rate - 0.25, 2),
                "ceiling": round(recommended_rate + 0.50, 2),
            },
            # Rate components
            "rate_breakdown": {
                "cost_of_funds": self.cost_of_funds,
                "risk_premium": risk_premium,
                "base_rate": round(base_rate, 2),
                "loan_type_adjustment": type_adjustment,
                "relationship_adjustments": adjustments,
                "total_adjustment": round(total_adjustment, 2),
            },
            # Risk assessment
            "risk_tier": risk_tier,
            "credit_score": credit_score,
            # Loan details
            "loan_type": loan_type,
            "loan_amount": amount,
            "term_months": term_months,
            # Financial projections
            "expected_margin": round(recommended_rate - self.cost_of_funds, 2),
            "monthly_payment": self._calculate_payment(amount, recommended_rate, term_months),
            "total_interest": self._calculate_total_interest(amount, recommended_rate, term_months),
            # Competitive analysis
            "market_rate_reference": market_rate,
            "vs_market": round(recommended_rate - market_rate, 2),
            "win_probability": win_probability,
            # Approval recommendation
            "approval_recommendation": self._get_approval_recommendation(risk_tier, member_features),
        }

    def _score_to_tier(self, score: int) -> str:
        """Convert credit score to risk tier."""
        for (low, high), tier in CREDIT_SCORE_TIERS.items():
            if low <= score <= high:
                return tier
        return "C"  # Default

    def _calculate_adjustments(
        self, features: dict[str, Any], loan_type: str, amount: float, term_months: int
    ) -> list[dict[str, Any]]:
        """Calculate relationship-based rate adjustments."""
        adjustments = []

        # Tenure discount
        tenure = features.get("tenure_years", 0)
        if tenure >= 10:
            adjustments.append(
                {
                    "name": "10+ year member loyalty",
                    "amount": -0.35,
                    "reason": f"Valued {tenure}-year relationship",
                }
            )
        elif tenure >= 5:
            adjustments.append(
                {
                    "name": "5+ year member loyalty",
                    "amount": -0.25,
                    "reason": f"{tenure}-year established relationship",
                }
            )
        elif tenure >= 2:
            adjustments.append(
                {
                    "name": "Established member",
                    "amount": -0.10,
                    "reason": f"{tenure}-year member",
                }
            )

        # Multi-product relationship
        products = features.get("product_count", 1)
        if products >= 5:
            adjustments.append(
                {
                    "name": "Premium relationship",
                    "amount": -0.30,
                    "reason": f"{products} products - full banking relationship",
                }
            )
        elif products >= 3:
            adjustments.append(
                {
                    "name": "Multi-product member",
                    "amount": -0.20,
                    "reason": f"{products} products",
                }
            )
        elif products >= 2:
            adjustments.append(
                {
                    "name": "Multiple products",
                    "amount": -0.10,
                    "reason": f"{products} products",
                }
            )

        # High deposit relationship
        balance = features.get("total_balance", 0)
        if balance >= 100000:
            adjustments.append(
                {
                    "name": "High-value depositor",
                    "amount": -0.35,
                    "reason": f"${balance:,.0f} in deposits",
                }
            )
        elif balance >= 50000:
            adjustments.append(
                {
                    "name": "Strong deposit relationship",
                    "amount": -0.25,
                    "reason": f"${balance:,.0f} in deposits",
                }
            )
        elif balance >= 25000:
            adjustments.append(
                {
                    "name": "Quality deposit relationship",
                    "amount": -0.15,
                    "reason": f"${balance:,.0f} in deposits",
                }
            )

        # Direct deposit = stable income
        if features.get("has_direct_deposit"):
            adjustments.append(
                {
                    "name": "Direct deposit active",
                    "amount": -0.10,
                    "reason": "Verified income through direct deposit",
                }
            )

        # Auto-pay discount
        if loan_type in ("auto_new", "auto_used", "personal"):
            adjustments.append(
                {
                    "name": "Auto-pay discount (if elected)",
                    "amount": -0.25,
                    "reason": "Required: Set up automatic payments",
                }
            )

        return adjustments

    def _calculate_payment(self, principal: float, annual_rate: float, months: int) -> float:
        """Calculate monthly payment."""
        if months <= 0 or annual_rate <= 0:
            return principal / max(months, 1)

        monthly_rate = annual_rate / 100 / 12
        payment = principal * (monthly_rate * (1 + monthly_rate) ** months) / ((1 + monthly_rate) ** months - 1)
        return round(payment, 2)

    def _calculate_total_interest(self, principal: float, annual_rate: float, months: int) -> float:
        """Calculate total interest over loan term."""
        monthly_payment = self._calculate_payment(principal, annual_rate, months)
        total_paid = monthly_payment * months
        return round(total_paid - principal, 2)

    def _estimate_win_probability(self, our_rate: float, market_rate: float) -> float:
        """Estimate probability of member accepting this rate."""
        # Simple linear model
        diff = market_rate - our_rate  # Positive = we're cheaper

        if diff >= 1.0:  # 1%+ cheaper
            return 0.90
        elif diff >= 0.5:
            return 0.80
        elif diff >= 0.0:
            return 0.70
        elif diff >= -0.5:
            return 0.55
        elif diff >= -1.0:
            return 0.40
        else:
            return 0.25

    def _get_approval_recommendation(self, risk_tier: str, features: dict[str, Any]) -> dict[str, Any]:
        """Generate approval recommendation."""
        if risk_tier in ("A+", "A"):
            return {
                "recommendation": "APPROVE",
                "confidence": "HIGH",
                "notes": "Low-risk member with strong credit",
            }
        elif risk_tier == "B":
            return {
                "recommendation": "APPROVE",
                "confidence": "MEDIUM",
                "notes": "Standard risk, recommend standard terms",
            }
        elif risk_tier == "C":
            tenure = features.get("tenure_years", 0)
            if tenure >= 3:
                return {
                    "recommendation": "APPROVE_WITH_CONDITIONS",
                    "confidence": "MEDIUM",
                    "notes": f"Elevated risk offset by {tenure}-year relationship",
                }
            else:
                return {
                    "recommendation": "MANUAL_REVIEW",
                    "confidence": "LOW",
                    "notes": "Elevated risk, limited relationship history",
                }
        else:  # D, E
            return {
                "recommendation": "MANUAL_REVIEW",
                "confidence": "LOW",
                "notes": "High risk tier requires underwriter review",
            }

    def get_loan_types(self) -> list[dict[str, str]]:
        """Get available loan types."""
        return [
            {"id": "auto_new", "name": "New Auto Loan"},
            {"id": "auto_used", "name": "Used Auto Loan"},
            {"id": "personal", "name": "Personal Loan"},
            {"id": "heloc", "name": "Home Equity Line"},
            {"id": "signature", "name": "Signature Loan"},
        ]

    def update_cost_of_funds(self, new_cost: float) -> None:
        """Update cost of funds parameter."""
        self.cost_of_funds = new_cost
        logger.info(f"Cost of funds updated to {new_cost}%")


# Singleton
_optimizer = None


def get_pricing_optimizer() -> PricingOptimizer:
    """Get pricing optimizer singleton."""
    global _optimizer
    if _optimizer is None:
        _optimizer = PricingOptimizer()
    return _optimizer
