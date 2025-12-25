"""
Market Basket Analysis Engine - Schema-Agnostic Version

Identifies product associations (affinity analysis) using Apriori/FP-Growth logic.
Recommends cross-sell and bundle opportunities.

NOW SCHEMA-AGNOSTIC: Works with any transaction dataset via semantic column mapping.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import logging
from collections import Counter
from itertools import combinations
from typing import Any

import pandas as pd
from core import ColumnMapper, ColumnProfiler, EngineRequirements, SemanticType
from core.gemma_summarizer import GemmaSummarizer
from core.premium_models import ConfigParameter

logger = logging.getLogger(__name__)


class MarketBasketAnalysisEngine:
    """
    Schema-agnostic market basket analysis engine.

    Automatically detects transaction and item columns using semantic intelligence.
    """

    def __init__(self):
        self.name = "Market Basket Analysis Engine (Schema-Agnostic)"
        self.profiler = ColumnProfiler()
        self.mapper = ColumnMapper()

    def get_engine_info(self) -> dict[str, str]:
        return {
            "name": "market_basket",
            "display_name": "Market Basket Analysis",
            "icon": "🛒",
            "task_type": "discovery",
        }

    def get_config_schema(self) -> list[ConfigParameter]:
        return [
            ConfigParameter(
                name="transaction_column",
                type="select",
                default=None,
                range=[],
                description="Transaction/basket identifier",
            ),
            ConfigParameter(
                name="item_column", type="select", default=None, range=[], description="Item/product column"
            ),
            ConfigParameter(
                name="min_support",
                type="float",
                default=0.01,
                range=[0.001, 0.5],
                description="Minimum support threshold",
            ),
        ]

    def get_methodology_info(self) -> dict[str, Any]:
        return {
            "name": "Market Basket Analysis (Association Rules)",
            "url": "https://en.wikipedia.org/wiki/Association_rule_learning",
            "steps": [
                {"step_number": 1, "title": "Transaction Prep", "description": "Convert data to transaction format"},
                {"step_number": 2, "title": "Frequent Itemsets", "description": "Find frequently co-occurring items"},
                {"step_number": 3, "title": "Association Rules", "description": "Generate and rank association rules"},
            ],
            "limitations": ["Requires sufficient transaction volume"],
            "assumptions": ["Items are discrete and identifiable"],
        }

    def get_requirements(self) -> EngineRequirements:
        """
        Define semantic requirements for market basket analysis.

        Returns:
            EngineRequirements specifying needed columns
        """
        return EngineRequirements(
            required_semantics=[SemanticType.CATEGORICAL],  # Need at least 2 categorical cols
            optional_semantics={},
            required_entities=[],
            preferred_domains=[],
            applicable_tasks=[],
            min_rows=10,
            min_numeric_cols=0,
        )

    def analyze(self, df: pd.DataFrame, config: dict | None = None) -> dict[str, Any]:
        """
        Run schema-agnostic market basket analysis.

        Args:
            df: DataFrame with transaction data
            config: Configuration with optional keys:
                - transaction_column: Hint for transaction ID
                - item_column: Hint for item/product column
                - min_support: Minimum support threshold (default: 0.01)
                - min_confidence: Minimum confidence threshold (default: 0.5)
                - skip_profiling: Skip schema intelligence (default: False)

        Returns:
            Dict with association rules, visualizations, and column mappings
        """
        config = config or {}
        min_support = config.get("min_support", 0.01)
        min_confidence = config.get("min_confidence", 0.5)

        results = {
            "summary": {},
            "rules": [],
            "graphs": [],
            "insights": [],
            "column_mappings": {},
            "profiling_used": not config.get("skip_profiling", False),
        }

        try:
            # SCHEMA INTELLIGENCE: Profile dataset
            if not config.get("skip_profiling", False):
                logger.info("Profiling dataset for schema-agnostic market basket analysis")
                profiles = self.profiler.profile_dataset(df)

                # Detect transaction and item columns
                trans_col = self._smart_detect(
                    df,
                    profiles,
                    keywords=["transaction", "order", "invoice", "basket", "receipt"],
                    hint=config.get("transaction_column"),
                    semantic_type=SemanticType.CATEGORICAL,
                )

                item_col = self._smart_detect(
                    df,
                    profiles,
                    keywords=["item", "product", "sku", "article", "name"],
                    hint=config.get("item_column"),
                    semantic_type=SemanticType.CATEGORICAL,
                )

                if trans_col:
                    results["column_mappings"]["transaction"] = {
                        "column": trans_col,
                        "confidence": 0.9,
                        "semantic_type": "CATEGORICAL",
                    }
                if item_col:
                    results["column_mappings"]["item"] = {
                        "column": item_col,
                        "confidence": 0.9,
                        "semantic_type": "CATEGORICAL",
                    }
            else:
                # Fallback: use hints or old detection
                trans_col = config.get("transaction_column") or self._detect_column(
                    df, ["transaction", "order", "invoice", "id"], None
                )
                item_col = config.get("item_column") or self._detect_column(
                    df, ["item", "product", "sku", "name"], None
                )
                results["profiling_used"] = False

            if trans_col and item_col:
                # Group items by transaction
                transactions = df.groupby(trans_col)[item_col].apply(list).tolist()
                n_transactions = len(transactions)

                # Calculate item support
                item_counts = Counter()
                for t in transactions:
                    item_counts.update(set(t))  # Use set to count presence only once per transaction

                # Filter frequent items
                frequent_items = {
                    item: count / n_transactions
                    for item, count in item_counts.items()
                    if count / n_transactions >= min_support
                }

                # Generate pairs (size 2 itemsets)
                pair_counts = Counter()
                for t in transactions:
                    # Only consider items that are frequent
                    items = [i for i in set(t) if i in frequent_items]
                    if len(items) >= 2:
                        pair_counts.update(combinations(sorted(items), 2))

                # Generate rules
                rules = []
                for (item_a, item_b), count in pair_counts.items():
                    support = count / n_transactions
                    if support >= min_support:
                        # Rule A -> B
                        conf_a_b = support / frequent_items[item_a]
                        lift_a_b = conf_a_b / frequent_items[item_b]

                        if conf_a_b >= min_confidence:
                            rules.append(
                                {
                                    "antecedent": item_a,
                                    "consequent": item_b,
                                    "support": float(support),
                                    "confidence": float(conf_a_b),
                                    "lift": float(lift_a_b),
                                }
                            )

                        # Rule B -> A
                        conf_b_a = support / frequent_items[item_b]
                        lift_b_a = conf_b_a / frequent_items[item_a]

                        if conf_b_a >= min_confidence:
                            rules.append(
                                {
                                    "antecedent": item_b,
                                    "consequent": item_a,
                                    "support": float(support),
                                    "confidence": float(conf_b_a),
                                    "lift": float(lift_b_a),
                                }
                            )

                # Sort rules by lift
                rules.sort(key=lambda x: x["lift"], reverse=True)
                top_rules = rules[:20]  # Top 20 rules

                results["rules"] = top_rules
                results["summary"] = {
                    "total_transactions": n_transactions,
                    "unique_items": len(item_counts),
                    "rules_found": len(rules),
                }

                # Visualizations

                # 1. Network Graph (Nodes = Items, Edges = Rules)
                network = self._network_graph(top_rules)
                results["graphs"].append(network["graph"])

                # 2. Scatter Plot (Support vs Confidence vs Lift)
                scatter = self._rule_scatter(rules)
                results["graphs"].append(scatter["graph"])

                # Generate insights
                results["insights"] = self._generate_insights(results, trans_col, item_col)

            else:
                # Use Gemma fallback when required columns are missing
                return GemmaSummarizer.generate_fallback_summary(
                    df,
                    engine_name="market_basket",
                    error_reason="Could not detect transaction and item columns. Market basket analysis requires transaction ID and product/item columns.",
                    config=config,
                )

        except Exception as e:
            logger.error(f"Market basket analysis failed: {e}")
            return GemmaSummarizer.generate_fallback_summary(
                df, engine_name="market_basket", error_reason=str(e), config=config
            )

        return results

    def _smart_detect(
        self,
        df: pd.DataFrame,
        profiles: dict,
        keywords: list[str],
        hint: str | None = None,
        semantic_type: SemanticType | None = None,
    ) -> str | None:
        """
        Smart column detection with semantic type filtering.

        Priority:
        1. User hint
        2. Keyword match + semantic type
        3. Semantic type only
        """
        if hint and hint in df.columns:
            return hint

        # Priority 1: Keyword match with semantic type
        for col in df.columns:
            if any(kw in col.lower() for kw in keywords):
                prof = profiles.get(col)
                if prof:
                    if semantic_type:
                        if prof.semantic_type == semantic_type:
                            return col
                    elif prof.semantic_type in [SemanticType.CATEGORICAL, SemanticType.IDENTIFIER]:
                        return col

        # Priority 2: Any column matching semantic type
        if semantic_type:
            for col, prof in profiles.items():
                if prof.semantic_type == semantic_type:
                    return col

        return None

    def _detect_column(self, df: pd.DataFrame, keywords: list[str], hint: str | None = None) -> str | None:
        """Legacy column detection (fallback only)"""
        if hint and hint in df.columns:
            return hint
        for col in df.columns:
            if any(kw in col.lower() for kw in keywords):
                return col
        return None

    def _network_graph(self, rules: list[dict]) -> dict:
        """Generate network graph data for associations"""
        nodes = set()
        links = []

        for r in rules:
            nodes.add(r["antecedent"])
            nodes.add(r["consequent"])
            links.append({"source": r["antecedent"], "target": r["consequent"], "value": r["lift"]})

        graph = {
            "type": "network_graph",
            "title": "Product Association Network",
            "nodes": [{"id": n, "group": 1} for n in nodes],
            "links": links,
        }

        return {"graph": graph}

    def _rule_scatter(self, rules: list[dict]) -> dict:
        """Scatter plot of rules"""
        # Limit to top 100 for performance
        plot_rules = rules[:100]

        graph = {
            "type": "bubble_chart",
            "title": "Association Rules (Support vs Confidence)",
            "x_data": [r["support"] for r in plot_rules],
            "y_data": [r["confidence"] for r in plot_rules],
            "size_data": [r["lift"] for r in plot_rules],
            "labels": [f"{r['antecedent']} -> {r['consequent']}" for r in plot_rules],
            "x_label": "Support",
            "y_label": "Confidence",
        }

        return {"graph": graph}

    def _generate_insights(self, results: dict, trans_col: str | None = None, item_col: str | None = None) -> list[str]:
        """Generate schema-aware insights"""
        insights = []

        # Schema-aware insight
        if trans_col and item_col:
            insights.append(f"📊 Analyzed transactions from '{trans_col}' and items from '{item_col}'")

        if results["rules"]:
            top = results["rules"][0]
            insights.append(
                f"🔗 Strongest Association: Customers who buy '{top['antecedent']}' "
                f"are {top['lift']:.1f}x more likely to buy '{top['consequent']}'"
            )

            insights.append(
                f"💡 Recommendation: Bundle '{top['antecedent']}' and '{top['consequent']}' for a cross-sell promotion"
            )

        return insights


__all__ = ["MarketBasketAnalysisEngine"]
