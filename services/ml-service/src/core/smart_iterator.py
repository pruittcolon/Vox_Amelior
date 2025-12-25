"""
Smart Iterator - Intelligent Feature Combination Search

Automatically tests feature combinations to find the best prediction.
Scales intelligently based on dataset size and time budget.

Engine Tier System:
- FAST engines (chaos, scout, flash): Auto-iterate ALL combinations
- MEDIUM engines (chronos, deep_feature, galileo): Limited iterations + "test more"
- SLOW engines (titan, oracle, newton, mirror): Single test, prompt user for variable

Author: NeMo Analytics Team
"""

import logging
import time
from collections.abc import Generator
from dataclasses import dataclass
from enum import Enum
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# ENGINE TIER CLASSIFICATION
# Based on observed timing benchmarks from GTX 1660 Ti testing
# =============================================================================


class EngineTier(Enum):
    """Engine speed classification for iteration strategy."""

    FAST = "fast"  # <3s per iteration - auto-iterate all
    MEDIUM = "medium"  # 3-15s per iteration - limited iterations
    SLOW = "slow"  # >15s per iteration - single test, prompt user


# Engine tier configuration with timing and iteration limits
ENGINE_TIERS: dict[str, dict[str, Any]] = {
    # FAST ENGINES: Auto-iterate all combinations (<3s per iteration)
    "chaos": {
        "tier": EngineTier.FAST,
        "avg_time_seconds": 1.5,
        "max_auto_iterations": 100,
        "prompt_user": False,
        "description": "Chaos Theory Analysis - Fast statistical tests",
    },
    "scout": {
        "tier": EngineTier.FAST,
        "avg_time_seconds": 2.0,
        "max_auto_iterations": 100,
        "prompt_user": False,
        "description": "Data Drift Detection - Quick distribution tests",
    },
    "flash": {
        "tier": EngineTier.FAST,
        "avg_time_seconds": 2.5,
        "max_auto_iterations": 80,
        "prompt_user": False,
        "description": "Counterfactual Analysis - Fast what-if scenarios",
    },
    # MEDIUM ENGINES: Limited iterations with "test more" option (3-15s per iteration)
    "chronos": {
        "tier": EngineTier.MEDIUM,
        "avg_time_seconds": 6.0,
        "max_auto_iterations": 15,
        "prompt_user": False,
        "description": "Time Series Forecasting - Prophet-based predictions",
    },
    "deep_feature": {
        "tier": EngineTier.MEDIUM,
        "avg_time_seconds": 8.0,
        "max_auto_iterations": 12,
        "prompt_user": False,
        "description": "Deep Feature Synthesis - Automated feature engineering",
    },
    "galileo": {
        "tier": EngineTier.MEDIUM,
        "avg_time_seconds": 10.0,
        "max_auto_iterations": 10,
        "prompt_user": False,
        "description": "Graph Neural Network - Relationship modeling",
    },
    # SLOW ENGINES: Single test, prompt user to select variable (>15s per iteration)
    "titan": {
        "tier": EngineTier.SLOW,
        "avg_time_seconds": 45.0,
        "max_auto_iterations": 1,
        "prompt_user": True,
        "description": "AutoML Engine - Full model training with cross-validation",
    },
    "oracle": {
        "tier": EngineTier.SLOW,
        "avg_time_seconds": 25.0,
        "max_auto_iterations": 1,
        "prompt_user": True,
        "description": "Causal Discovery - Granger causality analysis",
    },
    "newton": {
        "tier": EngineTier.SLOW,
        "avg_time_seconds": 30.0,
        "max_auto_iterations": 1,
        "prompt_user": True,
        "description": "Symbolic Regression - Genetic programming equations",
    },
    "mirror": {
        "tier": EngineTier.SLOW,
        "avg_time_seconds": 60.0,
        "max_auto_iterations": 1,
        "prompt_user": True,
        "description": "Synthetic Data Generation - CTGAN training",
    },
}


def get_engine_tier(engine_name: str) -> dict[str, Any]:
    """Get tier info for an engine. Returns MEDIUM defaults if unknown."""
    return ENGINE_TIERS.get(
        engine_name.lower(),
        {
            "tier": EngineTier.MEDIUM,
            "avg_time_seconds": 10.0,
            "max_auto_iterations": 10,
            "prompt_user": False,
            "description": f"Unknown engine: {engine_name}",
        },
    )


def should_prompt_user(engine_name: str) -> bool:
    """Check if this engine should prompt user to select variable first."""
    return get_engine_tier(engine_name).get("prompt_user", False)


def get_max_iterations(engine_name: str) -> int:
    """Get maximum auto-iterations for this engine."""
    return get_engine_tier(engine_name).get("max_auto_iterations", 10)


def estimate_total_time(engine_name: str, num_iterations: int) -> float:
    """Estimate total time in seconds for given number of iterations."""
    tier_info = get_engine_tier(engine_name)
    return tier_info.get("avg_time_seconds", 10.0) * num_iterations


class IterationStrategy(Enum):
    """Iteration strategies based on dataset size."""

    EXHAUSTIVE = "exhaustive"  # Try all combinations (<100 rows, <10 cols)
    GREEDY = "greedy"  # Forward selection (100-500 rows, <15 cols)
    SAMPLED = "sampled"  # Random sampling (500-2000 rows)
    QUICK = "quick"  # Best guess only (>2000 rows)
    SINGLE = "single"  # Single test for slow engines (user selected)


@dataclass
class IterationResult:
    """Result from a single iteration."""

    rank: int
    score: float
    score_type: str
    features_used: list[str]
    time_ms: float
    model_info: dict[str, Any]


@dataclass
class SmartIterateResult:
    """Complete result from smart iteration."""

    best: IterationResult
    all_iterations: list[IterationResult]
    failed_attempts: list[dict[str, Any]]
    remaining_combos: list[list[str]]
    metadata: dict[str, Any]


class SmartIterator:
    """
    Intelligently iterates through feature combinations to find best predictions.

    Tier-Based Strategy:
    - FAST engines (chaos, scout, flash): Auto-iterate all combinations
    - MEDIUM engines (chronos, deep_feature, galileo): Limited iterations + "test more"
    - SLOW engines (titan, oracle, newton, mirror): Single test only

    Usage:
        iterator = SmartIterator(df, 'salary', 'newton')
        results = iterator.iterate(max_iterations=50, time_budget=30.0)
    """

    # Timing estimates per iteration (ms) for GTX 1660 Ti
    BASE_TIME_MS = 50  # Base time for 100 rows, 5 features

    def __init__(self, df: pd.DataFrame, target: str, engine_type: str):
        self.df = df
        self.target = target
        self.engine_type = engine_type
        self.results_cache: list[IterationResult] = []
        self.failed_attempts: list[dict[str, Any]] = []

        # Get engine tier info
        self.tier_info = get_engine_tier(engine_type)
        self.tier = self.tier_info.get("tier", EngineTier.MEDIUM)

        # Get numeric feature columns (excluding target)
        self.numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col != target]

        # Dataset characteristics
        self.n_rows = len(df)
        self.n_features = len(self.numeric_cols)

    def determine_strategy(self) -> IterationStrategy:
        """Pick iteration strategy based on engine tier and dataset size."""
        # SLOW engines: Always single test (user already selected variable)
        if self.tier == EngineTier.SLOW:
            return IterationStrategy.SINGLE

        # FAST engines: Full iteration based on dataset size
        if self.tier == EngineTier.FAST:
            if self.n_rows <= 100 and self.n_features <= 10:
                return IterationStrategy.EXHAUSTIVE
            elif self.n_rows <= 500 and self.n_features <= 15:
                return IterationStrategy.GREEDY
            elif self.n_rows <= 2000:
                return IterationStrategy.SAMPLED
            else:
                return IterationStrategy.QUICK

        # MEDIUM engines: More conservative iteration
        if self.n_rows <= 200 and self.n_features <= 8:
            return IterationStrategy.GREEDY
        else:
            return IterationStrategy.QUICK

    def estimate_time_per_iteration(self) -> float:
        """Estimate milliseconds per iteration based on data size."""
        # Scale based on rows and features
        row_factor = self.n_rows / 100
        feat_factor = self.n_features / 5

        # Use tier-based timing estimates (more accurate than old multipliers)
        avg_time_s = self.tier_info.get("avg_time_seconds", 10.0)
        return avg_time_s * 1000  # Convert to ms

    def calculate_max_iterations(self, time_budget: float) -> int:
        """Calculate how many iterations fit within time budget, respecting tier limits."""
        # Get tier-based max iterations
        tier_max = self.tier_info.get("max_auto_iterations", 10)

        # SLOW engines: Always 1 iteration (user already picked variable)
        if self.tier == EngineTier.SLOW:
            return 1

        # Calculate time-based limit
        time_per_iter_ms = self.estimate_time_per_iteration()
        time_budget_ms = time_budget * 1000
        time_based_max = int(time_budget_ms / time_per_iter_ms) if time_per_iter_ms > 0 else tier_max

        # Apply strategy-specific caps
        strategy = self.determine_strategy()
        if strategy == IterationStrategy.SINGLE:
            return 1
        elif strategy == IterationStrategy.EXHAUSTIVE:
            total_combos = 2**self.n_features - 1
            return min(total_combos, time_based_max, tier_max, 127)
        elif strategy == IterationStrategy.GREEDY:
            return min(self.n_features * 5, time_based_max, tier_max, 50)
        elif strategy == IterationStrategy.SAMPLED:
            return min(self.n_features * 3, time_based_max, tier_max, 30)
        else:  # QUICK
            return min(self.n_features, time_based_max, tier_max, 15)

    def generate_feature_combinations(self, max_combos: int) -> Generator[list[str], None, None]:
        """Generate feature combinations based on strategy."""
        strategy = self.determine_strategy()

        if strategy == IterationStrategy.SINGLE:
            # For slow engines: use ALL features in single test
            yield self.numeric_cols
        elif strategy == IterationStrategy.EXHAUSTIVE:
            yield from self._exhaustive_combinations(max_combos)
        elif strategy == IterationStrategy.GREEDY:
            yield from self._greedy_combinations(max_combos)
        elif strategy == IterationStrategy.SAMPLED:
            yield from self._sampled_combinations(max_combos)
        else:
            yield from self._quick_combinations(max_combos)

    def _exhaustive_combinations(self, max_combos: int) -> Generator[list[str], None, None]:
        """Try all possible feature combinations."""
        count = 0
        # Start with larger combinations (more likely to be predictive)
        for r in range(self.n_features, 0, -1):
            for combo in combinations(self.numeric_cols, r):
                if count >= max_combos:
                    return
                yield list(combo)
                count += 1

    def _greedy_combinations(self, max_combos: int) -> Generator[list[str], None, None]:
        """Forward selection: start with best single feature, add incrementally."""
        # First, try each feature individually
        for feat in self.numeric_cols:
            yield [feat]

        # Then try pairs with the best single feature (determined after first pass)
        # This is a simplified greedy - full implementation would use results
        count = self.n_features
        for r in range(2, min(5, self.n_features + 1)):
            for combo in combinations(self.numeric_cols, r):
                if count >= max_combos:
                    return
                yield list(combo)
                count += 1

    def _sampled_combinations(self, max_combos: int) -> Generator[list[str], None, None]:
        """Random sampling of feature combinations."""
        import random

        random.seed(42)  # Reproducibility

        # Generate a pool of random combinations
        for _ in range(max_combos):
            # Random subset size (favor 3-5 features)
            size = random.choice([2, 3, 3, 4, 4, 5, min(6, self.n_features)])
            if size > self.n_features:
                size = self.n_features
            combo = random.sample(self.numeric_cols, size)
            yield combo

    def _quick_combinations(self, max_combos: int) -> Generator[list[str], None, None]:
        """Quick heuristics: try obvious combinations first."""
        # 1. All features
        yield self.numeric_cols

        # 2. Each feature individually
        for feat in self.numeric_cols[: max_combos - 1]:
            yield [feat]

    def iterate(
        self, max_iterations: int | None = None, time_budget: float = 30.0, engine_runner: callable | None = None
    ) -> SmartIterateResult:
        """
        Run iterations to find best prediction.

        Args:
            max_iterations: Maximum iterations (auto-calculated if None)
            time_budget: Max seconds to spend
            engine_runner: Function(df, target, features) -> score

        Returns:
            SmartIterateResult with best result, all iterations, and metadata
        """
        start_time = time.time()

        # Calculate max iterations if not specified
        if max_iterations is None:
            max_iterations = self.calculate_max_iterations(time_budget)

        # Get feature combinations generator
        combos = list(self.generate_feature_combinations(max_iterations))
        remaining_combos = combos[max_iterations:] if len(combos) > max_iterations else []
        combos_to_try = combos[:max_iterations]

        results: list[IterationResult] = []

        for idx, features in enumerate(combos_to_try):
            iter_start = time.time()

            # Check time budget
            elapsed = time.time() - start_time
            if elapsed >= time_budget:
                logger.info(f"Time budget exhausted after {idx} iterations")
                remaining_combos = combos_to_try[idx:] + remaining_combos
                break

            try:
                # Run the engine
                if engine_runner:
                    score, score_type, model_info = engine_runner(self.df, self.target, features)
                else:
                    # Default: simple correlation as placeholder
                    score, score_type, model_info = self._default_scorer(features)

                iter_time_ms = (time.time() - iter_start) * 1000

                results.append(
                    IterationResult(
                        rank=0,  # Will be set after sorting
                        score=score,
                        score_type=score_type,
                        features_used=features,
                        time_ms=iter_time_ms,
                        model_info=model_info,
                    )
                )

            except Exception as e:
                self.failed_attempts.append({"features": features, "error": str(e)})
                logger.warning(f"Iteration failed for {features}: {e}")

        # Sort by score (descending) and assign ranks
        results.sort(key=lambda x: x.score, reverse=True)
        for i, result in enumerate(results):
            result.rank = i + 1

        total_time = time.time() - start_time

        return SmartIterateResult(
            best=results[0] if results else None,
            all_iterations=results,
            failed_attempts=self.failed_attempts,
            remaining_combos=remaining_combos,
            metadata={
                "strategy": self.determine_strategy().value,
                "total_time_seconds": total_time,
                "iterations_completed": len(results),
                "iterations_failed": len(self.failed_attempts),
                "time_per_iteration_ms": (total_time * 1000) / len(results) if results else 0,
                "dataset_size": {"rows": self.n_rows, "cols": self.n_features},
                "can_iterate_more": len(remaining_combos) > 0,
            },
        )

    def iterate_remaining(
        self, remaining_combos: list[list[str]], max_iterations: int = 20, engine_runner: callable | None = None
    ) -> SmartIterateResult:
        """Continue iterating from remaining combinations."""
        combos_to_try = remaining_combos[:max_iterations]
        new_remaining = remaining_combos[max_iterations:]

        start_time = time.time()
        results: list[IterationResult] = []

        for features in combos_to_try:
            iter_start = time.time()
            try:
                if engine_runner:
                    score, score_type, model_info = engine_runner(self.df, self.target, features)
                else:
                    score, score_type, model_info = self._default_scorer(features)

                iter_time_ms = (time.time() - iter_start) * 1000

                results.append(
                    IterationResult(
                        rank=0,
                        score=score,
                        score_type=score_type,
                        features_used=features,
                        time_ms=iter_time_ms,
                        model_info=model_info,
                    )
                )
            except Exception as e:
                self.failed_attempts.append({"features": features, "error": str(e)})

        # Sort and rank
        results.sort(key=lambda x: x.score, reverse=True)
        for i, result in enumerate(results):
            result.rank = i + 1

        # Check if any new result beats cached best
        all_results = self.results_cache + results
        all_results.sort(key=lambda x: x.score, reverse=True)
        new_best = all_results[0] if all_results else None

        self.results_cache = all_results

        return SmartIterateResult(
            best=new_best,
            all_iterations=results,
            failed_attempts=self.failed_attempts,
            remaining_combos=new_remaining,
            metadata={
                "total_time_seconds": time.time() - start_time,
                "iterations_completed": len(results),
                "can_iterate_more": len(new_remaining) > 0,
            },
        )

    def _default_scorer(self, features: list[str]) -> tuple[float, str, dict]:
        """Default scorer using simple R² calculation."""
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import cross_val_score

        X = self.df[features].fillna(0).values
        y = self.df[self.target].fillna(0).values

        model = LinearRegression()
        scores = cross_val_score(model, X, y, cv=min(5, len(X) // 5), scoring="r2")

        return float(np.mean(scores)), "r2", {"model": "LinearRegression"}
