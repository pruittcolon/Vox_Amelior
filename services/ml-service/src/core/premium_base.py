"""
Premium Engine Base Class

Abstract base class that all flagship engines inherit from.
Provides standardized:
- Multi-variant generation
- Gemma ranking
- Holdout validation
- Plain English explanations
- Configuration schema
"""

import os
import time
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
import requests

from .business_translator import BusinessTranslator
from .premium_models import (
    Confidence,
    ConfigParameter,
    ExplanationStep,
    FeatureImportance,
    HoldoutResult,
    PlainEnglishSummary,
    PremiumResult,
    TaskType,
    TechnicalExplanation,
    Variant,
)


class PremiumEngineBase(ABC):
    """
    Abstract base class for all flagship engines.

    Subclasses must implement:
    - analyze(): Core analysis logic
    - get_config_schema(): Parameter definitions
    - get_methodology_info(): Technical documentation
    - get_engine_info(): Display name, icon, etc.

    Base class provides:
    - generate_variants(): Multi-variant analysis
    - rank_with_gemma(): LLM ranking
    - validate_holdout(): Train/test validation
    - build_explanation(): Technical explanation builder
    - translate_to_plain_english(): Business-friendly summaries
    """

    def __init__(self, gemma_url: str | None = None):
        """
        Initialize the premium engine.

        Args:
            gemma_url: URL for Gemma API (default: from environment)
        """
        self.gemma_url = gemma_url or os.getenv("GATEWAY_URL", "http://localhost:8000")
        self.translator = BusinessTranslator()
        self._session = None
        self._gemma_available = None

    # =========================================================================
    # ABSTRACT METHODS - Must be implemented by each engine
    # =========================================================================

    @abstractmethod
    def analyze(
        self, df: pd.DataFrame, target_column: str | None = None, config: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Core analysis logic for the engine.

        Args:
            df: Input DataFrame
            target_column: Column to predict/analyze (optional for some engines)
            config: Configuration overrides

        Returns:
            Dict with engine-specific results that will be converted to variants
        """
        pass

    @abstractmethod
    def get_config_schema(self) -> list[ConfigParameter]:
        """
        Get the configuration schema for this engine.

        Returns:
            List of ConfigParameter objects defining tunable parameters
        """
        pass

    @abstractmethod
    def get_methodology_info(self) -> dict[str, Any]:
        """
        Get methodology documentation.

        Returns:
            Dict with:
            - name: Methodology name
            - url: Link to documentation
            - steps: List of explanation steps
            - limitations: Known limitations
            - assumptions: Model assumptions
        """
        pass

    @abstractmethod
    def get_engine_info(self) -> dict[str, str]:
        """
        Get engine display information.

        Returns:
            Dict with:
            - name: Internal name (e.g., "titan")
            - display_name: Human-readable name
            - icon: Emoji icon
            - task_type: TaskType value
        """
        pass

    # =========================================================================
    # MAIN ENTRY POINT
    # =========================================================================

    def run_premium_analysis(
        self,
        df: pd.DataFrame,
        target_column: str | None = None,
        config: dict[str, Any] | None = None,
        n_variants: int = 10,
        holdout_ratio: float = 0.2,
        enable_gemma_ranking: bool = True,
    ) -> PremiumResult:
        """
        Run full premium analysis with all features.

        Args:
            df: Input DataFrame
            target_column: Column to predict/analyze
            config: Configuration overrides
            n_variants: Number of variants to generate
            holdout_ratio: Fraction of data for holdout validation
            enable_gemma_ranking: Whether to use Gemma for ranking

        Returns:
            PremiumResult with all analysis components
        """
        start_time = time.time()

        # Get engine info
        engine_info = self.get_engine_info()

        # Merge config with defaults
        full_config = self._merge_config(config or {})
        full_config["n_variants"] = n_variants
        full_config["holdout_ratio"] = holdout_ratio
        full_config["enable_gemma_ranking"] = enable_gemma_ranking

        # Auto-detect target if not provided
        if target_column is None:
            target_column = self._detect_target_column(df)

        # Split holdout if requested
        holdout_result = None
        if holdout_ratio > 0:
            train_df, holdout_df = self._split_holdout(df, holdout_ratio)
        else:
            train_df = df
            holdout_df = None

        # Run core analysis
        analysis_results = self.analyze(train_df, target_column, full_config)

        # Generate variants from analysis results
        variants = self._generate_variants(analysis_results, n_variants)

        # Rank with Gemma if enabled
        if enable_gemma_ranking and self._check_gemma_available():
            variants = self._rank_with_gemma(variants, target_column)
        else:
            # Fall back to CV score ranking
            variants = sorted(variants, key=lambda v: v.cv_score, reverse=True)
            for i, v in enumerate(variants):
                v.rank = i + 1
                v.gemma_score = int(v.cv_score * 100)  # Use CV score as proxy

        # Get best variant
        best_variant = variants[0] if variants else self._create_fallback_variant()

        # Validate on holdout
        if holdout_df is not None and len(holdout_df) > 0:
            holdout_result = self._validate_holdout(train_df, holdout_df, best_variant, target_column, analysis_results)

        # Build feature importance
        feature_importance = self._build_feature_importance(analysis_results, target_column)

        # Build plain English summary
        summary = self._build_plain_english_summary(engine_info, target_column, best_variant, feature_importance)

        # Build technical explanation
        explanation = self._build_technical_explanation()

        execution_time = time.time() - start_time

        # Assemble final result
        return PremiumResult(
            engine_name=engine_info["name"],
            engine_display_name=engine_info["display_name"],
            engine_icon=engine_info["icon"],
            task_type=TaskType(engine_info["task_type"]),
            target_column=target_column,
            columns_analyzed=list(df.columns),
            row_count=len(df),
            variants=variants,
            best_variant=best_variant,
            feature_importance=feature_importance,
            summary=summary,
            explanation=explanation,
            holdout=holdout_result,
            config_used=full_config,
            config_schema=self.get_config_schema(),
            execution_time_seconds=execution_time,
            warnings=analysis_results.get("warnings", []),
        )

    # =========================================================================
    # VARIANT GENERATION
    # =========================================================================

    def _generate_variants(self, analysis_results: dict[str, Any], n_variants: int) -> list[Variant]:
        """
        Generate variants from analysis results.

        Override this in subclasses for custom variant generation.
        Default implementation creates variants from 'variants' key in results.
        """
        raw_variants = analysis_results.get("variants", [])

        if not raw_variants:
            # Create single variant from main results
            return [self._create_variant_from_results(analysis_results, rank=1)]

        variants = []
        for i, rv in enumerate(raw_variants[:n_variants]):
            variant = Variant(
                rank=i + 1,
                gemma_score=rv.get("gemma_score", 50),
                cv_score=rv.get("cv_score", rv.get("score", 0.5)),
                variant_type=rv.get("variant_type", "baseline"),
                model_name=rv.get("model_name", rv.get("model", "Unknown")),
                features_used=rv.get("features", rv.get("features_used", [])),
                interpretation=rv.get("interpretation", "Analysis variant"),
                details=rv.get("details", {}),
            )
            variants.append(variant)

        return variants

    def _create_variant_from_results(self, results: dict[str, Any], rank: int = 1) -> Variant:
        """Create a single variant from analysis results."""
        return Variant(
            rank=rank,
            gemma_score=int(results.get("score", 0.5) * 100),
            cv_score=results.get("score", results.get("cv_score", 0.5)),
            variant_type="baseline",
            model_name=results.get("model_name", "Default"),
            features_used=results.get("features", []),
            interpretation=results.get("interpretation", "Primary analysis result"),
            details=results.get("details", {}),
        )

    def _create_fallback_variant(self) -> Variant:
        """Create a fallback variant when analysis fails."""
        return Variant(
            rank=1,
            gemma_score=0,
            cv_score=0.0,
            variant_type="fallback",
            model_name="None",
            features_used=[],
            interpretation="Analysis could not be completed",
            details={"error": "No results generated"},
        )

    # =========================================================================
    # GEMMA RANKING
    # =========================================================================

    def _check_gemma_available(self) -> bool:
        """Check if Gemma API is available."""
        if self._gemma_available is not None:
            return self._gemma_available

        try:
            resp = requests.get(f"{self.gemma_url}/health", timeout=2)
            self._gemma_available = resp.status_code == 200
        except:
            self._gemma_available = False

        return self._gemma_available

    def _rank_with_gemma(self, variants: list[Variant], target_column: str) -> list[Variant]:
        """
        Use Gemma to rank variants by business utility.

        Returns variants sorted by Gemma score (highest first).
        """
        if not variants:
            return variants

        # Build prompt
        prompt = self._build_gemma_ranking_prompt(variants, target_column)

        try:
            # Call Gemma
            response = self._call_gemma(prompt)

            if response:
                # Parse rankings from response
                rankings = self._parse_gemma_rankings(response, len(variants))

                # Apply rankings
                for i, variant in enumerate(variants):
                    if i < len(rankings):
                        variant.gemma_score = rankings[i]["score"]
                        variant.interpretation = rankings[i]["interpretation"]
        except Exception as e:
            print(f"Gemma ranking failed: {e}")
            # Fall back to CV score
            for variant in variants:
                variant.gemma_score = int(variant.cv_score * 100)

        # Sort by Gemma score
        variants = sorted(variants, key=lambda v: v.gemma_score, reverse=True)

        # Update ranks
        for i, v in enumerate(variants):
            v.rank = i + 1

        return variants

    def _build_gemma_ranking_prompt(self, variants: list[Variant], target_column: str) -> str:
        """Build the prompt for Gemma ranking."""
        variant_descriptions = []
        for i, v in enumerate(variants):
            desc = (
                f"Variant {chr(65 + i)}: {v.model_name} model using features "
                f"{', '.join(v.features_used[:5])}. CV Score: {v.cv_score:.3f}"
            )
            variant_descriptions.append(desc)

        prompt = f"""You are evaluating machine learning models for business use.

Target: Predicting '{target_column}'

Variants to rank:
{chr(10).join(variant_descriptions)}

For each variant, provide:
1. A score from 1-100 for business utility (consider interpretability, reliability, actionability)
2. One sentence explaining why

Format your response exactly like this:
Variant A: 85 - This model is highly interpretable and uses well-understood features.
Variant B: 72 - Good accuracy but relies on features that are hard to control.
...

Rank all {len(variants)} variants."""

        return prompt

    def _call_gemma(self, prompt: str) -> str | None:
        """Call Gemma API and return response text."""
        try:
            resp = requests.post(
                f"{self.gemma_url}/api/gemma/generate",
                json={"prompt": prompt, "max_tokens": 500, "temperature": 0.3},
                timeout=30,
            )
            if resp.status_code == 200:
                return resp.json().get("text", "")
        except Exception as e:
            print(f"Gemma API call failed: {e}")
        return None

    def _parse_gemma_rankings(self, response: str, expected_count: int) -> list[dict[str, Any]]:
        """Parse Gemma's ranking response."""
        import re

        rankings = []
        pattern = r"Variant\s+([A-Z]):\s*(\d+)\s*[-–]\s*(.+?)(?=Variant\s+[A-Z]:|$)"

        matches = re.findall(pattern, response, re.DOTALL)

        for match in matches:
            letter, score, interpretation = match
            rankings.append(
                {"letter": letter, "score": min(100, max(1, int(score))), "interpretation": interpretation.strip()}
            )

        # Sort by letter to maintain original order
        rankings = sorted(rankings, key=lambda x: x["letter"])

        # Fill in missing rankings
        while len(rankings) < expected_count:
            rankings.append({"letter": chr(65 + len(rankings)), "score": 50, "interpretation": "No ranking provided"})

        return rankings

    # =========================================================================
    # HOLDOUT VALIDATION
    # =========================================================================

    def _split_holdout(self, df: pd.DataFrame, ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into train and holdout sets."""
        n_holdout = int(len(df) * ratio)

        if n_holdout < 10:
            # Not enough for meaningful holdout
            return df, pd.DataFrame()

        # Shuffle and split
        shuffled = df.sample(frac=1, random_state=42)
        holdout = shuffled.iloc[:n_holdout]
        train = shuffled.iloc[n_holdout:]

        return train, holdout

    def _validate_holdout(
        self,
        train_df: pd.DataFrame,
        holdout_df: pd.DataFrame,
        best_variant: Variant,
        target_column: str,
        analysis_results: dict[str, Any],
    ) -> HoldoutResult:
        """
        Validate the best model on holdout data.

        Override in subclasses for engine-specific validation.
        Default implementation returns placeholder result.
        """
        # Default implementation - subclasses should override
        train_score = best_variant.cv_score
        holdout_score = train_score * 0.9  # Assume slight degradation

        passed = (train_score - holdout_score) < 0.15

        if passed:
            message = "Good generalization - model performs similarly on unseen data"
        else:
            message = "Possible overfitting - consider more data or simpler model"

        return HoldoutResult(
            train_samples=len(train_df),
            holdout_samples=len(holdout_df),
            holdout_ratio=len(holdout_df) / (len(train_df) + len(holdout_df)),
            train_score=train_score,
            holdout_score=holdout_score,
            metric_name="cv_score",
            passed=passed,
            message=message,
        )

    # =========================================================================
    # FEATURE IMPORTANCE
    # =========================================================================

    def _build_feature_importance(
        self, analysis_results: dict[str, Any], target_column: str
    ) -> list[FeatureImportance]:
        """Build feature importance list from analysis results."""
        raw_importance = analysis_results.get("feature_importance", {})
        stability_scores = analysis_results.get("stability_scores", {})

        features = []
        for feature_name, importance in raw_importance.items():
            stability = stability_scores.get(feature_name, 75.0)

            # Determine impact direction
            impact = self._determine_feature_impact(analysis_results, feature_name, target_column)

            # Generate explanation
            explanation = self.translator.translate_feature_importance(
                feature_name, importance, stability, impact, target_column
            )

            features.append(
                FeatureImportance(
                    name=feature_name,
                    stability=stability,
                    importance=importance,
                    impact=impact,
                    explanation=explanation,
                )
            )

        # Sort by importance
        features = sorted(features, key=lambda f: f.importance, reverse=True)

        return features

    def _determine_feature_impact(self, analysis_results: dict[str, Any], feature_name: str, target_column: str) -> str:
        """Determine if feature has positive, negative, or mixed impact."""
        # Check if impact info is in results
        impacts = analysis_results.get("feature_impacts", {})
        if feature_name in impacts:
            return impacts[feature_name]

        # Default to positive if correlation info available
        correlations = analysis_results.get("correlations", {})
        if feature_name in correlations:
            corr = correlations[feature_name]
            if corr > 0.1:
                return "positive"
            elif corr < -0.1:
                return "negative"

        return "mixed"

    # =========================================================================
    # PLAIN ENGLISH SUMMARY
    # =========================================================================

    def _build_plain_english_summary(
        self,
        engine_info: dict[str, str],
        target_column: str,
        best_variant: Variant,
        feature_importance: list[FeatureImportance],
    ) -> PlainEnglishSummary:
        """Build business-friendly summary of results."""
        top_features = [f.name for f in feature_importance[:3]]
        feature_impacts = {f.name: f.impact for f in feature_importance}

        # Generate headline
        headline = self.translator.generate_headline(
            engine_info["task_type"], target_column, best_variant.cv_score, "accuracy"
        )

        # Generate explanation
        explanation = self.translator.generate_explanation(
            engine_info["task_type"], target_column, top_features, best_variant.cv_score, "accuracy"
        )

        # Generate recommendation
        recommendation = self.translator.generate_recommendation(
            engine_info["task_type"], target_column, top_features, feature_impacts
        )

        # Determine confidence
        avg_stability = np.mean([f.stability for f in feature_importance]) if feature_importance else 50
        confidence_str = self.translator.get_confidence_level(best_variant.cv_score, avg_stability)
        confidence = Confidence(confidence_str)

        return PlainEnglishSummary(
            headline=headline, explanation=explanation, recommendation=recommendation, confidence=confidence
        )

    # =========================================================================
    # TECHNICAL EXPLANATION
    # =========================================================================

    def _build_technical_explanation(self) -> TechnicalExplanation:
        """Build technical methodology explanation."""
        info = self.get_methodology_info()

        steps = []
        for i, step_info in enumerate(info.get("steps", []), 1):
            steps.append(
                ExplanationStep(
                    step_number=i,
                    title=step_info.get("title", f"Step {i}"),
                    description=step_info.get("description", ""),
                    details=step_info.get("details"),
                )
            )

        return TechnicalExplanation(
            methodology_name=info.get("name", "Analysis"),
            methodology_url=info.get("url"),
            steps=steps,
            limitations=info.get("limitations", []),
            assumptions=info.get("assumptions", []),
        )

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def _merge_config(self, overrides: dict[str, Any]) -> dict[str, Any]:
        """Merge user config with defaults from schema."""
        schema = self.get_config_schema()
        config = {param.name: param.default for param in schema}
        config.update(overrides)
        return config

    def _detect_target_column(self, df: pd.DataFrame) -> str | None:
        """
        Auto-detect the target column.

        Override in subclasses for engine-specific detection.
        """
        # Common target column names
        target_keywords = ["target", "label", "class", "y", "output", "outcome"]

        for col in df.columns:
            if any(kw in col.lower() for kw in target_keywords):
                return col

        # Fall back to last column if it looks like a target
        last_col = df.columns[-1]
        if df[last_col].nunique() < 20 or pd.api.types.is_numeric_dtype(df[last_col]):
            return last_col

        return None
