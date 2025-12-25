"""
Standard Engine Base Class

Abstract base class for standard (non-premium) engines that provides
consistent interface matching the premium engine architecture.

This enables all engines to be invoked and registered uniformly,
while maintaining backward compatibility with existing analyze() methods.
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd

from .premium_models import (
    Confidence,
    ConfigParameter,
    ExplanationStep,
    FeatureImportance,
    PlainEnglishSummary,
    PremiumResult,
    TaskType,
    TechnicalExplanation,
    Variant,
)

logger = logging.getLogger(__name__)


class StandardEngineBase(ABC):
    """
    Abstract base class for standard (non-premium) engines.

    Provides a unified interface that matches the premium engine pattern
    while allowing simpler implementation requirements.

    Subclasses must implement:
    - analyze(): Core analysis logic (existing method)
    - get_engine_info(): Display name, icon, task type

    Optional overrides:
    - get_config_schema(): Parameter definitions
    - get_methodology_info(): Technical documentation
    - get_requirements(): Data requirements

    Base class provides:
    - run_standard_analysis(): Wrapper that produces PremiumResult-compatible output
    - Timing utilities
    - Error handling
    - Result conversion
    """

    def __init__(self):
        """Initialize the standard engine."""
        self._start_time = None
        self._execution_time = 0.0

    # =========================================================================
    # ABSTRACT METHODS - Must be implemented by each engine
    # =========================================================================

    @abstractmethod
    def analyze(self, df: pd.DataFrame, config: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Core analysis logic for the engine.

        Args:
            df: Input DataFrame
            config: Configuration options

        Returns:
            Dict with engine-specific results
        """
        pass

    @abstractmethod
    def get_engine_info(self) -> dict[str, str]:
        """
        Get engine display information.

        Returns:
            Dict with:
            - name: Internal name (e.g., "statistical")
            - display_name: Human-readable name
            - icon: Emoji icon
            - task_type: TaskType value string
        """
        pass

    # =========================================================================
    # OPTIONAL OVERRIDES
    # =========================================================================

    def get_config_schema(self) -> list[ConfigParameter]:
        """
        Get the configuration schema for this engine.
        Override in subclasses for custom parameters.

        Returns:
            List of ConfigParameter objects defining tunable parameters
        """
        return []

    def get_methodology_info(self) -> dict[str, Any]:
        """
        Get methodology documentation.
        Override in subclasses for detailed methodology.

        Returns:
            Dict with methodology information
        """
        info = self.get_engine_info()
        return {
            "name": info.get("display_name", "Standard Analysis"),
            "url": None,
            "steps": [
                {"step_number": 1, "title": "Data Loading", "description": "Load and validate input data"},
                {"step_number": 2, "title": "Analysis Execution", "description": "Run core analysis algorithm"},
                {"step_number": 3, "title": "Result Generation", "description": "Format and return results"},
            ],
            "limitations": ["Results depend on data quality"],
            "assumptions": [],
        }

    def get_requirements(self) -> dict[str, Any]:
        """
        Get data requirements for this engine.

        Returns:
            Dict with min_rows, min_numeric_cols, etc.
        """
        return {"min_rows": 5, "min_numeric_cols": 1, "requires_target": False}

    # =========================================================================
    # MAIN ENTRY POINT
    # =========================================================================

    def run_standard_analysis(
        self, df: pd.DataFrame, target_column: str | None = None, config: dict[str, Any] | None = None
    ) -> PremiumResult:
        """
        Run analysis and return PremiumResult-compatible output.

        This wrapper method provides consistent output format
        matching the premium engine interface.

        Args:
            df: Input DataFrame
            target_column: Optional target column
            config: Configuration options

        Returns:
            PremiumResult with standardized structure
        """
        self._start_time = time.time()

        # Get engine info
        engine_info = self.get_engine_info()

        # Prepare config
        full_config = config or {}
        if target_column:
            full_config["target"] = target_column

        try:
            # Run core analysis
            raw_results = self.analyze(df, full_config)

            # Convert to PremiumResult format
            result = self._convert_to_premium_result(raw_results, df, target_column, engine_info, full_config)

        except Exception as e:
            logger.error(f"Engine {engine_info.get('name', 'unknown')} failed: {e}")
            result = self._create_error_result(str(e), df, engine_info, full_config)

        self._execution_time = time.time() - self._start_time
        result.execution_time_seconds = self._execution_time

        return result

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _convert_to_premium_result(
        self,
        raw_results: dict[str, Any],
        df: pd.DataFrame,
        target_column: str | None,
        engine_info: dict[str, str],
        config: dict[str, Any],
    ) -> PremiumResult:
        """Convert raw engine results to PremiumResult format."""

        # Extract or generate variants
        variants = self._extract_variants(raw_results)
        best_variant = variants[0] if variants else self._create_default_variant(raw_results)

        # Extract feature importance if available
        feature_importance = self._extract_feature_importance(raw_results)

        # Build summary
        summary = self._build_summary(raw_results, engine_info)

        # Build technical explanation
        explanation = self._build_explanation(engine_info)

        return PremiumResult(
            engine_name=engine_info.get("name", "standard"),
            engine_display_name=engine_info.get("display_name", "Standard Analysis"),
            engine_icon=engine_info.get("icon", "📊"),
            task_type=TaskType(engine_info.get("task_type", "detection")),
            target_column=target_column,
            columns_analyzed=list(df.columns),
            row_count=len(df),
            variants=variants,
            best_variant=best_variant,
            feature_importance=feature_importance,
            summary=summary,
            explanation=explanation,
            holdout=None,
            config_used=config,
            config_schema=self.get_config_schema(),
            execution_time_seconds=0.0,  # Will be set by caller
            warnings=raw_results.get("warnings", []),
        )

    def _extract_variants(self, results: dict[str, Any]) -> list[Variant]:
        """Extract variants from results if available."""
        if "variants" in results:
            raw_variants = results["variants"]
            variants = []
            for i, rv in enumerate(raw_variants[:10]):
                variant = Variant(
                    rank=i + 1,
                    gemma_score=rv.get("score", 50),
                    cv_score=rv.get("cv_score", 0.5),
                    variant_type=rv.get("type", "analysis"),
                    model_name=rv.get("name", f"Variant {i + 1}"),
                    features_used=rv.get("features", []),
                    interpretation=rv.get("interpretation", "Analysis result"),
                    details=rv.get("details", {}),
                )
                variants.append(variant)
            return variants

        # Create single variant from results
        return [self._create_default_variant(results)]

    def _create_default_variant(self, results: dict[str, Any]) -> Variant:
        """Create a default variant from results."""
        # Try to extract a meaningful score
        score = 0.5
        if "summary" in results:
            summary = results["summary"]
            if isinstance(summary, dict):
                score = summary.get("quality_score", summary.get("confidence", 50)) / 100

        return Variant(
            rank=1,
            gemma_score=int(score * 100),
            cv_score=score,
            variant_type="primary",
            model_name="Default",
            features_used=list(results.get("columns_analyzed", [])),
            interpretation="Primary analysis result",
            details={"raw_keys": list(results.keys())},
        )

    def _extract_feature_importance(self, results: dict[str, Any]) -> list[FeatureImportance]:
        """Extract feature importance from results if available."""
        feature_importance = []

        # Check common patterns for feature importance
        if "feature_importance" in results:
            fi = results["feature_importance"]
            if isinstance(fi, dict):
                for name, importance in fi.items():
                    feature_importance.append(
                        FeatureImportance(
                            name=str(name),
                            stability=80.0,
                            importance=float(importance) if isinstance(importance, (int, float)) else 0.5,
                            impact="positive",
                            explanation=f"Feature {name} analysis",
                        )
                    )
            elif isinstance(fi, list):
                for item in fi:
                    if isinstance(item, dict):
                        feature_importance.append(
                            FeatureImportance(
                                name=item.get("name", "unknown"),
                                stability=item.get("stability", 80.0),
                                importance=item.get("importance", 0.5),
                                impact=item.get("impact", "positive"),
                                explanation=item.get("explanation", ""),
                            )
                        )

        # Check correlation results
        if "correlation" in results and isinstance(results["correlation"], dict):
            corr_data = results["correlation"]
            if "matrix" in corr_data:
                # Extract top correlations
                pass

        return feature_importance

    def _build_summary(self, results: dict[str, Any], engine_info: dict[str, str]) -> PlainEnglishSummary:
        """Build a plain English summary from results."""
        # Try to extract summary from results
        if "summary" in results and isinstance(results["summary"], dict):
            raw_summary = results["summary"]
            headline = raw_summary.get("title", raw_summary.get("headline", ""))
            explanation = raw_summary.get("overview", raw_summary.get("explanation", ""))
            recommendation = raw_summary.get("recommendation", "Review the detailed results.")
        else:
            headline = f"{engine_info.get('display_name', 'Analysis')} Complete"
            explanation = "The analysis has been completed successfully."
            recommendation = "Review the detailed results for insights."

        # Determine confidence
        if "insights" in results and len(results.get("insights", [])) > 5:
            confidence = Confidence.HIGH
        elif "error" in results:
            confidence = Confidence.LOW
        else:
            confidence = Confidence.MEDIUM

        return PlainEnglishSummary(
            headline=headline, explanation=explanation, recommendation=recommendation, confidence=confidence
        )

    def _build_explanation(self, engine_info: dict[str, str]) -> TechnicalExplanation:
        """Build technical explanation from methodology info."""
        method_info = self.get_methodology_info()

        steps = []
        for step_data in method_info.get("steps", []):
            if isinstance(step_data, dict):
                steps.append(
                    ExplanationStep(
                        step_number=step_data.get("step_number", 1),
                        title=step_data.get("title", "Step"),
                        description=step_data.get("description", ""),
                        details=step_data.get("details"),
                    )
                )

        return TechnicalExplanation(
            methodology_name=method_info.get("name", "Standard Analysis"),
            methodology_url=method_info.get("url"),
            steps=steps,
            limitations=method_info.get("limitations", []),
            assumptions=method_info.get("assumptions", []),
        )

    def _create_error_result(
        self, error_message: str, df: pd.DataFrame, engine_info: dict[str, str], config: dict[str, Any]
    ) -> PremiumResult:
        """Create an error result when analysis fails."""
        error_variant = Variant(
            rank=1,
            gemma_score=0,
            cv_score=0.0,
            variant_type="error",
            model_name="None",
            features_used=[],
            interpretation=f"Analysis failed: {error_message}",
            details={"error": error_message},
        )

        return PremiumResult(
            engine_name=engine_info.get("name", "standard"),
            engine_display_name=engine_info.get("display_name", "Standard Analysis"),
            engine_icon=engine_info.get("icon", "❌"),
            task_type=TaskType(engine_info.get("task_type", "detection")),
            target_column=None,
            columns_analyzed=list(df.columns),
            row_count=len(df),
            variants=[error_variant],
            best_variant=error_variant,
            feature_importance=[],
            summary=PlainEnglishSummary(
                headline="Analysis Failed",
                explanation=error_message,
                recommendation="Check data format and try again.",
                confidence=Confidence.LOW,
            ),
            explanation=TechnicalExplanation(
                methodology_name="Error",
                methodology_url=None,
                steps=[],
                limitations=["Analysis could not complete"],
                assumptions=[],
            ),
            holdout=None,
            config_used=config,
            config_schema=[],
            execution_time_seconds=0.0,
            warnings=[error_message],
        )


# =============================================================================
# MIXINS FOR COMMON FUNCTIONALITY
# =============================================================================


class TimingMixin:
    """Mixin for execution timing."""

    def start_timer(self):
        """Start the execution timer."""
        self._timer_start = time.time()

    def get_elapsed_time(self) -> float:
        """Get elapsed time since timer start."""
        if hasattr(self, "_timer_start"):
            return time.time() - self._timer_start
        return 0.0


class ValidationMixin:
    """Mixin for data validation."""

    def validate_dataframe(
        self, df: pd.DataFrame, min_rows: int = 5, min_numeric_cols: int = 1, required_columns: list[str] | None = None
    ) -> dict[str, Any]:
        """
        Validate DataFrame meets requirements.

        Returns:
            Dict with 'valid' bool and 'errors' list
        """
        errors = []

        if len(df) < min_rows:
            errors.append(f"Dataset has {len(df)} rows, minimum is {min_rows}")

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < min_numeric_cols:
            errors.append(f"Dataset has {len(numeric_cols)} numeric columns, minimum is {min_numeric_cols}")

        if required_columns:
            missing = [c for c in required_columns if c not in df.columns]
            if missing:
                errors.append(f"Missing required columns: {missing}")

        return {"valid": len(errors) == 0, "errors": errors}


class LoggingMixin:
    """Mixin for consistent logging."""

    def log_start(self, operation: str):
        """Log operation start."""
        engine_name = getattr(self, "name", "Engine")
        logger.info(f"[{engine_name}] Starting: {operation}")

    def log_complete(self, operation: str, duration: float = None):
        """Log operation completion."""
        engine_name = getattr(self, "name", "Engine")
        if duration:
            logger.info(f"[{engine_name}] Completed: {operation} ({duration:.2f}s)")
        else:
            logger.info(f"[{engine_name}] Completed: {operation}")

    def log_error(self, operation: str, error: Exception):
        """Log operation error."""
        engine_name = getattr(self, "name", "Engine")
        logger.error(f"[{engine_name}] Failed: {operation} - {str(error)}")
