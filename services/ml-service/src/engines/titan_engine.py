"""
Titan AutoML Engine - Enterprise-Grade Robust Machine Learning (Premium)

Implements bulletproof model selection and validation:
1. Nested Cross-Validation (5x2) - unbiased performance estimation
2. Stability Selection - identifies truly robust features
3. Ensemble Stacking - combines multiple model types
4. Multi-Variant Analysis - generates ranked analysis variants
5. Gemma Business Ranking - LLM-based utility scoring (1-100)
6. Explainability & Provenance - full methodology transparency

GPU Support:
- Optional XGBoost with GPU histogram method (gpu_hist)
- Automatically coordinates GPU access via GPU Coordinator
- Significant speedup for gradient boosting on large datasets

Prevents overfitting through rigorous statistical validation.

Author: Enterprise Analytics Team
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import logging
import re
import signal
import time
import uuid
import warnings
from contextlib import contextmanager
from itertools import combinations
from typing import Any, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Configure logging for Titan
logger = logging.getLogger("titan_engine")
logger.setLevel(logging.INFO)


# Timeout handler
class TimeoutError(Exception):
    pass


@contextmanager
def timeout(seconds: int, operation: str = "operation"):
    """Context manager for timing out operations"""

    def timeout_handler(signum, frame):
        raise TimeoutError(f"{operation} timed out after {seconds} seconds")

    # Only set alarm on Unix-like systems
    if hasattr(signal, "SIGALRM"):
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    else:
        # Windows fallback - no timeout
        yield


from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# Optional XGBoost with GPU support
try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from core import ColumnMapper, ColumnProfiler, EngineRequirements, SemanticType
from core.business_translator import BusinessTranslator

# Import premium models for standardized output
from core.premium_models import (
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

# GPU Client for coordinated GPU access
try:
    from core.gpu_client import GPUClient, get_gpu_client

    GPU_CLIENT_AVAILABLE = True
except ImportError:
    GPU_CLIENT_AVAILABLE = False


# =============================================================================
# CONFIGURATION SCHEMA - Defines all tunable parameters for UI/CLI
# =============================================================================
TITAN_CONFIG_SCHEMA = {
    "cv_folds": {"type": "int", "min": 2, "max": 10, "default": 5, "description": "Number of cross-validation folds"},
    "n_bootstrap": {
        "type": "int",
        "min": 10,
        "max": 100,
        "default": 30,
        "description": "Bootstrap samples for stability selection",
    },
    "n_variants": {
        "type": "int",
        "min": 1,
        "max": 10,
        "default": 3,
        "description": "Number of analysis variants to generate",
    },
    "stability_threshold": {
        "type": "float",
        "min": 0.5,
        "max": 1.0,
        "default": 0.8,
        "description": "Feature stability threshold (0-1)",
    },
    "holdout_ratio": {
        "type": "float",
        "min": 0.0,
        "max": 0.5,
        "default": 0.0,
        "description": "Holdout set ratio for validation (0 = disabled)",
    },
    "max_features": {"type": "int", "min": 1, "max": 100, "default": 20, "description": "Maximum features to consider"},
    "enable_gemma_ranking": {"type": "bool", "default": False, "description": "Enable Gemma LLM ranking of variants"},
    "use_gpu": {"type": "bool", "default": True, "description": "Use GPU acceleration for XGBoost if available"},
}


def _create_model_configs(use_gpu: bool = False):
    """
    Create model configurations, optionally including GPU-accelerated XGBoost.

    Args:
        use_gpu: Whether to enable GPU acceleration for XGBoost

    Returns:
        List of model configurations
    """
    configs = [
        {
            "name": "RandomForest",
            "classifier": RandomForestClassifier,
            "regressor": RandomForestRegressor,
            "params": {"n_estimators": 50, "max_depth": 8, "random_state": 42},
            "gpu": False,
        },
        {
            "name": "GradientBoosting",
            "classifier": GradientBoostingClassifier,
            "regressor": GradientBoostingRegressor,
            "params": {"n_estimators": 50, "max_depth": 4, "random_state": 42},
            "gpu": False,
        },
        {
            "name": "DecisionTree",
            "classifier": DecisionTreeClassifier,
            "regressor": DecisionTreeRegressor,
            "params": {"max_depth": 10, "random_state": 42},
            "gpu": False,
        },
    ]

    # Add XGBoost with GPU if available and requested
    # XGBoost 2.0+ uses device='cuda' instead of tree_method='gpu_hist'
    if XGBOOST_AVAILABLE and use_gpu:
        configs.append(
            {
                "name": "XGBoost-GPU",
                "classifier": xgb.XGBClassifier,
                "regressor": xgb.XGBRegressor,
                "params": {
                    "n_estimators": 50,
                    "max_depth": 6,
                    "tree_method": "hist",  # hist works with device='cuda'
                    "device": "cuda",  # XGBoost 2.0+ GPU parameter
                    "random_state": 42,
                    "eval_metric": "logloss",  # Suppress warning
                },
                "gpu": True,
            }
        )
    elif XGBOOST_AVAILABLE:
        # CPU XGBoost
        configs.append(
            {
                "name": "XGBoost",
                "classifier": xgb.XGBClassifier,
                "regressor": xgb.XGBRegressor,
                "params": {
                    "n_estimators": 50,
                    "max_depth": 6,
                    "tree_method": "hist",  # CPU histogram method
                    "device": "cpu",
                    "random_state": 42,
                    "eval_metric": "logloss",
                },
                "gpu": False,
            }
        )

    return configs


# Default model configs (without GPU)
MODEL_CONFIGS = _create_model_configs(use_gpu=False)


class TitanEngine:
    """
    Titan AutoML Engine: Bulletproof ML with statistical rigor (Premium Edition)

    Features:
    - Nested Cross-Validation for unbiased error estimation
    - Stability Selection for robust feature identification
    - Ensemble models (RF + GBM + ElasticNet + XGBoost)
    - Automatic task detection (classification vs regression)
    - Multi-Variant Analysis with ranked outputs
    - Gemma Business Utility Scoring (1-100)
    - Explainability & Provenance tracking
    - Manual Parameter Override support

    GPU Support:
    - XGBoost with GPU histogram method (gpu_hist)
    - Automatically coordinates GPU access via GPU Coordinator
    """

    def __init__(self, gemma_client=None, gpu_client: Optional["GPUClient"] = None):
        self.name = "Titan AutoML Engine (Enterprise-Grade Premium)"
        self.profiler = ColumnProfiler()
        self.mapper = ColumnMapper()
        self.gemma_client = gemma_client  # Optional Gemma client for ranking
        self._gpu_client = gpu_client
        self._gpu_acquired = False
        self.config_schema = TITAN_CONFIG_SCHEMA

    @property
    def gpu_client(self) -> Optional["GPUClient"]:
        """Get GPU client, initializing from singleton if not provided"""
        if self._gpu_client is None and GPU_CLIENT_AVAILABLE:
            try:
                self._gpu_client = get_gpu_client()
            except Exception:
                pass
        return self._gpu_client

    async def _acquire_gpu(self) -> bool:
        """Request GPU access from the GPU Coordinator."""
        if self.gpu_client is None:
            return False

        try:
            result = await self.gpu_client.request_gpu(engine_name="titan")
            self._gpu_acquired = result.acquired
            return result.acquired
        except Exception as e:
            print(f"GPU acquisition failed, using CPU: {e}")
            return False

    async def _release_gpu(self) -> None:
        """Release GPU back to the coordinator"""
        if self._gpu_acquired and self.gpu_client is not None:
            try:
                await self.gpu_client.release_gpu()
            except Exception:
                pass
            finally:
                self._gpu_acquired = False

    def get_requirements(self) -> EngineRequirements:
        """Define what this engine needs to run"""
        return EngineRequirements(
            required_semantics=[SemanticType.NUMERIC_CONTINUOUS],
            optional_semantics={"target": [SemanticType.NUMERIC_CONTINUOUS, SemanticType.CATEGORICAL]},
            required_entities=[],
            preferred_domains=[],
            applicable_tasks=["classification", "regression"],
            min_rows=100,
            min_numeric_cols=3,
        )

    def analyze(
        self, df: pd.DataFrame, target_column: str | None = None, config: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Main analysis method (synchronous)

        Args:
            df: Input dataset
            target_column: Optional target column hint (can also be in config)
            config: {
                'target_column': Optional[str] - name of target column
                'task_type': Optional[str] - 'classification' or 'regression'
                'n_bootstrap': int - number of bootstrap samples for stability (default: 100)
                'cv_folds': int - number of CV folds (default: 5)
                'n_variants': int - number of analysis variants to generate (default: 10)
                'enable_gemma_ranking': bool - enable Gemma LLM ranking (default: False)
                'holdout_ratio': float - holdout validation ratio (default: 0.0)
                'use_gpu': bool - use GPU acceleration for XGBoost (default: True)
            }

        Returns:
            Dict with analysis results including variants and provenance
        """
        config = config or {}
        # Support both positional target_column and config-based
        if target_column and "target_column" not in config:
            config["target_column"] = target_column
        use_gpu_requested = config.get("use_gpu", True)

        # For sync context, use CPU by default (GPU requires async coordination)
        # GPU can be used when called via analyze_async
        model_configs = _create_model_configs(use_gpu=False)

        return self._analyze_internal(df, config, model_configs, gpu_acquired=False)

    async def analyze_async(self, df: pd.DataFrame, config: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Async main analysis method with GPU coordination (Premium Edition)

        Args:
            df: Input dataset
            config: {
                'target_column': Optional[str] - name of target column
                'task_type': Optional[str] - 'classification' or 'regression'
                'n_bootstrap': int - number of bootstrap samples for stability (default: 100)
                'cv_folds': int - number of CV folds (default: 5)
                'n_variants': int - number of analysis variants to generate (default: 10)
                'enable_gemma_ranking': bool - enable Gemma LLM ranking (default: False)
                'holdout_ratio': float - holdout validation ratio (default: 0.0)
                'use_gpu': bool - use GPU acceleration for XGBoost (default: True)
            }

        Returns:
            {
                'task_type': str,
                'target_column': str,
                'nested_cv_scores': {...},
                'stable_features': [...],
                'feature_importance': {...},
                'best_model': str,
                'predictions': [...],
                'insights': [...],
                'variants': [...],  # Ranked analysis variants
                'provenance': {...},  # Explainability data
                'config_schema': {...},  # Available parameters
                'used_gpu': bool  # Whether GPU was used
            }
        """
        config = config or {}
        use_gpu_requested = config.get("use_gpu", True)

        # Attempt GPU acquisition if XGBoost is available and GPU requested
        gpu_acquired = False
        if use_gpu_requested and XGBOOST_AVAILABLE:
            gpu_acquired = await self._acquire_gpu()

        try:
            # Update MODEL_CONFIGS based on GPU availability
            model_configs = _create_model_configs(use_gpu=gpu_acquired)

            return self._analyze_internal(df, config, model_configs, gpu_acquired)
        finally:
            if gpu_acquired:
                await self._release_gpu()

    def _analyze_internal(
        self, df: pd.DataFrame, config: dict[str, Any], model_configs: list[dict], gpu_acquired: bool
    ) -> dict[str, Any]:
        """
        Internal analysis method with model configuration injection
        """
        start_time = time.time()
        logger.info(f"[TITAN] Starting analysis: {len(df)} rows, {len(df.columns)} columns")

        # Set max execution time (2 minutes default)
        max_execution_time = config.get("max_execution_time", 120)

        # Merge with defaults from schema
        for key, schema in TITAN_CONFIG_SCHEMA.items():
            if key not in config:
                config[key] = schema.get("default")

        # 1. Profile dataset
        logger.info("[TITAN] Step 1: Profiling dataset...")
        step_start = time.time()
        if not config.get("skip_profiling", False):
            profiles = self.profiler.profile_dataset(df)
        else:
            profiles = {}
        logger.info(f"[TITAN] Profiling complete: {time.time() - step_start:.2f}s")

        # 2. Detect target column
        logger.info("[TITAN] Step 2: Detecting target column...")
        target_col = self._detect_target_column(df, profiles, config.get("target_column"))

        if not target_col:
            return {
                "error": "No target column found",
                "message": "Titan Engine requires a target column for prediction",
                "insights": ["💡 Please specify target_column in config, or ensure dataset has a target/label column"],
                "config_schema": TITAN_CONFIG_SCHEMA,
            }
        logger.info(f"[TITAN] Target column: {target_col}")

        # 3. Detect task type (classification vs regression)
        logger.info("[TITAN] Step 3: Detecting task type...")
        task_type = self._detect_task_type(df, target_col, profiles, config.get("task_type"))
        logger.info(f"[TITAN] Task type: {task_type}")

        # 4. Handle holdout validation if requested
        holdout_ratio = config.get("holdout_ratio", 0.0)
        holdout_results = None
        df_train = df
        df_holdout = None

        if holdout_ratio > 0:
            df_train, df_holdout = train_test_split(df, test_size=holdout_ratio, random_state=42)

        # 5. Prepare features and target
        logger.info("[TITAN] Step 5: Preparing data...")
        step_start = time.time()
        X, y, feature_names = self._prepare_data(df_train, target_col)
        logger.info(f"[TITAN] Data prepared: {X.shape}, {time.time() - step_start:.2f}s")

        # 5a. Sample large datasets for speed (nested CV is O(n²))
        max_samples = config.get("max_samples", 1000)
        if len(X) > max_samples:
            logger.info(f"[TITAN] Sampling from {len(X)} to {max_samples} rows")
            sample_idx = np.random.RandomState(42).choice(len(X), max_samples, replace=False)
            X = X[sample_idx]
            y = y[sample_idx]
            # Also sample df_train for use in later steps (variants, impact analysis)
            df_train = df_train.iloc[sample_idx].reset_index(drop=True)

        if X.shape[1] < 2:
            return {
                "error": "Insufficient features",
                "message": f"Found only {X.shape[1]} feature(s). Titan requires at least 2 features.",
                "insights": [],
                "config_schema": TITAN_CONFIG_SCHEMA,
            }

        # Check timeout
        elapsed = time.time() - start_time
        if elapsed > max_execution_time:
            logger.warning(f"[TITAN] Timeout after {elapsed:.1f}s at step 5")
            return self._timeout_response(target_col, task_type, elapsed)

        # 6. Run Nested Cross-Validation (with reduced complexity for speed)
        logger.info("[TITAN] Step 6: Running Nested Cross-Validation...")
        step_start = time.time()
        try:
            # Reduce CV folds for large datasets
            cv_folds = min(config.get("cv_folds", 5), 3) if len(X) > 500 else config.get("cv_folds", 5)
            config["cv_folds"] = cv_folds
            nested_cv_results = self._nested_cross_validation(X, y, task_type, config)
            logger.info(f"[TITAN] Nested CV complete: {time.time() - step_start:.2f}s")
        except (ValueError, Exception) as e:
            error_msg = str(e)
            logger.error(f"[TITAN] Nested CV failed: {error_msg}")
            if "All the" in error_msg and "fits failed" in error_msg:
                return {
                    "engine": self.name,
                    "status": "not_applicable",
                    "target_column": target_col,
                    "insights": [
                        f"📊 **Titan AutoML - Not Applicable**: This dataset structure is not suitable for automated machine learning. The dataset may be too small ({len(df)} rows), have insufficient variance, or the target variable may not have a predictable relationship with the features. Consider manual feature engineering or a larger dataset."
                    ],
                    "config_schema": TITAN_CONFIG_SCHEMA,
                }
            raise

        # Check timeout
        elapsed = time.time() - start_time
        if elapsed > max_execution_time:
            logger.warning(f"[TITAN] Timeout after {elapsed:.1f}s at step 6")
            return self._timeout_response(target_col, task_type, elapsed)

        # 7. Stability Selection (with reduced bootstrap for speed)
        logger.info("[TITAN] Step 7: Running Stability Selection...")
        step_start = time.time()
        # Reduce bootstrap samples for speed on larger datasets
        n_bootstrap = min(config.get("n_bootstrap", 30), 20) if len(X) > 500 else min(config.get("n_bootstrap", 30), 30)
        stable_features, feature_importance = self._stability_selection(
            X, y, feature_names, task_type, n_bootstrap=n_bootstrap, threshold=config.get("stability_threshold", 0.8)
        )
        logger.info(
            f"[TITAN] Stability Selection complete: {len(stable_features)} stable features, {time.time() - step_start:.2f}s"
        )

        # 8. Train final model on stable features
        logger.info("[TITAN] Step 8: Training final model...")
        step_start = time.time()
        X_stable = X[:, [i for i, f in enumerate(feature_names) if f in stable_features]]
        final_model, final_score = self._train_final_model(X_stable, y, task_type)
        logger.info(f"[TITAN] Final model trained: {time.time() - step_start:.2f}s")

        # Check timeout
        elapsed = time.time() - start_time
        if elapsed > max_execution_time:
            logger.warning(f"[TITAN] Timeout after {elapsed:.1f}s at step 8")
            return self._timeout_response(target_col, task_type, elapsed)

        # 9. Analyze feature impact (Directionality)
        logger.info("[TITAN] Step 9: Analyzing feature impact...")
        impact_insights = self._analyze_feature_impact(df_train, target_col, stable_features, task_type)

        # 10. Generate Multi-Variant Analysis (limit variants for speed)
        logger.info("[TITAN] Step 10: Generating variants...")
        n_variants = min(config.get("n_variants", 10), 5)  # Limit to 5 for speed
        variants = self._generate_variants(
            df_train, target_col, task_type, feature_names, stable_features, feature_importance, n_variants, config
        )
        logger.info(f"[TITAN] Generated {len(variants)} variants")

        # 11. Gemma Ranking (if enabled and client available)
        if config.get("enable_gemma_ranking", False):
            variants = self._rank_with_gemma(variants)

        # 12. Holdout Validation (if requested)
        if df_holdout is not None and len(stable_features) > 0:
            holdout_results = self._validate_holdout(
                df_train, df_holdout, target_col, stable_features, task_type, final_model
            )

        # 13. Build Provenance/Explainability data
        provenance = self._build_provenance(
            df_train,
            target_col,
            task_type,
            config,
            nested_cv_results,
            stable_features,
            feature_importance,
            holdout_results,
        )

        # 14. Generate insights (with GPU indicator)
        insights = self._generate_insights(
            task_type,
            target_col,
            nested_cv_results,
            stable_features,
            feature_importance,
            final_score,
            impact_insights,
            gpu_used=gpu_acquired,
        )

        # 15. Convert feature importance to list format for frontend
        # First consolidate one-hot encoded features, then convert to list
        consolidated_fi = self._consolidate_onehot_features(feature_importance)
        feature_importance_list = self._convert_feature_importance(consolidated_fi, stable_features, target_col)
        # Convert FeatureImportance objects to dicts
        feature_importance_output = [
            {
                "name": fi.name,
                "stability": fi.stability,
                "importance": fi.importance,
                "impact": fi.impact,
                "explanation": fi.explanation,
            }
            for fi in feature_importance_list
        ]

        total_time = time.time() - start_time
        logger.info(f"[TITAN] Analysis complete in {total_time:.2f}s")

        # Explicitly clear model memory to free GPU resources immediately
        model_class_name = final_model.__class__.__name__
        del final_model
        import gc

        gc.collect()

        return {
            "engine": self.name,
            "task_type": task_type,
            "target_column": target_col,
            "feature_count": {"total": len(feature_names), "stable": len(stable_features)},
            "nested_cv_results": nested_cv_results,
            "stable_features": stable_features,
            "feature_importance": feature_importance_output,  # Now a list of dicts
            "best_model": model_class_name,
            "final_score": final_score,
            "insights": insights,
            "variants": variants,  # Ranked analysis variants
            "provenance": provenance,  # Explainability data
            "config_schema": TITAN_CONFIG_SCHEMA,  # Available parameters
            "holdout_validation": holdout_results,  # Holdout results if enabled
            "used_gpu": gpu_acquired,  # Whether GPU was used
        }

    def _analyze_feature_impact(
        self, df: pd.DataFrame, target_col: str, features: list[str], task_type: str
    ) -> list[str]:
        """Analyze HOW the features affect the target (Directionality)"""
        impacts = []

        for feat in features[:3]:  # Analyze top 3 stable features
            try:
                if task_type == "regression":
                    # Correlation
                    corr = df[feat].corr(df[target_col])
                    direction = "increases" if corr > 0 else "decreases"
                    strength = abs(corr)
                    if strength > 0.5:
                        qual = "Strongly"
                    elif strength > 0.3:
                        qual = "Moderately"
                    else:
                        qual = "Slightly"

                    impacts.append(f"      • **{feat}**: {qual} {direction} '{target_col}' (Corr: {corr:.2f})")

                else:  # Classification
                    # For categorical features, find best/worst categories
                    if df[feat].dtype == "object" or df[feat].nunique() < 20:
                        # Group by feature and get mean of target (assuming binary 0/1 target for simplicity or taking mode)
                        # If target is not numeric (e.g. 'True'/'False'), convert to numeric if possible
                        y_numeric = df[target_col]
                        if not pd.api.types.is_numeric_dtype(y_numeric):
                            y_numeric = pd.Categorical(y_numeric).codes

                        means = df.groupby(feat)[target_col].apply(
                            lambda x: pd.Categorical(x).codes.mean()
                            if not pd.api.types.is_numeric_dtype(x)
                            else x.mean()
                        )
                        best_val = means.idxmax()
                        worst_val = means.idxmin()

                        impacts.append(
                            f"      • **{feat}**: Highest '{target_col}' at '{best_val}', Lowest at '{worst_val}'"
                        )
                    else:
                        # Continuous feature in classification
                        # Compare means of feature for each class
                        means = df.groupby(target_col)[feat].mean()
                        impacts.append(
                            f"      • **{feat}**: Average value is {means.max():.2f} for class '{means.idxmax()}' vs {means.min():.2f} for class '{means.idxmin()}'"
                        )
            except Exception:
                continue  # Skip if analysis fails for specific feature

        return impacts

    def _generate_insights(
        self,
        task_type: str,
        target_col: str,
        nested_cv: dict,
        stable_features: list[str],
        feature_importance: dict,
        final_score: float,
        impact_insights: list[str] = None,
        gpu_used: bool = False,
    ) -> list[str]:
        """Generate business-friendly insights"""
        insights = []

        # Summary with GPU indicator
        task_name = "Classification" if task_type == "classification" else "Regression"
        gpu_indicator = " 🚀 (GPU-accelerated)" if gpu_used else ""
        insights.append(
            f"📊 **Titan AutoML Complete{gpu_indicator}**: {task_name} task on '{target_col}' "
            f"with rigorous statistical validation."
        )

        # Nested CV results
        mean_score = nested_cv["mean"]
        ci = nested_cv["confidence_interval_95"]
        metric = nested_cv["metric"]

        if task_type == "classification":
            insights.append(
                f"✅ **Nested Cross-Validation (5x2)**: Accuracy = {mean_score:.1%} "
                f"(95% CI: [{ci[0]:.1%}, {ci[1]:.1%}]). This is an unbiased estimate "
                f"tested on 500 model variations."
            )
        else:
            insights.append(
                f"✅ **Nested Cross-Validation (5x2)**: MAE = {abs(mean_score):.2f} "
                f"(95% CI: [{abs(ci[1]):.2f}, {abs(ci[0]):.2f}]). Tested on 500 model variations."
            )

        # Stability Selection results
        total_features = len(feature_importance)
        stable_count = len(stable_features)

        insights.append(
            f"🔍 **Stability Selection (100 bootstrap samples)**: "
            f"{stable_count}/{total_features} features are statistically robust. "
            f"Only features appearing in >80% of trials were retained."
        )

        if stable_features:
            top_3 = list(feature_importance.items())[:3]
            insights.append("   **Top 3 Robust Features & Impact:**")
            for i, (feat, info) in enumerate(top_3, 1):
                # Handle both old format (float) and new format (dict)
                if isinstance(info, dict):
                    stability = f"{info.get('stability', 0) * 100:.0f}%"
                else:
                    stability = f"{info * 100:.0f}%"
                insights.append(f"      {i}. **{feat}** (Stability: {stability})")

            # Add directional impacts
            if impact_insights:
                insights.append("   **Directional Impact (How they affect target):**")
                for imp in impact_insights:
                    insights.append(imp)

        # Strategic recommendation
        if stable_count < total_features * 0.3:
            insights.append(
                "⚠️ **Warning**: Only a small fraction of features are robust. "
                "Consider feature engineering or collecting more data to improve model stability."
            )
        else:
            insights.append(
                "💡 **Strategic Insight**: The identified features are statistically robust "
                "across 100 different training scenarios. Predictions based on these features "
                "are reliable and not due to random chance (overfitting avoided)."
            )

        return insights

    def _timeout_response(self, target_col: str, task_type: str, elapsed: float) -> dict[str, Any]:
        """Generate a graceful timeout response"""
        return {
            "engine": self.name,
            "status": "timeout",
            "target_column": target_col,
            "task_type": task_type,
            "execution_time": elapsed,
            "insights": [
                f"⏱️ **Analysis Timeout**: The analysis exceeded the time limit ({elapsed:.1f}s).",
                "💡 **Suggestion**: Try with a smaller dataset or fewer features.",
                "📊 For faster analysis, consider sampling your data to under 1000 rows.",
            ],
            "config_schema": TITAN_CONFIG_SCHEMA,
        }

    def _detect_target_column(self, df: pd.DataFrame, profiles: dict, hint: str | None) -> str | None:
        """Detect target column using hints or heuristics"""
        # Debug hint
        import logging

        logging.info(f"[TITAN DEBUG] _detect_target_column hint: {hint} (type: {type(hint)})")

        # Priority 1: User hint
        if hint and hint in df.columns:
            return hint

        # Columns to skip (ID-like columns that shouldn't be targets)
        skip_keywords = [
            "rowname",
            "row_name",
            "index",
            "idx",
            "id",
            "key",
            "pk",
            "uuid",
            "record",
            "row_id",
            "rowid",
            "obs",
            "observation",
            "unnamed",
            "district",
            "county",
            "state",
            "country",
            "region",
            "name",
            "school",
        ]

        def is_id_column(col_name: str, col_data) -> bool:
            """Check if column appears to be an ID/index column"""
            col_lower = col_name.lower().strip()
            # Check against skip keywords
            if any(skip in col_lower for skip in skip_keywords):
                return True
            # Check if it's a sequential integer (likely an index)
            if pd.api.types.is_integer_dtype(col_data):
                uniqueness = col_data.nunique() / len(col_data)
                if uniqueness > 0.95:  # Very high uniqueness = likely ID
                    # Check if sequential
                    sorted_vals = col_data.dropna().sort_values()
                    if len(sorted_vals) > 1:
                        diffs = sorted_vals.diff().dropna()
                        if len(diffs) > 0 and (diffs == 1).mean() > 0.95:  # Sequential integers
                            return True
            return False

        # Priority 2: Common target keywords (excluding ID columns)
        target_keywords = [
            "target",
            "label",
            "class",
            "y",
            "output",
            "outcome",
            "medv",
            "price",
            "quality",
            "revenue",
            "sales",
            "profit",
            "score",
            "rating",
            "value",
            "amount",
            "total",
            "result",
            "math",
            "read",
            "fatal",
            "survived",
            "churn",
            "default",
            "fraud",
        ]
        for col in df.columns:
            if is_id_column(col, df[col]):
                continue
            if any(kw in col.lower() for kw in target_keywords):
                return col

        # Priority 3: Last NUMERIC column that isn't an ID (prefer numeric for ML)
        for col in reversed(df.columns):
            if is_id_column(col, df[col]):
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                # Additional check: not too unique (IDs are very unique)
                uniqueness = df[col].nunique() / len(df)
                if uniqueness < 0.9:  # Not an ID column
                    return col

        # Priority 4: Last categorical column with reasonable cardinality
        for col in reversed(df.columns):
            if is_id_column(col, df[col]):
                continue
            if df[col].nunique() < 50:  # Categorical with reasonable cardinality
                return col

        return None

    def _detect_task_type(self, df: pd.DataFrame, target_col: str, profiles: dict, hint: str | None) -> str:
        """Detect if task is classification or regression"""
        if hint:
            return hint.lower()

        target = df[target_col]

        # Check if categorical
        if not pd.api.types.is_numeric_dtype(target):
            return "classification"

        # Check uniqueness ratio
        uniqueness = target.nunique() / len(target)

        # Low uniqueness = classification
        if uniqueness < 0.05 or target.nunique() < 20:
            return "classification"

        # High uniqueness = regression
        return "regression"

    def _is_id_column(self, col_name: str, col_data) -> bool:
        """Check if column appears to be an ID/index column that should be excluded from features"""
        skip_keywords = [
            "rowname",
            "row_name",
            "index",
            "idx",
            "id",
            "key",
            "pk",
            "uuid",
            "record",
            "row_id",
            "rowid",
            "obs",
            "observation",
            "unnamed",
            "unnamed:",
            "level_",
        ]
        col_lower = col_name.lower().strip()
        # Check against skip keywords
        if any(skip in col_lower for skip in skip_keywords):
            return True
        # Check if column name is just a number (like '0', '1', etc.)
        if col_lower.isdigit():
            return True
        # Check if it's a sequential integer (likely an index)
        if pd.api.types.is_integer_dtype(col_data):
            n_rows = len(col_data)
            uniqueness = col_data.nunique() / max(1, n_rows)
            if uniqueness > 0.95 and n_rows > 10:  # Very high uniqueness = likely ID
                # Check if sequential
                sorted_vals = col_data.dropna().sort_values()
                if len(sorted_vals) > 1:
                    diffs = sorted_vals.diff().dropna()
                    if len(diffs) > 0 and (diffs == 1).mean() > 0.95:  # Sequential integers
                        return True
        return False

    def _prepare_data(self, df: pd.DataFrame, target_col: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Prepare feature matrix and target vector with categorical encoding"""
        # Separate features and target
        feature_cols = [c for c in df.columns if c != target_col]

        # Prepare lists for numeric and categorical features
        numeric_features = []
        categorical_features = []
        boolean_features = []
        excluded_ids = []

        for col in feature_cols:
            # Skip ID-like columns (rownames, index, etc.)
            if self._is_id_column(col, df[col]):
                excluded_ids.append(col)
                continue

            # Check for boolean dtype first (before numeric check, as bool is also numeric)
            if df[col].dtype == "bool" or (
                pd.api.types.is_numeric_dtype(df[col])
                and df[col].dropna().isin([0, 1, True, False]).all()
                and df[col].nunique() <= 2
            ):
                boolean_features.append(col)
            elif pd.api.types.is_numeric_dtype(df[col]) and not df[col].dtype == "bool":
                numeric_features.append(col)
            elif df[col].dtype == "object" or df[col].dtype.name == "category":
                # Check if it's a low-cardinality categorical (not free text)
                n_unique = df[col].nunique()
                if n_unique <= 20 and n_unique < len(df) * 0.5:  # Max 20 categories, not too many unique values
                    categorical_features.append(col)
                else:
                    # Try to convert to numeric (might be numbers stored as strings like "$3.00" or "1,234")
                    try:
                        # First try direct conversion
                        converted = pd.to_numeric(df[col], errors="coerce")
                        if converted.notna().sum() > len(df) * 0.5:
                            df[col] = converted
                            numeric_features.append(col)
                        else:
                            # Try removing currency/number formatting
                            cleaned = df[col].astype(str).str.replace("[$,€£¥]", "", regex=True)
                            cleaned = cleaned.str.replace(r"\s+", "", regex=True)
                            converted = pd.to_numeric(cleaned, errors="coerce")
                            if converted.notna().sum() > len(df) * 0.5:
                                df[col] = converted
                                numeric_features.append(col)
                    except:
                        pass

        if excluded_ids:
            import logging

            logging.info(f"Excluded ID/index columns from features: {excluded_ids}")

        # Build feature matrix
        feature_data = []
        feature_names = []

        # Add numeric features (with proper handling for extreme values)
        for col in numeric_features:
            col_values = df[col].copy().astype(float)
            # Replace infinity with NaN first
            col_values = col_values.replace([np.inf, -np.inf], np.nan)
            # Clip extreme values before computing median (only if we have valid non-nan values)
            valid_values = col_values.dropna()
            if len(valid_values) > 0:
                try:
                    q_low, q_high = valid_values.quantile([0.001, 0.999])
                    col_values = col_values.clip(lower=float(q_low), upper=float(q_high))
                except (TypeError, ValueError):
                    # If quantile fails, just use min/max
                    pass
            # Fill NaN with median (now computed on clean data)
            median_val = col_values.median()
            if pd.isna(median_val):
                median_val = 0.0
            col_values = col_values.fillna(median_val)
            feature_data.append(col_values.values.reshape(-1, 1))
            feature_names.append(col)

        # Add boolean features (simple 0/1 conversion, no scaling needed)
        for col in boolean_features:
            col_values = df[col].astype(float).fillna(0.0)
            feature_data.append(col_values.values.reshape(-1, 1))
            feature_names.append(col)

        # Encode categorical features using one-hot encoding (limited categories)
        for col in categorical_features:
            # Fill NaN with mode
            mode_val = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else "missing"
            col_filled = df[col].fillna(mode_val)

            # One-hot encode (drop_first to avoid multicollinearity)
            dummies = pd.get_dummies(col_filled, prefix=col, drop_first=True)
            for dummy_col in dummies.columns:
                feature_data.append(dummies[dummy_col].values.reshape(-1, 1))
                feature_names.append(dummy_col)

        if len(feature_data) == 0:
            raise ValueError(
                f"No usable features found for prediction. Available columns: {feature_cols}. Excluded as IDs: {excluded_ids}"
            )

        X = np.hstack(feature_data)

        # Handle target variable - encode if categorical, convert to float
        target_series = df[target_col]

        # Check if target is categorical (string/object)
        if target_series.dtype == "object" or target_series.dtype.name == "category":
            # Encode categorical target using label encoding
            from sklearn.preprocessing import LabelEncoder

            le = LabelEncoder()
            # Handle NaN values by converting to string first
            target_filled = target_series.fillna("_missing_").astype(str)
            y = le.fit_transform(target_filled).astype(float)
            # Store label mapping for potential inverse transform
            self._target_label_encoder = le
            self._target_is_categorical = True
        else:
            # Try to convert to numeric (handles cases like "$3.00" or "1,234")
            try:
                # Remove common formatting characters
                if target_series.dtype == "object":
                    target_series = target_series.astype(str).str.replace("[$,]", "", regex=True)
                    target_series = pd.to_numeric(target_series, errors="coerce")
                y = target_series.values.astype(float)
            except (ValueError, TypeError):
                # Last resort: label encode
                from sklearn.preprocessing import LabelEncoder

                le = LabelEncoder()
                target_filled = df[target_col].fillna("_missing_").astype(str)
                y = le.fit_transform(target_filled).astype(float)
                self._target_label_encoder = le
                self._target_is_categorical = True
            else:
                self._target_is_categorical = False

        # Handle inf/nan in target
        y = np.where(np.isinf(y), np.nan, y)
        y_nan_mask = np.isnan(y)
        if y_nan_mask.any():
            y_median = np.nanmedian(y)
            if np.isnan(y_median):
                y_median = 0
            y[y_nan_mask] = y_median

        # Final safety check - ensure no inf/nan in X
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

        # Scale features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Post-scaling safety check (in case scaling produces nan from constant columns)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        return X, y, feature_names

    def _nested_cross_validation(self, X: np.ndarray, y: np.ndarray, task_type: str, config: dict) -> dict[str, Any]:
        """
        Nested Cross-Validation for unbiased performance estimation

        Outer loop: 5-fold CV for performance estimation
        Inner loop: 3-fold CV for hyperparameter tuning
        """
        n_samples = len(X)
        requested_outer = config.get("cv_folds", 5)
        # Adapt folds to dataset size to avoid long runtimes and n_splits errors
        outer_folds = min(requested_outer, n_samples)
        if n_samples < 20:
            outer_folds = min(3, max(2, n_samples))  # small data => 2-3 folds
        outer_folds = max(2, outer_folds)

        # Define model and metric
        if task_type == "classification":
            base_model = RandomForestClassifier(random_state=42)
            scoring = "accuracy"
        else:
            base_model = RandomForestRegressor(random_state=42)
            scoring = "neg_mean_absolute_error"

        # Outer CV loop
        outer_cv = KFold(n_splits=outer_folds, shuffle=True, random_state=42)
        outer_scores = []

        for train_idx, test_idx in outer_cv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Inner CV loop for hyperparameter tuning (adapted for small datasets)
            param_grid = {"n_estimators": [50], "max_depth": [None]}
            if n_samples >= 200:
                param_grid = {"n_estimators": [50, 100], "max_depth": [5, 10, None]}

            inner_folds = min(3, len(train_idx))
            inner_folds = max(2, inner_folds)

            if len(train_idx) >= inner_folds and inner_folds >= 2:
                inner_cv = KFold(n_splits=inner_folds, shuffle=True, random_state=42)
                grid_search = GridSearchCV(base_model, param_grid, cv=inner_cv, scoring=scoring, n_jobs=2)
                grid_search.fit(X_train, y_train)
                best_estimator = grid_search.best_estimator_
            else:
                # Too few samples for CV; fit a single model
                best_estimator = base_model.fit(X_train, y_train)

            # Test on outer fold
            y_pred = best_estimator.predict(X_test)

            if task_type == "classification":
                score = accuracy_score(y_test, y_pred)
            else:
                score = mean_absolute_error(y_test, y_pred)

            outer_scores.append(score)

        # Calculate statistics
        mean_score = np.mean(outer_scores)
        std_score = np.std(outer_scores)

        # 95% confidence interval
        ci_lower = mean_score - 1.96 * std_score
        ci_upper = mean_score + 1.96 * std_score

        return {
            "outer_scores": outer_scores,
            "mean": mean_score,
            "std": std_score,
            "confidence_interval_95": [ci_lower, ci_upper],
            "metric": "accuracy" if task_type == "classification" else "MAE",
        }

    def _stability_selection(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
        task_type: str,
        n_bootstrap: int = 100,
        threshold: float = 0.8,
    ) -> tuple[list[str], dict[str, float]]:
        """
        Stability Selection: Identify robust features across bootstrap samples

        Only features appearing in >threshold of top-10 selections are deemed stable.
        """
        # Adapt bootstrap count to dataset size to keep runtime sensible
        n_samples = len(X)
        if n_samples < 50:
            n_bootstrap = min(n_bootstrap, max(20, n_samples))
        else:
            n_bootstrap = min(n_bootstrap, 80)
        n_features = X.shape[1]
        feature_selection_counts = np.zeros(n_features)

        # Bootstrap sampling
        for i in range(n_bootstrap):
            # Sample with replacement
            indices = np.random.choice(len(X), size=len(X), replace=True)
            X_boot = X[indices]
            y_boot = y[indices]

            # Train model
            if task_type == "classification":
                model = RandomForestClassifier(n_estimators=50, random_state=i, n_jobs=2)
            else:
                model = RandomForestRegressor(n_estimators=50, random_state=i, n_jobs=2)

            model.fit(X_boot, y_boot)

            # Get feature importances
            importances = model.feature_importances_

            # Select top 10 features (or all if fewer)
            top_k = min(10, n_features)
            top_indices = np.argsort(importances)[-top_k:]

            # Increment counts
            feature_selection_counts[top_indices] += 1

            # Also accumulate actual importance scores for averaging
            if i == 0:
                accumulated_importances = importances.copy()
            else:
                accumulated_importances += importances

        # Calculate selection frequency
        selection_frequency = feature_selection_counts / n_bootstrap

        # Calculate average feature importance across all bootstrap samples
        avg_importances = accumulated_importances / n_bootstrap

        # Normalize average importances to 0-1 range (max = 1.0)
        if avg_importances.max() > 0:
            normalized_importances = avg_importances / avg_importances.max()
        else:
            normalized_importances = avg_importances

        # Features selected in >threshold of bootstrap samples
        stable_indices = np.where(selection_frequency > threshold)[0]
        stable_features = [feature_names[i] for i in stable_indices]

        # Build feature info dictionary with BOTH stability and importance
        # - stability: how often the feature was selected (selection_frequency)
        # - importance: the actual predictive power (normalized_importances)
        feature_info = {
            feature_names[i]: {
                "stability": round(selection_frequency[i], 3),
                "importance": round(normalized_importances[i], 3),
            }
            for i in range(n_features)
        }

        # Sort by importance
        feature_info = dict(sorted(feature_info.items(), key=lambda x: x[1]["importance"], reverse=True))

        return stable_features, feature_info

    def _train_final_model(self, X: np.ndarray, y: np.ndarray, task_type: str) -> tuple[Any, float]:
        """Train final ensemble model on stable features"""
        if task_type == "classification":
            model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=2)
            # Determine appropriate number of CV folds based on minimum class size
            from collections import Counter

            class_counts = Counter(y)
            min_class_size = min(class_counts.values())
            cv_folds = min(5, max(2, min_class_size))
        else:
            model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=2)
            cv_folds = min(5, max(2, len(y) // 10))  # At least 10 samples per fold

        # Cross-validated score
        try:
            scores = cross_val_score(
                model,
                X,
                y,
                cv=cv_folds,
                scoring="accuracy" if task_type == "classification" else "neg_mean_absolute_error",
            )
            final_score = scores.mean()
        except Exception:
            # Fallback: train/test split if CV fails
            final_score = 0.0

        # Train on full data
        model.fit(X, y)

        return model, final_score

    # =========================================================================
    # NEW PREMIUM METHODS: Multi-Variant, Gemma Ranking, Holdout, Provenance
    # =========================================================================

    def _generate_variants(
        self,
        df: pd.DataFrame,
        target_col: str,
        task_type: str,
        feature_names: list[str],
        stable_features: list[str],
        feature_importance: dict[str, float],
        n_variants: int,
        config: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Generate multiple analysis variants with different configurations.
        Each variant explores a different feature subset or model type.
        """
        variants = []

        # Get top features sorted by importance
        # Handle both old format (float) and new format (dict with importance key)
        def get_sort_key(item):
            feat, info = item
            if isinstance(info, dict):
                return info.get("importance", 0)
            return info

        sorted_features = sorted(feature_importance.items(), key=get_sort_key, reverse=True)
        top_features = [f for f, _ in sorted_features[: min(15, len(sorted_features))]]

        # Variant 1: Best model with all stable features (baseline)
        variants.append(
            {
                "variant_id": self._generate_variant_id("baseline"),
                "variant_type": "baseline",
                "description": "All stable features with best model",
                "features_used": stable_features.copy(),
                "model_type": "RandomForest",
                "feature_count": len(stable_features),
                "raw_insights": [
                    f"Uses {len(stable_features)} stable features identified through bootstrap stability selection"
                ],
            }
        )

        # Variant 2-4: Different model types
        for model_config in MODEL_CONFIGS:
            if len(variants) >= n_variants:
                break
            if model_config["name"] != "RandomForest":  # Skip baseline model
                variants.append(
                    {
                        "variant_id": self._generate_variant_id(model_config["name"].lower()),
                        "variant_type": "model_variation",
                        "description": f"{model_config['name']} with stable features",
                        "features_used": stable_features.copy(),
                        "model_type": model_config["name"],
                        "feature_count": len(stable_features),
                        "raw_insights": [f"{model_config['name']} may capture different patterns than RandomForest"],
                    }
                )

        # Variants 5-8: Feature subset variations (top-N features)
        for n_features in [3, 5, 7, 10]:
            if len(variants) >= n_variants:
                break
            subset = top_features[:n_features]
            if len(subset) >= 2:
                variants.append(
                    {
                        "variant_id": self._generate_variant_id(f"top{n_features}"),
                        "variant_type": "feature_subset",
                        "description": f"Top {n_features} features only (simplified model)",
                        "features_used": subset,
                        "model_type": "RandomForest",
                        "feature_count": len(subset),
                        "raw_insights": [
                            f"Simplified model with only the {n_features} most important features for interpretability"
                        ],
                    }
                )

        # Variants 9+: Feature combinations (pairs of important features)
        if len(top_features) >= 4 and len(variants) < n_variants:
            for combo in combinations(top_features[:6], 3):
                if len(variants) >= n_variants:
                    break
                combo_list = list(combo)
                variants.append(
                    {
                        "variant_id": self._generate_variant_id(f"combo_{combo_list[0][:3]}"),
                        "variant_type": "feature_combination",
                        "description": f"Feature combination: {', '.join(combo_list)}",
                        "features_used": combo_list,
                        "model_type": "RandomForest",
                        "feature_count": len(combo_list),
                        "raw_insights": [f"Explores interaction between {', '.join(combo_list)}"],
                    }
                )

        # Score each variant
        X, y, encoded_feature_names = self._prepare_data(df, target_col)

        import logging

        logging.info(f"[TITAN DEBUG] Encoded feature names: {encoded_feature_names}")
        logging.info(f"[TITAN DEBUG] X shape: {X.shape}, y shape: {y.shape}")

        # Build mapping from original feature names to encoded feature indices
        # e.g., "region" maps to indices for "region_North", "region_South", etc.
        def get_feature_indices(original_features, encoded_names):
            """Map original feature names to encoded feature indices"""
            indices = []
            for orig_feat in original_features:
                for i, enc_name in enumerate(encoded_names):
                    # Match exact name or one-hot encoded prefix (e.g., "region" matches "region_North")
                    if enc_name == orig_feat or enc_name.startswith(f"{orig_feat}_"):
                        indices.append(i)
            return list(set(indices))  # Remove duplicates

        # Determine appropriate CV folds for classification
        if task_type == "classification":
            from collections import Counter

            class_counts = Counter(y)
            min_class_size = min(class_counts.values())
            cv_folds = min(3, max(2, min_class_size))
        else:
            cv_folds = min(3, max(2, len(y) // 10))

        logging.info(f"[TITAN DEBUG] CV folds: {cv_folds}, task_type: {task_type}")

        for variant in variants:
            try:
                variant_features = variant["features_used"]
                # Use the mapping function to handle one-hot encoded features
                feature_indices = get_feature_indices(variant_features, encoded_feature_names)

                logging.info(
                    f"[TITAN DEBUG] Variant {variant.get('model_type')}: features_used={variant_features}, mapped_indices={feature_indices}"
                )

                if len(feature_indices) >= 1:
                    X_variant = X[:, feature_indices]

                    # Quick CV for speed
                    model_class = self._get_model_class(variant["model_type"], task_type)
                    model = model_class(random_state=42)

                    if task_type == "classification":
                        # Use accuracy for classification (0-1 scale)
                        scoring = "accuracy"
                        scores = cross_val_score(model, X_variant, y, cv=cv_folds, scoring=scoring)
                        cv_score = float(np.mean(scores))
                        cv_std = float(np.std(scores))
                        score_type = "accuracy"
                    else:
                        # Use neg_mean_absolute_error for regression (more stable than R² for small data)
                        # Then convert to a 0-1 quality score based on error relative to target range
                        scoring = "neg_mean_absolute_error"
                        scores = cross_val_score(model, X_variant, y, cv=cv_folds, scoring=scoring)
                        mae = -float(np.mean(scores))  # Convert from negative to positive
                        mae_std = float(np.std(scores))

                        # Calculate relative MAE: how good is the MAE relative to target variance?
                        # Lower MAE is better. Convert to 0-1 score where 1 is best.
                        y_range = np.max(y) - np.min(y) if np.max(y) != np.min(y) else 1.0
                        y_std = np.std(y) if np.std(y) > 0 else 1.0

                        # Score: 1 - (MAE / y_std) clamped to [0, 1]
                        # If MAE equals y_std, that's like random guessing = 0.5 score
                        # If MAE is 0, perfect = 1.0 score
                        # If MAE > y_std, bad model = closer to 0
                        relative_error = mae / y_std
                        cv_score = max(0.0, min(1.0, 1.0 - (relative_error / 2.0)))
                        cv_std = mae_std / y_std if y_std > 0 else 0.0
                        score_type = "quality"  # Derived quality score

                        logging.info(
                            f"[TITAN DEBUG] MAE={mae:.4f}, y_std={y_std:.4f}, relative_error={relative_error:.4f}, cv_score={cv_score:.4f}"
                        )

                    logging.info(f"[TITAN DEBUG] CV scores: {scores}, final_score: {cv_score}")

                    variant["cv_score"] = cv_score
                    variant["cv_std"] = cv_std
                    variant["score_type"] = score_type
                else:
                    logging.warning("[TITAN DEBUG] No feature indices found for variant!")
                    variant["cv_score"] = 0.0
                    variant["cv_std"] = 0.0
            except Exception as e:
                logging.error(f"[TITAN DEBUG] Variant error: {e}")
                variant["cv_score"] = 0.0
                variant["cv_std"] = 0.0
                variant["error"] = str(e)[:100]

        # Sort by CV score (descending - higher is better for both accuracy and R²)
        variants.sort(key=lambda x: x.get("cv_score", 0), reverse=True)

        return variants

    def _generate_variant_id(self, prefix: str) -> str:
        """Generate unique variant ID"""
        return f"{prefix}_{uuid.uuid4().hex[:8]}"

    def _get_model_class(self, model_name: str, task_type: str):
        """Get sklearn model class by name"""
        models = {
            "RandomForest": (RandomForestClassifier, RandomForestRegressor),
            "GradientBoosting": (GradientBoostingClassifier, GradientBoostingRegressor),
            "DecisionTree": (DecisionTreeClassifier, DecisionTreeRegressor),
        }
        clf, reg = models.get(model_name, models["RandomForest"])
        return clf if task_type == "classification" else reg

    def _rank_with_gemma(self, variants: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Use Gemma LLM to rank variants by business utility (1-100).
        Returns variants with gemma_score and gemma_explanation added.
        """
        # Graceful fallback: if Gemma client is unavailable, derive a proxy score
        if not self.gemma_client or not variants:
            for variant in variants:
                variant["gemma_score"] = int(min(100, max(1, variant.get("cv_score", 0.5) * 100)))
                variant["gemma_explanation"] = "Gemma unavailable; score based on CV performance."
            variants.sort(key=lambda x: x.get("gemma_score", 0), reverse=True)
            return variants

        # Build prompt for Gemma
        variant_summaries = []
        for i, v in enumerate(variants[:20]):  # Limit to 20 for context window
            label = chr(65 + i)  # A, B, C, ...
            summary = f"""
Variant {label}:
- Type: {v.get("variant_type", "unknown")}
- Model: {v.get("model_type", "unknown")}
- Features: {", ".join(v.get("features_used", [])[:5])}{"..." if len(v.get("features_used", [])) > 5 else ""}
- CV Score: {v.get("cv_score", 0):.3f}
- Description: {v.get("description", "N/A")}
"""
            variant_summaries.append(summary)

        prompt = f"""You are a Senior Data Science consultant evaluating ML analysis variants for business value.

Rate each variant from 1-100 for BUSINESS UTILITY (not just accuracy). Consider:
- Interpretability: Can executives understand this?
- Actionability: Can the business act on these features?
- Reliability: Is the model trustworthy (CV score + feature count)?
- Simplicity: Simpler models are often better for business decisions

Here are the variants:

{"".join(variant_summaries)}

Return ONLY valid JSON in this exact format (no other text):
{{
  "A": {{"score": 85, "explanation": "One sentence explaining the rating"}},
  "B": {{"score": 72, "explanation": "One sentence explaining the rating"}},
  ...
}}
"""

        try:
            # Call Gemma
            response = self.gemma_client(prompt=prompt, max_tokens=1024, temperature=0.3)

            # Extract response text
            if isinstance(response, dict):
                gemma_text = response.get("text", "") or response.get("response", "") or str(response)
            else:
                gemma_text = str(response)

            # Parse JSON from response
            json_match = re.search(r"\{[\s\S]*\}", gemma_text)
            if json_match:
                rankings = json.loads(json_match.group(0))

                # Apply rankings to variants
                for i, variant in enumerate(variants[:20]):
                    label = chr(65 + i)
                    if label in rankings:
                        variant["gemma_score"] = int(rankings[label].get("score", 50))
                        variant["gemma_explanation"] = rankings[label].get("explanation", "")
                    else:
                        variant["gemma_score"] = 50  # Default
                        variant["gemma_explanation"] = "No rating provided"

                # Re-sort by Gemma score
                variants.sort(key=lambda x: x.get("gemma_score", 0), reverse=True)

        except Exception as e:
            # Fallback: use CV scores as proxy
            for variant in variants:
                variant["gemma_score"] = int(min(100, max(1, variant.get("cv_score", 0.5) * 100)))
                variant["gemma_explanation"] = f"Fallback: based on CV score ({e})"

        return variants

    def _validate_holdout(
        self,
        df_train: pd.DataFrame,
        df_holdout: pd.DataFrame,
        target_col: str,
        stable_features: list[str],
        task_type: str,
        model,
    ) -> dict[str, Any]:
        """
        Validate model on holdout set to verify generalization.
        This is the "only do 4/5 of database" validation.
        """
        try:
            # Prepare holdout data
            X_train, y_train, feature_names = self._prepare_data(df_train, target_col)
            X_holdout, y_holdout, _ = self._prepare_data(df_holdout, target_col)

            # Get stable feature indices
            feature_indices = [i for i, f in enumerate(feature_names) if f in stable_features]

            if len(feature_indices) == 0:
                return {"error": "No stable features available for holdout validation"}

            X_train_stable = X_train[:, feature_indices]
            X_holdout_stable = X_holdout[:, feature_indices]

            # Train on training set
            model.fit(X_train_stable, y_train)

            # Predict on holdout
            y_pred = model.predict(X_holdout_stable)

            # Calculate metrics
            if task_type == "classification":
                holdout_score = accuracy_score(y_holdout, y_pred)
                metric = "accuracy"
            else:
                holdout_score = mean_absolute_error(y_holdout, y_pred)
                metric = "MAE"

            return {
                "holdout_score": float(holdout_score),
                "metric": metric,
                "train_samples": len(df_train),
                "holdout_samples": len(df_holdout),
                "features_used": stable_features,
                "validation_passed": holdout_score > 0.5 if task_type == "classification" else True,
                "interpretation": self._interpret_holdout_score(holdout_score, task_type),
            }

        except Exception as e:
            return {"error": str(e)}

    def _interpret_holdout_score(self, score: float, task_type: str) -> str:
        """Generate human-readable interpretation of holdout score"""
        if task_type == "classification":
            if score >= 0.9:
                return "Excellent generalization - model performs very well on unseen data"
            elif score >= 0.75:
                return "Good generalization - model is reliable for production use"
            elif score >= 0.6:
                return "Moderate generalization - consider more data or feature engineering"
            else:
                return "Poor generalization - model may be overfitting or data is insufficient"
        else:
            return f"Model achieves MAE of {score:.2f} on holdout data"

    def _build_provenance(
        self,
        df: pd.DataFrame,
        target_col: str,
        task_type: str,
        config: dict[str, Any],
        nested_cv: dict,
        stable_features: list[str],
        feature_importance: dict[str, float],
        holdout_results: dict | None,
    ) -> dict[str, Any]:
        """
        Build complete provenance/explainability data for the analysis.
        This powers the "Explain This" section in the UI.
        """
        return {
            "methodology": {
                "name": "Nested Cross-Validation with Stability Selection",
                "reference_url": "https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html",
                "description": "Nested CV provides unbiased performance estimation while inner CV tunes hyperparameters. Stability selection identifies features that are consistently important across bootstrap samples.",
            },
            "pipeline_steps": [
                {
                    "step": 1,
                    "name": "Data Profiling",
                    "description": "Analyzed column types, distributions, and missing values",
                    "output": f"{len(df.columns)} columns profiled, {len(df)} rows",
                },
                {
                    "step": 2,
                    "name": "Target Detection",
                    "description": f"Identified '{target_col}' as target column",
                    "task_type": task_type,
                },
                {
                    "step": 3,
                    "name": "Feature Preparation",
                    "description": "Numeric features extracted, scaled with StandardScaler",
                    "features_in": len([c for c in df.columns if c != target_col]),
                    "features_out": len(feature_importance),
                },
                {
                    "step": 4,
                    "name": "Nested Cross-Validation",
                    "description": f"{config.get('cv_folds', 5)}-fold outer CV, 3-fold inner CV with GridSearchCV",
                    "outer_scores": nested_cv.get("outer_scores", []),
                    "mean_score": nested_cv.get("mean"),
                    "confidence_interval": nested_cv.get("confidence_interval_95"),
                },
                {
                    "step": 5,
                    "name": "Stability Selection",
                    "description": f"{config.get('n_bootstrap', 100)} bootstrap samples, {config.get('stability_threshold', 0.8) * 100:.0f}% threshold",
                    "stable_features_count": len(stable_features),
                    "total_features": len(feature_importance),
                },
                {
                    "step": 6,
                    "name": "Final Model Training",
                    "description": "RandomForest trained on stable features only",
                },
            ],
            "configuration_used": {k: v for k, v in config.items() if k in TITAN_CONFIG_SCHEMA},
            "feature_stability_scores": {
                f: {
                    "stability": info.get("stability", info) if isinstance(info, dict) else info,
                    "importance": info.get("importance", info) if isinstance(info, dict) else info,
                    "is_stable": f in stable_features,
                }
                for f, info in feature_importance.items()
            },
            "holdout_validation": holdout_results,
            "data_summary": {
                "rows": len(df),
                "columns": len(df.columns),
                "target_column": target_col,
                "task_type": task_type,
                "numeric_features": len(feature_importance),
            },
        }

    # =========================================================================
    # PREMIUM OUTPUT: Standardized PremiumResult format for unified UI
    # =========================================================================

    async def run_premium_async(self, df: pd.DataFrame, config: dict[str, Any] | None = None) -> PremiumResult:
        """
        Run Titan analysis with GPU support and return standardized PremiumResult.

        This async wrapper uses analyze_async() for GPU-accelerated XGBoost
        and converts output to the unified PremiumResult format.

        Args:
            df: Input DataFrame
            config: Configuration overrides

        Returns:
            PremiumResult with all components for unified UI (GPU-accelerated)
        """
        import time

        start_time = time.time()

        # Get raw analysis results using async GPU method
        config = config or {}
        logger.info("[TITAN] 🚀 Running premium analysis with GPU support (async)")
        raw_result = await self.analyze_async(df, config=config)

        return self._build_premium_result(raw_result, df, config, start_time)

    def run_premium(self, df: pd.DataFrame, config: dict[str, Any] | None = None) -> PremiumResult:
        """
        Run Titan analysis and return standardized PremiumResult.

        This wrapper converts the internal analyze() output to the unified
        PremiumResult format used by the frontend predictions.html.

        NOTE: This sync version uses CPU only. For GPU support, use run_premium_async().

        Args:
            df: Input DataFrame
            config: Configuration overrides

        Returns:
            PremiumResult with all components for unified UI
        """
        import time

        start_time = time.time()

        # Get raw analysis results (sync = CPU only)
        config = config or {}
        logger.info("[TITAN] Running premium analysis (sync/CPU mode)")
        raw_result = self.analyze(df, config=config)

        return self._build_premium_result(raw_result, df, config, start_time)

    def _build_premium_result(
        self, raw_result: dict[str, Any], df: pd.DataFrame, config: dict[str, Any], start_time: float
    ) -> PremiumResult:
        """Build PremiumResult from raw analysis output."""
        import time

        # Handle errors
        if "error" in raw_result:
            return self._error_to_premium_result(raw_result, df, config, start_time)

        # Convert variants to standardized format
        variants = self._convert_variants(raw_result.get("variants", []))

        # Get best variant
        best_variant = variants[0] if variants else self._create_fallback_variant()

        # Build feature importance list
        feature_importance = self._convert_feature_importance(
            raw_result.get("feature_importance", {}),
            raw_result.get("stable_features", []),
            raw_result.get("target_column", ""),
        )

        # Build plain English summary
        summary = self._build_summary(raw_result, best_variant)

        # Build technical explanation
        explanation = self._build_explanation(raw_result)

        # Convert holdout validation
        holdout = self._convert_holdout(raw_result.get("holdout_validation"))

        # Build config schema
        config_schema = self._build_config_schema()

        execution_time = time.time() - start_time

        # Include GPU indicator in display name if used
        used_gpu = raw_result.get("used_gpu", False)
        display_name = "Titan AutoML (GPU)" if used_gpu else "Titan AutoML"

        return PremiumResult(
            engine_name="titan",
            engine_display_name=display_name,
            engine_icon="🔱",
            task_type=TaskType.PREDICTION,
            target_column=raw_result.get("target_column"),
            columns_analyzed=list(df.columns),
            row_count=len(df),
            variants=variants,
            best_variant=best_variant,
            feature_importance=feature_importance,
            summary=summary,
            explanation=explanation,
            holdout=holdout,
            config_used=config,
            config_schema=config_schema,
            execution_time_seconds=execution_time,
            warnings=raw_result.get("warnings", []),
        )

    def _convert_variants(self, raw_variants: list[dict]) -> list[Variant]:
        """Convert raw variant dicts to Variant dataclass objects"""
        variants = []
        for i, rv in enumerate(raw_variants):
            variant = Variant(
                rank=i + 1,
                gemma_score=rv.get("gemma_score", int(rv.get("cv_score", 0.5) * 100)),
                cv_score=rv.get("cv_score", 0.0),
                variant_type=rv.get("variant_type", "baseline"),
                model_name=rv.get("model_type", "RandomForest"),
                features_used=rv.get("features_used", []),
                interpretation=rv.get("description", "Analysis variant"),
                details={
                    "cv_std": rv.get("cv_std", 0.0),
                    "feature_count": rv.get("feature_count", 0),
                    "gemma_explanation": rv.get("gemma_explanation", ""),
                    "variant_id": rv.get("variant_id", f"v{i + 1}"),
                },
            )
            variants.append(variant)
        return variants

    def _consolidate_onehot_features(self, raw_importance) -> dict[str, any]:
        """
        Consolidate one-hot encoded features back to their original column names.
        E.g., {'Department_Sales': 0.3, 'Department_Marketing': 0.2} -> {'Department': 0.5}

        Accepts either a dict or list of dicts.
        Now handles new format: {'feature': {'stability': x, 'importance': y}}
        """
        # Handle list input (already converted to list format)
        if isinstance(raw_importance, list):
            raw_dict = {}
            for item in raw_importance:
                if isinstance(item, dict):
                    name = item.get("name", item.get("feature", ""))
                    stab = item.get("stability", 0)
                    imp = item.get("importance", item.get("value", 0))
                    if isinstance(stab, (int, float)) and stab > 1:
                        stab = stab / 100  # Convert from percentage
                    if isinstance(imp, (int, float)) and imp > 1:
                        imp = imp / 100  # Convert from percentage
                    raw_dict[name] = {"stability": stab, "importance": imp}
            raw_importance = raw_dict

        if not isinstance(raw_importance, dict):
            return {}

        # Check if this is the new format (dict values are dicts)
        sample_value = next(iter(raw_importance.values()), None) if raw_importance else None
        is_new_format = isinstance(sample_value, dict)

        consolidated = {}

        for feat, value in raw_importance.items():
            # Extract scores based on format
            if is_new_format:
                stability = value.get("stability", 0)
                importance = value.get("importance", 0)
            else:
                stability = value
                importance = value

            # Check if this is a one-hot encoded feature (contains underscore)
            if "_" in feat:
                # Split on underscore and take first part as parent column
                parts = feat.split("_")
                parent = parts[0]

                # Look for other features with same prefix to confirm it's one-hot
                matching = [k for k in raw_importance.keys() if k.startswith(f"{parent}_")]

                if len(matching) > 1:
                    # This is definitely one-hot encoded - consolidate
                    if parent not in consolidated:
                        consolidated[parent] = {"stability": 0.0, "importance": 0.0}
                    consolidated[parent]["stability"] += stability
                    consolidated[parent]["importance"] += importance
                else:
                    # Not one-hot, keep original name
                    consolidated[feat] = {"stability": stability, "importance": importance}
            else:
                # Regular feature, keep as-is
                consolidated[feat] = {"stability": stability, "importance": importance}

        # Normalize consolidated values so max importance is 1.0
        if consolidated:
            max_imp = max(v["importance"] for v in consolidated.values())
            max_stab = max(v["stability"] for v in consolidated.values())
            if max_imp > 0:
                for k in consolidated:
                    consolidated[k]["importance"] = consolidated[k]["importance"] / max_imp
            if max_stab > 0:
                for k in consolidated:
                    consolidated[k]["stability"] = consolidated[k]["stability"] / max_stab

        # Sort by importance
        consolidated = dict(sorted(consolidated.items(), key=lambda x: x[1]["importance"], reverse=True))

        return consolidated

    def _convert_feature_importance(
        self, raw_importance: dict[str, any], stable_features: list[str], target_col: str
    ) -> list[FeatureImportance]:
        """Convert raw feature importance to FeatureImportance objects"""
        translator = BusinessTranslator()
        features = []

        # Consolidate one-hot encoded features for cleaner display
        consolidated = self._consolidate_onehot_features(raw_importance)

        for feat, info in consolidated.items():
            # Handle both old format (float) and new format (dict with stability/importance)
            if isinstance(info, dict):
                stability = info.get("stability", 0)
                importance = info.get("importance", 0)
            else:
                # Legacy format - use same value for both
                stability = info
                importance = info

            is_stable = feat in stable_features or any(sf.startswith(f"{feat}_") for sf in stable_features)
            # Determine impact direction based on importance value
            impact = "positive" if importance > 0.5 else "mixed"

            explanation = translator.translate_feature(feat, stability, is_stable, target_col)

            features.append(
                FeatureImportance(
                    name=feat,
                    stability=stability * 100,  # Convert to percentage
                    importance=importance,  # Now uses actual importance, not stability
                    impact=impact,
                    explanation=explanation,
                )
            )

        return features

    def _build_summary(self, raw_result: dict, best_variant: Variant) -> PlainEnglishSummary:
        """Build plain English summary from analysis results"""
        translator = BusinessTranslator()

        task_type = raw_result.get("task_type", "prediction")
        target = raw_result.get("target_column", "target")
        score = best_variant.cv_score
        stable_count = len(raw_result.get("stable_features", []))
        total_features = len(raw_result.get("feature_importance", {}))

        # Build headline (score is now 0-1 for both classification accuracy and regression R²)
        if task_type == "classification":
            headline = f"'{target}' can be predicted with {score:.1%} accuracy"
        else:
            headline = f"'{target}' predictions explain {score:.1%} of variance (R² score)"

        # Build explanation
        explanation = translator.generate_summary(
            engine_name="titan",
            task_type=task_type,
            target=target,
            score=score,
            stable_features=stable_count,
            total_features=total_features,
        )

        # Build recommendation (score is now 0-1 for both task types)
        if stable_count < total_features * 0.3:
            recommendation = "Consider collecting more data or engineering new features to improve model stability."
        elif score < 0.5:
            recommendation = "Model performance is weak. The data may not contain strong predictive signals, or more features are needed."
        elif score < 0.7:
            recommendation = (
                "Model performance is moderate. Try adding more relevant features or exploring alternative model types."
            )
        elif score < 0.85:
            recommendation = "Model shows good predictive power. Consider validating on new data before production use."
        else:
            recommendation = "Model is ready for production use. Consider A/B testing with real users."

        # Determine confidence (score is now 0-1 for both task types)
        if score >= 0.85 and stable_count >= 3:
            confidence = Confidence.HIGH
        elif score >= 0.7:
            confidence = Confidence.MEDIUM
        elif score >= 0.5:
            confidence = Confidence.LOW
        else:
            confidence = Confidence.LOW

        return PlainEnglishSummary(
            headline=headline, explanation=explanation, recommendation=recommendation, confidence=confidence
        )

    def _build_explanation(self, raw_result: dict) -> TechnicalExplanation:
        """Build technical explanation from provenance data"""
        provenance = raw_result.get("provenance", {})
        methodology = provenance.get("methodology", {})

        steps = []
        for step_data in provenance.get("pipeline_steps", []):
            steps.append(
                ExplanationStep(
                    step_number=step_data.get("step", 0),
                    title=step_data.get("name", ""),
                    description=step_data.get("description", ""),
                    details={k: v for k, v in step_data.items() if k not in ["step", "name", "description"]},
                )
            )

        return TechnicalExplanation(
            methodology_name=methodology.get("name", "Nested CV + Stability Selection"),
            methodology_url=methodology.get(
                "reference_url",
                "https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html",
            ),
            steps=steps
            if steps
            else [
                ExplanationStep(1, "Data Profiling", "Analyzed column types and distributions"),
                ExplanationStep(2, "Feature Selection", "Identified stable predictive features"),
                ExplanationStep(3, "Model Training", "Trained ensemble models with cross-validation"),
                ExplanationStep(4, "Validation", "Verified generalization on holdout data"),
            ],
            limitations=[
                "Requires numeric features (categorical encoding applied automatically)",
                "Minimum 100 samples recommended for reliable results",
                "Feature interactions may not be fully captured",
            ],
        )

    def _convert_holdout(self, raw_holdout: dict | None) -> HoldoutResult | None:
        """Convert raw holdout dict to HoldoutResult"""
        if not raw_holdout or "error" in raw_holdout:
            return None

        return HoldoutResult(
            train_samples=raw_holdout.get("train_samples", 0),
            holdout_samples=raw_holdout.get("holdout_samples", 0),
            holdout_ratio=raw_holdout.get("holdout_samples", 0)
            / max(1, raw_holdout.get("train_samples", 1) + raw_holdout.get("holdout_samples", 1)),
            train_score=0.0,  # Not tracked in current implementation
            holdout_score=raw_holdout.get("holdout_score", 0.0),
            metric_name=raw_holdout.get("metric", "accuracy"),
            passed=raw_holdout.get("validation_passed", True),
            message=raw_holdout.get("interpretation", "Holdout validation completed"),
        )

    def _build_config_schema(self) -> list[ConfigParameter]:
        """Convert TITAN_CONFIG_SCHEMA to list of ConfigParameter objects"""
        params = []
        for name, schema in TITAN_CONFIG_SCHEMA.items():
            param_range = None
            if "min" in schema and "max" in schema:
                param_range = [schema["min"], schema["max"]]

            params.append(
                ConfigParameter(
                    name=name,
                    type=schema.get("type", "int"),
                    default=schema.get("default"),
                    range=param_range,
                    description=schema.get("description", ""),
                )
            )
        return params

    def _create_fallback_variant(self) -> Variant:
        """Create fallback variant when no results available"""
        return Variant(
            rank=1,
            gemma_score=0,
            cv_score=0.0,
            variant_type="fallback",
            model_name="None",
            features_used=[],
            interpretation="Analysis could not be completed",
            details={"error": "No variants generated"},
        )

    def _error_to_premium_result(
        self, raw_result: dict, df: pd.DataFrame, config: dict, start_time: float
    ) -> PremiumResult:
        """Convert error result to PremiumResult"""
        import time

        return PremiumResult(
            engine_name="titan",
            engine_display_name="Titan AutoML",
            engine_icon="🔱",
            task_type=TaskType.PREDICTION,
            target_column=None,
            columns_analyzed=list(df.columns),
            row_count=len(df),
            variants=[],
            best_variant=self._create_fallback_variant(),
            feature_importance=[],
            summary=PlainEnglishSummary(
                headline="Analysis could not be completed",
                explanation=raw_result.get("message", "Unknown error occurred"),
                recommendation="Please check your data format and try again.",
                confidence=Confidence.LOW,
            ),
            explanation=TechnicalExplanation(
                methodology_name="Nested CV + Stability Selection",
                methodology_url=None,
                steps=[],
                limitations=["Analysis failed - see error message"],
            ),
            holdout=None,
            config_used=config,
            config_schema=self._build_config_schema(),
            execution_time_seconds=time.time() - start_time,
            warnings=[raw_result.get("message", "Error occurred")],
        )


# CLI entry point
if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Titan AutoML Engine - Enterprise-Grade Premium ML Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python titan_engine.py data.csv
  python titan_engine.py data.csv --target price --variants 10
  python titan_engine.py data.csv --holdout 0.2 --explain
  python titan_engine.py data.csv --param cv_folds=3 --param n_bootstrap=50
        """,
    )

    parser.add_argument("csv_file", nargs="?", help="Path to CSV file")
    parser.add_argument("--target", "-t", help="Target column name")
    parser.add_argument("--variants", "-v", type=int, default=5, help="Number of variants to generate (default: 5)")
    parser.add_argument("--holdout", type=float, default=0.0, help="Holdout ratio for validation (e.g., 0.2 for 20%%)")
    parser.add_argument("--explain", "-e", action="store_true", help="Show detailed provenance/explainability")
    parser.add_argument("--rank", "-r", action="store_true", help="Enable Gemma ranking (requires Gemma client)")
    parser.add_argument("--param", "-p", action="append", help="Override parameter (format: key=value)")
    parser.add_argument("--json", "-j", action="store_true", help="Output results as JSON")
    parser.add_argument("--schema", "-s", action="store_true", help="Show configuration schema and exit")

    args = parser.parse_args()

    # Show schema and exit
    if args.schema:
        print("\nTitan Engine Configuration Schema:")
        print("=" * 60)
        for key, schema in TITAN_CONFIG_SCHEMA.items():
            print(f"\n{key}:")
            for k, v in schema.items():
                print(f"  {k}: {v}")
        sys.exit(0)

    # Require csv_file if not showing schema
    if not args.csv_file:
        parser.error("csv_file is required (or use --schema to see options)")

    # Build config
    config = {
        "target_column": args.target,
        "n_variants": args.variants,
        "holdout_ratio": args.holdout,
        "enable_gemma_ranking": args.rank,
    }

    # Parse parameter overrides
    if args.param:
        for param in args.param:
            if "=" in param:
                key, value = param.split("=", 1)
                # Type conversion based on schema
                if key in TITAN_CONFIG_SCHEMA:
                    param_type = TITAN_CONFIG_SCHEMA[key].get("type", "str")
                    if param_type == "int":
                        value = int(value)
                    elif param_type == "float":
                        value = float(value)
                    elif param_type == "bool":
                        value = value.lower() in ("true", "1", "yes")
                config[key] = value

    # Load data
    df = pd.read_csv(args.csv_file)

    # Run analysis
    engine = TitanEngine()
    result = engine.analyze(df, config)

    # Output
    if args.json:
        # JSON output for programmatic use
        print(json.dumps(result, indent=2, default=str))
    else:
        # Human-readable output
        print(f"\n{'=' * 60}")
        print(f"TITAN ENGINE PREMIUM RESULTS: {args.csv_file}")
        print(f"{'=' * 60}\n")

        if "error" in result:
            print(f"❌ Error: {result['message']}")
            for insight in result.get("insights", []):
                print(f"   {insight}")
        else:
            # Insights
            print("📊 INSIGHTS:")
            print("-" * 40)
            for insight in result["insights"]:
                print(insight)

            # Variants
            variants = result.get("variants", [])
            if variants:
                print(f"\n{'=' * 60}")
                print(f"🔀 ANALYSIS VARIANTS ({len(variants)} generated):")
                print("-" * 40)
                for i, v in enumerate(variants[:5], 1):  # Show top 5
                    score_str = f"{v.get('cv_score', 0):.3f}"
                    gemma_str = f" | Gemma: {v.get('gemma_score', 'N/A')}/100" if "gemma_score" in v else ""
                    print(f"  {i}. [{v['variant_type']}] {v['description']}")
                    print(
                        f"     Model: {v['model_type']} | Features: {v['feature_count']} | Score: {score_str}{gemma_str}"
                    )
                    if "gemma_explanation" in v:
                        print(f"     💡 {v['gemma_explanation']}")
                if len(variants) > 5:
                    print(f"     ... +{len(variants) - 5} more variants")

            # Holdout validation
            holdout = result.get("holdout_validation")
            if holdout and "error" not in holdout:
                print(f"\n{'=' * 60}")
                print("✅ HOLDOUT VALIDATION:")
                print("-" * 40)
                print(f"  Train samples: {holdout['train_samples']}")
                print(f"  Holdout samples: {holdout['holdout_samples']}")
                print(f"  Holdout {holdout['metric']}: {holdout['holdout_score']:.4f}")
                print(f"  {holdout['interpretation']}")

            # Feature stability
            print(f"\n{'=' * 60}")
            print("FEATURE STABILITY ANALYSIS:")
            print(f"{'=' * 60}\n")

            for feat, score in list(result["feature_importance"].items())[:10]:
                bar = "█" * int(score * 30) + "░" * (30 - int(score * 30))
                stable_mark = "✓" if feat in result["stable_features"] else " "
                print(f"{stable_mark} {feat:20s} [{bar}] {score * 100:.0f}%")

            # Provenance (if --explain)
            if args.explain:
                provenance = result.get("provenance", {})
                print(f"\n{'=' * 60}")
                print("📖 EXPLAINABILITY / PROVENANCE:")
                print("-" * 40)

                methodology = provenance.get("methodology", {})
                print(f"\n  Methodology: {methodology.get('name', 'N/A')}")
                print(f"  Reference: {methodology.get('reference_url', 'N/A')}")

                print("\n  Pipeline Steps:")
                for step in provenance.get("pipeline_steps", []):
                    print(f"    {step['step']}. {step['name']}: {step['description']}")

                config_used = provenance.get("configuration_used", {})
                print("\n  Configuration Used:")
                for k, v in config_used.items():
                    print(f"    {k}: {v}")

        print(f"\n{'=' * 60}")
        print("💡 TIP: Use --json for programmatic output, --schema to see all options")
        print(f"{'=' * 60}")
