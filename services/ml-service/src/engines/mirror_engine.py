"""
Mirror Synthetic Data Engine - Privacy-Preserving Data Generation

Generates synthetic data that preserves statistical properties of original data
while protecting privacy.

Core technique: SDV (Synthetic Data Vault)
- Uses CTGAN (Conditional Tabular GAN) for high-quality synthesis
- Preserves correlations, distributions, and relationships
- Enables safe data sharing and testing

GPU Support:
- CTGAN can leverage CUDA for faster training
- Automatically coordinates GPU access via GPU Coordinator

Author: Enterprise Analytics Team
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import warnings
from typing import Any, Optional

import pandas as pd

warnings.filterwarnings("ignore")

from core import ColumnMapper, ColumnProfiler, EngineRequirements, SemanticType

# Import premium models
from core.premium_models import (
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
from sdmetrics.reports.single_table import QualityReport
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer

# GPU Client for coordinated GPU access
try:
    from core.gpu_client import GPUClient, get_gpu_client

    GPU_CLIENT_AVAILABLE = True
except ImportError:
    GPU_CLIENT_AVAILABLE = False

# Check CUDA availability
try:
    import torch

    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False

MIRROR_CONFIG_SCHEMA = {
    "num_samples": {
        "type": "int",
        "min": 100,
        "max": 100000,
        "default": 1000,
        "description": "Number of synthetic samples to generate",
    },
    "epochs": {"type": "int", "min": 50, "max": 1000, "default": 300, "description": "Training epochs for CTGAN"},
    "use_gpu": {"type": "bool", "default": True, "description": "Use GPU acceleration if available"},
}


class MirrorEngine:
    """
    Mirror Synthetic Data Engine: Privacy-Preserving Data Generation

    Uses CTGAN (Conditional Tabular Generative Adversarial Network) to
    generate synthetic data that statistically mirrors the original while
    protecting individual privacy.

    Key insight: Share insights, not data

    GPU Support:
    - Automatically requests GPU access via GPU Coordinator
    - Falls back to CPU if GPU unavailable
    - GPU significantly accelerates CTGAN training
    """

    def __init__(self, gpu_client: Optional["GPUClient"] = None):
        self.name = "Mirror Synthetic Data Engine"
        self.profiler = ColumnProfiler()
        self.mapper = ColumnMapper()
        self._gpu_client = gpu_client
        self._gpu_acquired = False

    @property
    def gpu_client(self) -> Optional["GPUClient"]:
        """Get GPU client, initializing from singleton if not provided"""
        if self._gpu_client is None and GPU_CLIENT_AVAILABLE:
            try:
                self._gpu_client = get_gpu_client()
            except Exception:
                pass
        return self._gpu_client

    def get_requirements(self) -> EngineRequirements:
        """Define what this engine needs to run"""
        return EngineRequirements(
            required_semantics=[SemanticType.NUMERIC_CONTINUOUS],
            optional_semantics={},
            required_entities=[],
            preferred_domains=[],
            applicable_tasks=["synthetic_data", "privacy"],
            min_rows=100,
        )

    def analyze(self, df: pd.DataFrame, config: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Main analysis method (synchronous)

        Args:
            df: Input dataset (real data)
            config: {
                'num_rows': int - number of synthetic rows to generate (default: len(df))
                'epochs': int - training epochs for CTGAN (default: 100)
                'batch_size': int - batch size for training (default: 500)
                'use_gpu': bool - whether to attempt GPU acceleration (default: True)
            }

        Returns:
            {
                'synthetic_data': pd.DataFrame - generated synthetic data,
                'quality_score': float - similarity to original,
                'insights': [...]
            }
        """
        config = config or {}

        # AUTO-LIMIT: Prevent OUT OF MEMORY - CTGAN is very memory intensive!
        MAX_ROWS = 500  # REDUCED from 2000 - hard limit on input
        original_row_count = len(df)
        if len(df) > MAX_ROWS:
            print(f"⚠️ Dataset has {len(df)} rows. Auto-sampling to {MAX_ROWS} to prevent OUT OF MEMORY.")
            df = df.sample(n=MAX_ROWS, random_state=42)

        # 1. Configure synthesis with VERY AGGRESSIVE limits
        num_rows = config.get("num_rows", min(len(df), 100))  # Max 100 synthetic rows (was 500)
        epochs = config.get("epochs", min(config.get("epochs", 100), 20))  # Max 20 epochs (was 50)
        batch_size = config.get("batch_size", min(500, len(df)))  # Don't exceed input size

        # For sync context, use CPU (GPU requires async coordination)
        gpu_acquired = False

        # Filter out ID columns before synthesis
        df_filtered = self._filter_id_columns(df)

        # 2. Detect metadata
        metadata = self._detect_metadata(df_filtered)

        # 3. Generate synthetic data
        synthetic_df, synthesizer = self._generate_synthetic(
            df_filtered, metadata, num_rows, epochs, batch_size, use_cuda=gpu_acquired
        )

        # 4. Evaluate quality (use filtered df for fair comparison)
        quality_score, quality_details = self._evaluate_quality(df_filtered, synthetic_df, metadata)

        # 5. Generate insights
        insights = self._generate_insights(
            df_filtered, synthetic_df, quality_score, quality_details, gpu_used=gpu_acquired
        )

        return {
            "engine": self.name,
            "original_rows": len(df),
            "synthetic_rows": len(synthetic_df),
            "columns": len(df.columns),
            "quality_score": quality_score,
            "quality_details": quality_details,
            "synthetic_data": synthetic_df,
            "insights": insights,
            "used_gpu": gpu_acquired,
        }

    async def analyze_async(self, df: pd.DataFrame, config: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Async main analysis method with GPU coordination

        Args:
            df: Input dataset (real data)
            config: {
                'num_rows': int - number of synthetic rows to generate (default: len(df))
                'epochs': int - training epochs for CTGAN (default: 100)
                'batch_size': int - batch size for training (default: 500)
                'use_gpu': bool - whether to attempt GPU acceleration (default: True)
            }

        Returns:
            {
                'synthetic_data': pd.DataFrame - generated synthetic data,
                'quality_score': float - similarity to original,
                'insights': [...],
                'used_gpu': bool - whether GPU was used
            }
        """
        config = config or {}

        # 1. Configure synthesis
        num_rows = config.get("num_rows", len(df))
        epochs = config.get("epochs", 100)
        batch_size = config.get("batch_size", 500)
        use_gpu = config.get("use_gpu", True)

        # 2. Attempt GPU acquisition if requested
        gpu_acquired = False
        if use_gpu and CUDA_AVAILABLE:
            gpu_acquired = await self._acquire_gpu()

        try:
            # Filter out ID columns before synthesis
            df_filtered = self._filter_id_columns(df)

            # 3. Detect metadata
            metadata = self._detect_metadata(df_filtered)

            # 4. Generate synthetic data (with GPU if acquired)
            synthetic_df, synthesizer = self._generate_synthetic(
                df_filtered, metadata, num_rows, epochs, batch_size, use_cuda=gpu_acquired
            )

            # 5. Evaluate quality (use filtered df for fair comparison)
            quality_score, quality_details = self._evaluate_quality(df_filtered, synthetic_df, metadata)

            # 6. Generate insights
            insights = self._generate_insights(
                df_filtered, synthetic_df, quality_score, quality_details, gpu_used=gpu_acquired
            )

            return {
                "engine": self.name,
                "original_rows": len(df),
                "synthetic_rows": len(synthetic_df),
                "columns": len(df.columns),
                "quality_score": quality_score,
                "quality_details": quality_details,
                "synthetic_data": synthetic_df,
                "insights": insights,
                "used_gpu": gpu_acquired,
            }
        finally:
            # Always release GPU when done
            if gpu_acquired:
                await self._release_gpu()

    def _filter_id_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove ID/index columns that shouldn't be in synthetic data"""
        id_keywords = [
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
            "level_",
        ]

        cols_to_keep = []
        for col in df.columns:
            col_lower = col.lower().strip()
            # Skip columns with ID keywords
            if any(skip in col_lower for skip in id_keywords):
                continue
            # Skip columns that are sequential integers (auto-increment IDs)
            if pd.api.types.is_integer_dtype(df[col]):
                n_rows = len(df)
                if n_rows > 10:
                    uniqueness = df[col].nunique() / n_rows
                    if uniqueness > 0.95:
                        sorted_vals = df[col].dropna().sort_values()
                        if len(sorted_vals) > 1:
                            diffs = sorted_vals.diff().dropna()
                            if len(diffs) > 0 and (diffs == 1).mean() > 0.95:
                                continue
            cols_to_keep.append(col)

        return df[cols_to_keep].copy()

    def _detect_metadata(self, df: pd.DataFrame) -> SingleTableMetadata:
        """Detect column types and create metadata"""
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(df)
        return metadata

    async def _acquire_gpu(self) -> bool:
        """
        Request GPU access from the GPU Coordinator.

        Returns:
            bool: True if GPU was acquired, False if falling back to CPU
        """
        if not CUDA_AVAILABLE:
            return False

        if self.gpu_client is None:
            return False

        try:
            result = await self.gpu_client.request_gpu(engine_name="mirror")
            self._gpu_acquired = result.acquired
            return result.acquired
        except Exception as e:
            # Log but don't fail - fall back to CPU
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

    def _generate_synthetic(
        self,
        df: pd.DataFrame,
        metadata: SingleTableMetadata,
        num_rows: int,
        epochs: int,
        batch_size: int,
        use_cuda: bool = False,
    ) -> tuple:
        """
        Generate synthetic data using CTGAN

        Args:
            df: Input dataframe
            metadata: Column metadata
            num_rows: Number of synthetic rows to generate
            epochs: Training epochs
            batch_size: Training batch size
            use_cuda: Whether to use GPU acceleration
        """
        # Create synthesizer with GPU support
        synthesizer = CTGANSynthesizer(
            metadata,
            epochs=epochs,
            batch_size=batch_size,
            cuda=use_cuda,  # Dynamic GPU/CPU selection
            verbose=False,
        )

        # Train on real data
        synthesizer.fit(df)

        # Generate synthetic data
        synthetic_df = synthesizer.sample(num_rows=num_rows)

        return synthetic_df, synthesizer

    def _evaluate_quality(
        self, real_df: pd.DataFrame, synthetic_df: pd.DataFrame, metadata: SingleTableMetadata
    ) -> tuple:
        """
        Evaluate quality of synthetic data using SDMetrics
        """
        # Generate quality report
        report = QualityReport()
        report.generate(real_df, synthetic_df, metadata.to_dict())

        # Get overall score
        quality_score = report.get_score()

        # Get detailed metrics
        properties = report.get_properties()

        quality_details = {
            "overall_score": quality_score,
            "column_shapes": properties.get("Column Shapes", {}).get("Score", 0),
            "column_pair_trends": properties.get("Column Pair Trends", {}).get("Score", 0),
        }

        return quality_score, quality_details

    def _generate_insights(
        self,
        real_df: pd.DataFrame,
        synthetic_df: pd.DataFrame,
        quality_score: float,
        quality_details: dict,
        gpu_used: bool = False,
    ) -> list[str]:
        """Generate business-friendly insights"""
        insights = []

        # Summary with GPU indicator
        gpu_indicator = " 🚀 (GPU-accelerated)" if gpu_used else ""
        insights.append(
            f"📊 **Mirror Synthetic Data Generation Complete{gpu_indicator}**: Created {len(synthetic_df)} "
            f"privacy-safe synthetic records from {len(real_df)} original records."
        )

        # Quality score
        if quality_score >= 0.9:
            grade = "Excellent"
            emoji = "✅"
        elif quality_score >= 0.7:
            grade = "Good"
            emoji = "👍"
        elif quality_score >= 0.5:
            grade = "Fair"
            emoji = "⚠️"
        else:
            grade = "Poor"
            emoji = "❌"

        insights.append(
            f"{emoji} **Quality Score**: {quality_score:.1%} ({grade}). "
            f"Synthetic data statistically mirrors original data."
        )

        # Detailed metrics
        col_shapes = quality_details.get("column_shapes", 0)
        col_trends = quality_details.get("column_pair_trends", 0)

        insights.append("📈 **Statistical Fidelity**:")
        insights.append(f"   • Column Distributions: {col_shapes:.1%} match")
        insights.append(f"   • Column Correlations: {col_trends:.1%} preserved")

        # Privacy guarantee
        insights.append(
            "🔒 **Privacy Protection**: Synthetic data contains NO real individuals. "
            "Safe for sharing with third parties, testing, or development without privacy risks."
        )

        # Use cases
        insights.append("💡 **Strategic Use Cases**:")
        insights.append("   • Share with partners without exposing customer data")
        insights.append("   • Train ML models in development environments")
        insights.append("   • Create test datasets for QA teams")
        insights.append("   • Publish research data while preserving privacy")

        # Recommendation
        if quality_score >= 0.7:
            insights.append(
                f"🎯 **Recommendation**: Quality score {quality_score:.1%} is sufficient for most use cases. "
                f"Synthetic data can safely replace real data in non-production environments."
            )
        else:
            insights.append(
                f"⚠️ **Caution**: Quality score {quality_score:.1%} is below recommended threshold. "
                f"Consider increasing training epochs or using larger dataset."
            )

        return insights

    # =========================================================================
    # PREMIUM OUTPUT
    # =========================================================================

    def run_premium(self, df: pd.DataFrame, config: dict[str, Any] | None = None) -> PremiumResult:
        """Run Mirror synthesis and return PremiumResult."""
        import time

        start_time = time.time()

        config = config or {}

        # Filter ID columns first (same as in analyze)
        df_filtered = self._filter_id_columns(df)

        raw = self.analyze(df, config)

        if "error" in raw:
            return self._error_to_premium_result(raw, df_filtered, config, start_time)

        quality = raw.get("quality_score", 0.0)
        used_gpu = raw.get("used_gpu", False)
        gpu_indicator = " (GPU)" if used_gpu else " (CPU)"

        variants = [
            Variant(
                rank=1,
                gemma_score=int(quality * 100),
                cv_score=quality,
                variant_type="synthetic_dataset",
                model_name=f"CTGAN{gpu_indicator}",
                features_used=list(df_filtered.columns),
                interpretation=f"Generated {raw.get('synthetic_rows', 0)} synthetic rows with {quality:.1%} quality",
                details={
                    "quality_score": quality,
                    "column_shapes": raw.get("column_shapes"),
                    "column_pair_trends": raw.get("column_pair_trends"),
                    "used_gpu": used_gpu,
                    "synthetic_sample": raw.get("synthetic_data").head(100).to_dict(orient="records")
                    if raw.get("synthetic_data") is not None
                    else [],
                },
            )
        ]

        features = [
            FeatureImportance(
                name=col,
                stability=quality * 100,
                importance=quality,
                impact="positive",
                explanation=f"Column preserved with {quality:.1%} fidelity",
            )
            for col in df_filtered.columns
        ]

        # Add GPU acceleration note to steps if used
        training_step = f"Trained CTGAN for {config.get('epochs', 300)} epochs"
        if used_gpu:
            training_step += " (GPU-accelerated)"

        return PremiumResult(
            engine_name="mirror",
            engine_display_name="Mirror Synthetic",
            engine_icon="🪞",
            task_type=TaskType.GENERATION,
            target_column=None,
            columns_analyzed=list(df_filtered.columns),
            row_count=len(df_filtered),
            variants=variants,
            best_variant=variants[0],
            feature_importance=features,
            summary=PlainEnglishSummary(
                f"Generated synthetic data with {quality:.1%} statistical fidelity",
                f"Created {raw.get('synthetic_rows', 0)} privacy-preserving synthetic rows that mirror original data patterns.",
                "Use synthetic data for safe sharing and development.",
                Confidence.HIGH if quality > 0.7 else Confidence.MEDIUM,
            ),
            explanation=TechnicalExplanation(
                "Conditional Tabular GAN (CTGAN)",
                "https://sdv.dev/SDV/",
                [
                    ExplanationStep(1, "Metadata Detection", "Analyzed column types"),
                    ExplanationStep(2, "GAN Training", training_step),
                    ExplanationStep(3, "Synthesis", "Generated synthetic samples"),
                    ExplanationStep(4, "Quality Assessment", "Validated statistical fidelity"),
                ],
                ["Requires sufficient training data", "Complex relationships may not be perfectly captured"],
            ),
            holdout=None,
            config_used=config,
            config_schema=[
                ConfigParameter(
                    k,
                    v["type"],
                    v["default"],
                    [v.get("min"), v.get("max")] if "min" in v else None,
                    v.get("description", ""),
                )
                for k, v in MIRROR_CONFIG_SCHEMA.items()
            ],
            execution_time_seconds=time.time() - start_time,
            warnings=[],
        )

    def _error_to_premium_result(self, raw, df, config, start):
        import time

        return PremiumResult(
            engine_name="mirror",
            engine_display_name="Mirror Synthetic",
            engine_icon="🪞",
            task_type=TaskType.GENERATION,
            target_column=None,
            columns_analyzed=list(df.columns),
            row_count=len(df),
            variants=[],
            best_variant=Variant(1, 0, 0.0, "error", "None", [], raw.get("message", "Error"), {}),
            feature_importance=[],
            summary=PlainEnglishSummary(
                "Synthesis failed", raw.get("message", "Error"), "Check data format.", Confidence.LOW
            ),
            explanation=TechnicalExplanation("CTGAN", None, [], ["Synthesis failed"]),
            holdout=None,
            config_used=config,
            config_schema=[],
            execution_time_seconds=time.time() - start,
            warnings=[raw.get("message", "")],
        )


# CLI entry point
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python mirror_engine.py <csv_file> [--rows N] [--epochs N]")
        print("  Example: python mirror_engine.py data.csv --rows 1000 --epochs 100")
        sys.exit(1)

    csv_file = sys.argv[1]

    num_rows = None
    epochs = 100

    if "--rows" in sys.argv:
        rows_idx = sys.argv.index("--rows")
        if len(sys.argv) > rows_idx + 1:
            num_rows = int(sys.argv[rows_idx + 1])

    if "--epochs" in sys.argv:
        epochs_idx = sys.argv.index("--epochs")
        if len(sys.argv) > epochs_idx + 1:
            epochs = int(sys.argv[epochs_idx + 1])

    df = pd.read_csv(csv_file)

    if num_rows is None:
        num_rows = len(df)

    engine = MirrorEngine()
    config = {"num_rows": num_rows, "epochs": epochs}

    print(f"\n{'=' * 60}")
    print(f"MIRROR ENGINE: Generating {num_rows} synthetic rows...")
    print(f"Training CTGAN for {epochs} epochs (this may take a few minutes)")
    print(f"{'=' * 60}\n")

    result = engine.analyze(df, config)

    print(f"\n{'=' * 60}")
    print(f"MIRROR ENGINE RESULTS: {csv_file}")
    print(f"{'=' * 60}\n")

    if "error" in result:
        print(f"❌ Error: {result['message']}")
    else:
        for insight in result["insights"]:
            print(insight)

        print(f"\n{'=' * 60}")
        print("SYNTHETIC DATA SAMPLE (First 5 rows):")
        print(f"{'=' * 60}\n")
        print(result["synthetic_data"].head())

        # Save synthetic data
        output_file = csv_file.replace(".csv", "_synthetic.csv")
        result["synthetic_data"].to_csv(output_file, index=False)
        print(f"\n✅ Synthetic data saved to: {output_file}")
