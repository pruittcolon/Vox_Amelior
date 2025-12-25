"""Schema Intelligence Core Package"""

from .applicability_scorer import (
    AnalyticsEngine,
    ApplicabilityScorer,
    EngineRequirements,
    EngineScore,
    get_top_recommendations,
)
from .column_mapper import ColumnMapper, ColumnMapping, MappingResult, auto_map_columns
from .dataset_classifier import BusinessDomain, DatasetClassification, DatasetClassifier, MLTaskType
from .schema_intelligence import BusinessEntity, ColumnProfile, ColumnProfiler, SemanticType, summarize_profiles

__all__ = [
    # Schema Intelligence
    "ColumnProfiler",
    "ColumnProfile",
    "SemanticType",
    "BusinessEntity",
    "summarize_profiles",
    # Dataset Classifier
    "DatasetClassifier",
    "DatasetClassification",
    "BusinessDomain",
    "MLTaskType",
    # Applicability Scorer
    "ApplicabilityScorer",
    "AnalyticsEngine",
    "EngineScore",
    "EngineRequirements",
    "get_top_recommendations",
    # Column Mapper
    "ColumnMapper",
    "ColumnMapping",
    "MappingResult",
    "auto_map_columns",
    # Premium Engine Framework
    "PremiumEngineBase",
    "PremiumResult",
    "Variant",
    "PlainEnglishSummary",
    "TechnicalExplanation",
    "ExplanationStep",
    "HoldoutResult",
    "FeatureImportance",
    "ConfigParameter",
    "TaskType",
    "Confidence",
    "BusinessTranslator",
    "MultiEngineResult",
    "EngineAnalysisRequest",
    # Standard Engine Framework
    "StandardEngineBase",
    "TimingMixin",
    "ValidationMixin",
    "LoggingMixin",
    # Gemma Fallback Summarizer
    "GemmaSummarizer",
    "needs_gemma_fallback",
]

# Premium Engine Framework imports
from .business_translator import BusinessTranslator

# Gemma Summarizer for fallback
from .gemma_summarizer import GemmaSummarizer, needs_gemma_fallback
from .premium_base import PremiumEngineBase
from .premium_models import (
    Confidence,
    ConfigParameter,
    EngineAnalysisRequest,
    ExplanationStep,
    FeatureImportance,
    HoldoutResult,
    MultiEngineResult,
    PlainEnglishSummary,
    PremiumResult,
    TaskType,
    TechnicalExplanation,
    Variant,
)

# Standard Engine Framework imports
from .standard_base import LoggingMixin, StandardEngineBase, TimingMixin, ValidationMixin
