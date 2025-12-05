"""Schema Intelligence Core Package"""

from .schema_intelligence import (
    ColumnProfiler,
    ColumnProfile,
    SemanticType,
    BusinessEntity,
    summarize_profiles
)
from .dataset_classifier import (
    DatasetClassifier,
    DatasetClassification,
    BusinessDomain,
    MLTaskType
)
from .applicability_scorer import (
    ApplicabilityScorer,
    AnalyticsEngine,
    EngineScore,
    EngineRequirements,
    get_top_recommendations
)
from .column_mapper import (
    ColumnMapper,
    ColumnMapping,
    MappingResult,
    auto_map_columns
)

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
from .premium_models import (
    PremiumResult,
    Variant,
    PlainEnglishSummary,
    TechnicalExplanation,
    ExplanationStep,
    HoldoutResult,
    FeatureImportance,
    ConfigParameter,
    TaskType,
    Confidence,
    MultiEngineResult,
    EngineAnalysisRequest
)
from .premium_base import PremiumEngineBase
from .business_translator import BusinessTranslator

# Standard Engine Framework imports
from .standard_base import (
    StandardEngineBase,
    TimingMixin,
    ValidationMixin,
    LoggingMixin
)

# Gemma Summarizer for fallback
from .gemma_summarizer import (
    GemmaSummarizer,
    needs_gemma_fallback
)
