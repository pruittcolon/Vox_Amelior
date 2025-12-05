"""
Premium Engine Models

Standardized dataclasses for all flagship engine outputs.
These ensure consistent structure across Titan, Chaos, Scout, Oracle,
Newton, Flash, Mirror, Chronos, DeepFeature, and Galileo engines.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class TaskType(Enum):
    """Types of ML tasks engines can perform"""
    PREDICTION = "prediction"           # Titan, DeepFeature
    DETECTION = "detection"             # Scout, Chaos
    GENERATION = "generation"           # Mirror
    DISCOVERY = "discovery"             # Newton, Oracle
    FORECASTING = "forecasting"         # Chronos
    EXPLANATION = "explanation"         # Flash
    GRAPH_ANALYSIS = "graph_analysis"   # Galileo


class Confidence(Enum):
    """Confidence levels for results"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class ConfigParameter:
    """Schema for a tunable parameter"""
    name: str
    type: str  # "int", "float", "bool", "select"
    default: Any
    range: Optional[List[Any]] = None  # [min, max] for numbers, options for select
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "default": self.default,
            "range": self.range,
            "description": self.description
        }


@dataclass
class FeatureImportance:
    """Feature importance with business interpretation"""
    name: str
    stability: float  # 0-100, how often feature appears in bootstrap samples
    importance: float  # 0-1, relative importance score
    impact: str  # "positive", "negative", "mixed"
    explanation: str  # Plain English explanation
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "stability": self.stability,
            "importance": self.importance,
            "impact": self.impact,
            "explanation": self.explanation
        }


@dataclass
class ExplanationStep:
    """A single step in the methodology explanation"""
    step_number: int
    title: str
    description: str
    details: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_number": self.step_number,
            "title": self.title,
            "description": self.description,
            "details": self.details or {}
        }


@dataclass
class TechnicalExplanation:
    """
    Technical methodology explanation (no Gemma required).
    Shows users exactly how the analysis was performed.
    """
    methodology_name: str
    methodology_url: Optional[str]  # Link to documentation
    steps: List[ExplanationStep]
    limitations: List[str]
    assumptions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "methodology_name": self.methodology_name,
            "methodology_url": self.methodology_url,
            "steps": [s.to_dict() for s in self.steps],
            "limitations": self.limitations,
            "assumptions": self.assumptions
        }


@dataclass
class PlainEnglishSummary:
    """
    Business-friendly summary of results.
    No ML jargon - written for executives and non-technical users.
    """
    headline: str  # One-line summary, e.g., "Revenue can be predicted with 89% accuracy"
    explanation: str  # 2-3 sentences explaining what this means
    recommendation: str  # Actionable next step
    confidence: Confidence
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "headline": self.headline,
            "explanation": self.explanation,
            "recommendation": self.recommendation,
            "confidence": self.confidence.value
        }


@dataclass
class HoldoutResult:
    """Results from holdout validation (train/test split verification)"""
    train_samples: int
    holdout_samples: int
    holdout_ratio: float
    train_score: float
    holdout_score: float
    metric_name: str  # "accuracy", "r2", "mse", etc.
    passed: bool  # True if holdout score is within acceptable range of train score
    message: str  # "Good generalization" or "Possible overfitting"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "train_samples": self.train_samples,
            "holdout_samples": self.holdout_samples,
            "holdout_ratio": self.holdout_ratio,
            "train_score": self.train_score,
            "holdout_score": self.holdout_score,
            "metric_name": self.metric_name,
            "passed": self.passed,
            "message": self.message
        }


@dataclass
class Variant:
    """
    A single analysis variant.
    Engines generate multiple variants (different models, parameters, feature sets).
    Gemma ranks them 1-100 for business utility.
    """
    rank: int  # Position after Gemma ranking
    gemma_score: int  # 1-100, business utility score from Gemma
    cv_score: float  # Cross-validation score (fallback if no Gemma)
    variant_type: str  # "baseline", "model_variation", "feature_subset", etc.
    model_name: str  # "RandomForest", "GradientBoosting", etc.
    features_used: List[str]
    interpretation: str  # One sentence explaining this variant
    details: Dict[str, Any] = field(default_factory=dict)  # Engine-specific details
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "gemma_score": self.gemma_score,
            "cv_score": self.cv_score,
            "variant_type": self.variant_type,
            "model_name": self.model_name,
            "features_used": self.features_used,
            "interpretation": self.interpretation,
            "details": self.details
        }


@dataclass
class PremiumResult:
    """
    Standardized output format for all flagship engines.
    
    Every engine (Titan, Chaos, Scout, Oracle, Newton, Flash, 
    Mirror, Chronos, DeepFeature, Galileo) returns this format.
    """
    # Identification
    engine_name: str
    engine_display_name: str
    engine_icon: str  # Emoji for UI
    task_type: TaskType
    
    # Context
    target_column: Optional[str]
    columns_analyzed: List[str]
    row_count: int
    
    # Core Results
    variants: List[Variant]
    best_variant: Variant
    feature_importance: List[FeatureImportance]
    
    # Explanations
    summary: PlainEnglishSummary
    explanation: TechnicalExplanation
    
    # Validation
    holdout: Optional[HoldoutResult]
    
    # Configuration
    config_used: Dict[str, Any]
    config_schema: List[ConfigParameter]
    
    # Metadata
    execution_time_seconds: float = 0.0
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "engine_name": self.engine_name,
            "engine_display_name": self.engine_display_name,
            "engine_icon": self.engine_icon,
            "task_type": self.task_type.value,
            "target_column": self.target_column,
            "columns_analyzed": self.columns_analyzed,
            "row_count": self.row_count,
            "variants": [v.to_dict() for v in self.variants],
            "best_variant": self.best_variant.to_dict(),
            "feature_importance": [f.to_dict() for f in self.feature_importance],
            "summary": self.summary.to_dict(),
            "explanation": self.explanation.to_dict(),
            "holdout": self.holdout.to_dict() if self.holdout else None,
            "config_used": self.config_used,
            "config_schema": [c.to_dict() for c in self.config_schema],
            "execution_time_seconds": self.execution_time_seconds,
            "warnings": self.warnings
        }
    
    def get_top_variants(self, n: int = 5) -> List[Variant]:
        """Get top N variants by Gemma score"""
        return sorted(self.variants, key=lambda v: v.gemma_score, reverse=True)[:n]
    
    def get_stable_features(self, threshold: float = 80.0) -> List[FeatureImportance]:
        """Get features with stability above threshold"""
        return [f for f in self.feature_importance if f.stability >= threshold]


@dataclass 
class EngineAnalysisRequest:
    """Request format for premium engine analysis"""
    filename: str
    target_column: Optional[str] = None  # Auto-detect if not provided
    config_overrides: Dict[str, Any] = field(default_factory=dict)
    holdout_ratio: float = 0.2
    n_variants: int = 10
    enable_gemma_ranking: bool = True


@dataclass
class MultiEngineResult:
    """Results from running multiple engines on the same dataset"""
    session_id: str
    filename: str
    target_column: str
    engines_run: List[str]
    results: Dict[str, PremiumResult]  # engine_name -> result
    auto_summary: str  # Combined summary across all engines
    best_engine: str  # Which engine had highest Gemma score
    execution_time_seconds: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "filename": self.filename,
            "target_column": self.target_column,
            "engines_run": self.engines_run,
            "results": {k: v.to_dict() for k, v in self.results.items()},
            "auto_summary": self.auto_summary,
            "best_engine": self.best_engine,
            "execution_time_seconds": self.execution_time_seconds
        }
