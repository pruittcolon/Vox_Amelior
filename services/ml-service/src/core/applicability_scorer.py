"""
Applicability Scorer - Engine Recommendation System

Scores and ranks analytics engines based on dataset characteristics to determine
which analyses are most applicable and likely to provide valuable insights.

Author: Nemo Server ML Team
Date: 2025-11-27
"""

from dataclasses import dataclass
from typing import Dict, List
from enum import Enum
import logging

from .schema_intelligence import ColumnProfile, SemanticType, BusinessEntity
from .dataset_classifier import DatasetClassification, BusinessDomain, MLTaskType

logger = logging.getLogger(__name__)


class AnalyticsEngine(Enum):
    """Available analytics engines in the platform."""
    # Money & Financial
    COST_OPTIMIZATION = "cost-optimization"
    ROI_PREDICTION = "roi-prediction"
    SPEND_PATTERNS = "spend-patterns"
    BUDGET_VARIANCE = "budget-variance"
    PROFIT_MARGINS = "profit-margins"
    CASH_FLOW = "cash-flow"
    
    # Business Intelligence
    CUSTOMER_LTV = "customer-ltv"
    REVENUE_FORECAST = "revenue-forecast"
    INVENTORY_OPTIMIZATION = "inventory-optimization"
    PRICING_STRATEGY = "pricing-strategy"
    MARKET_BASKET = "market-basket"
    RESOURCE_UTILIZATION = "resource-utilization"
    
    # Advanced ML
    STATISTICAL = "statistical"
    UNIVERSAL_GRAPHS = "universal-graphs"
    PREDICTIVE = "predictive"
    ANOMALY_DETECTION = "anomaly-detection"
    CLUSTERING = "clustering"
    TREND_ANALYSIS = "trend-analysis"
    RAG_EVALUATION = "rag-evaluation"


@dataclass
class EngineRequirements:
    """Requirements for an analytics engine to be applicable."""
    required_semantics: List[SemanticType]
    optional_semantics: Dict[str, List[SemanticType]]
    required_entities: List[BusinessEntity]
    preferred_domains: List[BusinessDomain]
    applicable_tasks: List[MLTaskType]
    min_rows: int = 50
    min_numeric_cols: int = 0
    min_categorical_cols: int = 0
    requires_temporal: bool = False


@dataclass
class EngineScore:
    """Score and metadata for an analytics engine."""
    engine: AnalyticsEngine
    applicability_score: float
    confidence: float
    reason: str
    missing_requirements: List[str]
    suggested_columns: Dict[str, str]  # semantic_role -> column_name


class ApplicabilityScorer:
    """
    Score and rank analytics engines based on dataset characteristics.
    
    Determines which engines are most applicable to a given dataset and provides
    confidence scores and recommendations.
    """
    
    def __init__(self):
        """Initialize the applicability scorer with engine requirements."""
        self.engine_requirements = self._define_requirements()
    
    def _define_requirements(self) -> Dict[AnalyticsEngine, EngineRequirements]:
        """Define semantic requirements for each analytics engine."""
        return {
            AnalyticsEngine.COST_OPTIMIZATION: EngineRequirements(
                required_semantics=[SemanticType.NUMERIC_CONTINUOUS],
                optional_semantics={"grouping": [SemanticType.CATEGORICAL]},
                required_entities=[BusinessEntity.COST],
                preferred_domains=[BusinessDomain.FINANCE, BusinessDomain.OPERATIONS],
                applicable_tasks=[MLTaskType.REGRESSION, MLTaskType.CLUSTERING],
                min_rows=100,
                min_numeric_cols=1
            ),
            
            AnalyticsEngine.ROI_PREDICTION: EngineRequirements(
                required_semantics=[SemanticType.NUMERIC_CONTINUOUS],
                optional_semantics={"temporal": [SemanticType.TEMPORAL]},
                required_entities=[],  # Can work with any numeric financial data
                preferred_domains=[BusinessDomain.FINANCE, BusinessDomain.SALES],
                applicable_tasks=[MLTaskType.REGRESSION, MLTaskType.FORECASTING],
                min_rows=20,
                min_numeric_cols=2
            ),
            
            AnalyticsEngine.SPEND_PATTERNS: EngineRequirements(
                required_semantics=[SemanticType.NUMERIC_CONTINUOUS, SemanticType.TEMPORAL],
                optional_semantics={"grouping": [SemanticType.CATEGORICAL]},
                required_entities=[],
                preferred_domains=[BusinessDomain.FINANCE, BusinessDomain.OPERATIONS],
                applicable_tasks=[MLTaskType.TIME_SERIES, MLTaskType.ANOMALY_DETECTION],
                min_rows=50,
                requires_temporal=True
            ),
            
            AnalyticsEngine.BUDGET_VARIANCE: EngineRequirements(
                required_semantics=[SemanticType.NUMERIC_CONTINUOUS],
                optional_semantics={"grouping": [SemanticType.CATEGORICAL]},
                required_entities=[],
                preferred_domains=[BusinessDomain.FINANCE],
                applicable_tasks=[MLTaskType.REGRESSION],
                min_rows=12,
                min_numeric_cols=2  # Need budget and actual
            ),
            
            AnalyticsEngine.REVENUE_FORECAST: EngineRequirements(
                required_semantics=[SemanticType.NUMERIC_CONTINUOUS, SemanticType.TEMPORAL],
                optional_semantics={},
                required_entities=[],
                preferred_domains=[BusinessDomain.SALES, BusinessDomain.FINANCE, BusinessDomain.ECOMMERCE],
                applicable_tasks=[MLTaskType.TIME_SERIES, MLTaskType.FORECASTING],
                min_rows=50,
                requires_temporal=True
            ),
            
            AnalyticsEngine.CUSTOMER_LTV: EngineRequirements(
                required_semantics=[SemanticType.CATEGORICAL],  # Customer ID
                optional_semantics={"monetary": [SemanticType.NUMERIC_CONTINUOUS], 
                                  "temporal": [SemanticType.TEMPORAL]},
                required_entities=[BusinessEntity.CUSTOMER_ID],
                preferred_domains=[BusinessDomain.ECOMMERCE, BusinessDomain.SALES],
                applicable_tasks=[MLTaskType.CLASSIFICATION, MLTaskType.REGRESSION],
                min_rows=100
            ),
            
            AnalyticsEngine.MARKET_BASKET: EngineRequirements(
                required_semantics=[SemanticType.CATEGORICAL],
                optional_semantics={},
                required_entities=[],
                preferred_domains=[BusinessDomain.ECOMMERCE, BusinessDomain.RETAIL],
                applicable_tasks=[MLTaskType.ASSOCIATION_MINING],
                min_rows=100,
                min_categorical_cols=2
            ),
            
            AnalyticsEngine.ANOMALY_DETECTION: EngineRequirements(
                required_semantics=[SemanticType.NUMERIC_CONTINUOUS],
                optional_semantics={"temporal": [SemanticType.TEMPORAL]},
                required_entities=[],
                preferred_domains=[],  # Universal
                applicable_tasks=[MLTaskType.ANOMALY_DETECTION],
                min_rows=100,
                min_numeric_cols=1
            ),
            
            AnalyticsEngine.CLUSTERING: EngineRequirements(
                required_semantics=[SemanticType.NUMERIC_CONTINUOUS],
                optional_semantics={"categorical": [SemanticType.CATEGORICAL]},
                required_entities=[],
                preferred_domains=[],  # Universal
                applicable_tasks=[MLTaskType.CLUSTERING],
                min_rows=50,
                min_numeric_cols=2
            ),
            
            AnalyticsEngine.STATISTICAL: EngineRequirements(
                required_semantics=[SemanticType.NUMERIC_CONTINUOUS],
                optional_semantics={},
                required_entities=[],
                preferred_domains=[],  # Universal
                applicable_tasks=[],  # Works with any
                min_rows=10,
                min_numeric_cols=1
            ),
        }
    
    def score_engines(self, profiles: Dict[str, ColumnProfile],
                     classification: DatasetClassification) -> List[EngineScore]:
        """
        Score all engines and return ranked list of applicable ones.
        
        Args:
            profiles: Column profiles from ColumnProfiler
            classification: Dataset classification from DatasetClassifier
            
        Returns:
            List of EngineScore objects sorted by applicability_score (descending)
        """
        logger.info("Scoring applicability of all analytics engines")
        
        scores = []
        for engine, requirements in self.engine_requirements.items():
            score = self._score_engine(engine, requirements, profiles, classification)
            if score.applicability_score > 0.3:  # Only include promising engines
                scores.append(score)
        
        # Sort by applicability score
        scores.sort(key=lambda x: x.applicability_score, reverse=True)
        
        logger.info(f"Found {len(scores)} applicable engines")
        return scores
    
    def _score_engine(self, engine: AnalyticsEngine, requirements: EngineRequirements,
                     profiles: Dict[str, ColumnProfile],
                     classification: DatasetClassification) -> EngineScore:
        """
        Score a single engine's applicability.
        
        Args:
            engine: Engine to score
            requirements: Engine requirements
            profiles: Column profiles
            classification: Dataset classification
            
        Returns:
            EngineScore with applicability and confidence
        """
        score = 0.0
        confidence = 1.0
        missing = []
        suggested_columns = {}
        
        # Check row count requirement
        if classification.row_count < requirements.min_rows:
            missing.append(f"Need {requirements.min_rows} rows, have {classification.row_count}")
            confidence *= 0.5
        else:
            score += 0.2
        
        # Check required semantics
        available_semantics = {prof.semantic_type for prof in profiles.values()}
        for req_semantic in requirements.required_semantics:
            matching_cols = [col for col, prof in profiles.items() 
                           if prof.semantic_type == req_semantic]
            if matching_cols:
                score += 0.3 / len(requirements.required_semantics)
                # Suggest the best matching column
                best_col = max(matching_cols, key=lambda c: profiles[c].confidence)
                suggested_columns[req_semantic.value] = best_col
            else:
                missing.append(f"Missing {req_semantic.value} column")
                confidence *= 0.6
        
        # Check required entities
        available_entities = {prof.detected_entity for prof in profiles.values()}
        for req_entity in requirements.required_entities:
            if req_entity in available_entities:
                score += 0.2 / max(len(requirements.required_entities), 1)
            else:
                missing.append(f"Missing {req_entity.value} column")
                confidence *= 0.7
        
        # Check domain match
        if classification.domain in requirements.preferred_domains:
            score += 0.2
        elif requirements.preferred_domains:  # Has preferences but doesn't match
            confidence *= 0.8
        
        # Check task type match
        task_match = any(task in classification.task_types for task in requirements.applicable_tasks)
        if task_match:
            score += 0.1
        elif requirements.applicable_tasks:  # Has task requirements but doesn't match
            confidence *= 0.9
        
        # Apply confidence penalty
        final_score = score * confidence
        
        # Generate reason
        reason = self._generate_reason(engine, final_score, missing, classification)
        
        return EngineScore(
            engine=engine,
            applicability_score=final_score,
            confidence=confidence,
            reason=reason,
            missing_requirements=missing,
            suggested_columns=suggested_columns
        )
    
    def _generate_reason(self, engine: AnalyticsEngine, score: float,
                        missing: List[str], classification: DatasetClassification) -> str:
        """Generate human-readable reason for the score."""
        if score > 0.8:
            return f"Excellent match! {classification.domain.value.title()} data ideal for {engine.value}"
        elif score > 0.6:
            return f"Good match. {engine.value} analysis recommended"
        elif score > 0.4:
            return f"Possible match. May have limited results"
        else:
            reasons = missing[:2] if missing else ["Insufficient data characteristics"]
            return f"Low applicability: {', '.join(reasons)}"


def get_top_recommendations(profiles: Dict[str, ColumnProfile],
                           classification: DatasetClassification,
                           top_n: int = 5) -> List[EngineScore]:
    """
    Convenience function to get top N engine recommendations.
    
    Args:
        profiles: Column profiles
        classification: Dataset classification
        top_n: Number of top recommendations to return
        
    Returns:
        List of top N EngineScore objects
    """
    scorer = ApplicabilityScorer()
    all_scores = scorer.score_engines(profiles, classification)
    return all_scores[:top_n]
