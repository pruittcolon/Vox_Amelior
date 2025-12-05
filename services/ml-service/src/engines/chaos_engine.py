"""
Chaos Engine - Non-Linear Dynamics Analysis

Detects "invisible" relationships that standard linear correlation misses.
Uses advanced techniques like Distance Correlation and Mutual Information.

Author: Enterprise Analytics Team
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_regression
from core import (
    ColumnProfiler,
    ColumnMapper,
    SemanticType,
    BusinessEntity,
    EngineRequirements,
    DatasetClassifier
)

# Import premium models for standardized output
from core.premium_models import (
    PremiumResult,
    Variant,
    PlainEnglishSummary,
    TechnicalExplanation,
    ExplanationStep,
    HoldoutResult,
    FeatureImportance,
    ConfigParameter,
    TaskType,
    Confidence
)
from core.business_translator import BusinessTranslator

# Configuration schema for Chaos Engine
CHAOS_CONFIG_SCHEMA = {
    "mi_threshold": {"type": "float", "min": 0.1, "max": 1.0, "default": 0.4, "description": "Mutual information threshold for hidden link detection"},
    "pearson_threshold": {"type": "float", "min": 0.1, "max": 0.9, "default": 0.6, "description": "Pearson threshold below which relationship is considered 'hidden'"},
    "n_top_links": {"type": "int", "min": 1, "max": 50, "default": 10, "description": "Number of top hidden links to return"},
}


class ChaosEngine:
    """
    Chaos Engine: Detects non-linear relationships using:
    1. Distance Correlation (dCor) - detects ANY dependency (linear + non-linear)
    2. Mutual Information (MI) - information-theoretic measure of dependency
    3. Comparison with Pearson - identifies "hidden links"
    """
    
    def __init__(self):
        self.name = "Chaos Engine (Non-Linear Dynamics)"
        self.profiler = ColumnProfiler()
        self.mapper = ColumnMapper()
    
    def get_requirements(self) -> EngineRequirements:
        """Define what this engine needs to run"""
        return EngineRequirements(
            required_semantics=[SemanticType.NUMERIC_CONTINUOUS],
            optional_semantics={},
            required_entities=[],
            preferred_domains=[],
            applicable_tasks=[],
            min_rows=50,
            min_numeric_cols=2
        )
    
    def analyze(
        self, 
        df: pd.DataFrame, 
        target_column: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Main analysis method
        
        Args:
            df: Input dataset
            target_column: Optional target column (unused for Chaos, but required by registry)
            config: Optional configuration (hints, skip_profiling, etc.)
        
        Returns:
            {
                'complexity_matrix': {...},  # dCor + MI matrices
                'hidden_links': [...],       # Relationships with high MI, low Pearson
                'insights': [...],           # Business-friendly explanations
                'column_mappings': {...}     # Detected columns
            }
        """
        config = config or {}
        
        # 1. Profile dataset
        if not config.get('skip_profiling', False):
            profiles = self.profiler.profile_dataset(df)
        else:
            profiles = {}
        
        # 2. Select numeric columns
        numeric_cols = self._get_numeric_columns(df, profiles)
        
        if len(numeric_cols) < 2:
            return {
                'error': 'Insufficient numeric columns',
                'message': f'Chaos Engine requires at least 2 numeric columns, found {len(numeric_cols)}',
                'insights': []
            }
        
        # 3. Calculate correlation matrices
        pearson_matrix = self._calculate_pearson(df[numeric_cols])
        dcor_matrix = self._calculate_distance_correlation(df[numeric_cols])
        mi_matrix = self._calculate_mutual_information(df[numeric_cols])
        
        # 4. Identify "hidden links" (high MI/dCor, low Pearson)
        hidden_links = self._find_hidden_links(
            numeric_cols, 
            pearson_matrix, 
            dcor_matrix, 
            mi_matrix
        )
        
        # 5. Generate insights
        insights = self._generate_insights(
            numeric_cols,
            pearson_matrix,
            dcor_matrix,
            mi_matrix,
            hidden_links
        )
        
        return {
            'engine': self.name,
            'numeric_columns': numeric_cols,
            'pearson_correlation': pearson_matrix,
            'distance_correlation': dcor_matrix,
            'mutual_information': mi_matrix,
            'hidden_links': hidden_links,
            'insights': insights,
            'complexity_matrix': {
                'columns': numeric_cols,
                'dcor': dcor_matrix,
                'mi': mi_matrix
            }
        }
    
    def _get_numeric_columns(self, df: pd.DataFrame, profiles: Dict) -> List[str]:
        """Extract numeric continuous columns"""
        numeric_cols = []
        
        for col in df.columns:
            # Check if numeric dtype
            if pd.api.types.is_numeric_dtype(df[col]):
                # Exclude pure identifiers (high uniqueness)
                if profiles:
                    profile = profiles.get(col)
                    if profile and profile.semantic_type == SemanticType.IDENTIFIER:
                        continue
                numeric_cols.append(col)
        
        return numeric_cols
    
    def _calculate_pearson(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate Pearson correlation matrix"""
        matrix = {}
        cols = df.columns.tolist()
        
        for i, col1 in enumerate(cols):
            for col2 in cols[i+1:]:
                # Handle missing values
                valid_mask = df[col1].notna() & df[col2].notna()
                if valid_mask.sum() < 10:
                    matrix[f"{col1}__{col2}"] = 0.0
                    continue
                
                try:
                    corr, _ = pearsonr(df.loc[valid_mask, col1], df.loc[valid_mask, col2])
                    matrix[f"{col1}__{col2}"] = corr if not np.isnan(corr) else 0.0
                except:
                    matrix[f"{col1}__{col2}"] = 0.0
        
        return matrix
    
    def _calculate_distance_correlation(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate Distance Correlation (dCor)
        
        dCor = 0 implies independence
        dCor = 1 implies perfect dependency (linear OR non-linear)
        """
        matrix = {}
        cols = df.columns.tolist()
        
        for i, col1 in enumerate(cols):
            for col2 in cols[i+1:]:
                # Handle missing values
                valid_mask = df[col1].notna() & df[col2].notna()
                if valid_mask.sum() < 10:
                    matrix[f"{col1}__{col2}"] = 0.0
                    continue
                
                try:
                    X = df.loc[valid_mask, col1].values.reshape(-1, 1)
                    Y = df.loc[valid_mask, col2].values.reshape(-1, 1)
                    
                    dcor = self._distance_correlation_statistic(X.flatten(), Y.flatten())
                    matrix[f"{col1}__{col2}"] = dcor
                except Exception as e:
                    matrix[f"{col1}__{col2}"] = 0.0
        
        return matrix
    
    def _distance_correlation_statistic(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute Distance Correlation statistic
        Implementation based on Székely, Rizzo, Bakirov (2007)
        """
        n = len(x)
        
        # Pairwise distances
        a = squareform(pdist(x.reshape(-1, 1), 'euclidean'))
        b = squareform(pdist(y.reshape(-1, 1), 'euclidean'))
        
        # Double centering
        A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
        B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()
        
        # Distance covariance
        dcov_sq = (A * B).sum() / (n * n)
        
        # Distance variance
        dvar_x = (A * A).sum() / (n * n)
        dvar_y = (B * B).sum() / (n * n)
        
        # Distance correlation
        if dvar_x > 0 and dvar_y > 0:
            dcor = np.sqrt(dcov_sq) / np.sqrt(np.sqrt(dvar_x) * np.sqrt(dvar_y))
        else:
            dcor = 0.0
        
        return dcor
    
    def _calculate_mutual_information(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate Mutual Information (MI) - information-theoretic measure
        
        MI = 0 implies independence
        MI > 0 implies dependency (can detect non-linear relationships)
        """
        matrix = {}
        cols = df.columns.tolist()
        
        for i, col1 in enumerate(cols):
            for col2 in cols[i+1:]:
                # Handle missing values
                valid_mask = df[col1].notna() & df[col2].notna()
                if valid_mask.sum() < 10:
                    matrix[f"{col1}__{col2}"] = 0.0
                    continue
                
                try:
                    X = df.loc[valid_mask, col1].values.reshape(-1, 1)
                    y = df.loc[valid_mask, col2].values
                    
                    # Use sklearn's mutual_info_regression
                    mi = mutual_info_regression(X, y, random_state=42)[0]
                    
                    # Normalize to [0, 1] range (approximate)
                    # True MI can be unbounded, but we normalize for interpretability
                    mi_normalized = min(mi / 2.0, 1.0)  # Heuristic normalization
                    
                    matrix[f"{col1}__{col2}"] = mi_normalized
                except Exception as e:
                    matrix[f"{col1}__{col2}"] = 0.0
        
        return matrix
    
    def _find_hidden_links(
        self,
        cols: List[str],
        pearson: Dict[str, float],
        dcor: Dict[str, float],
        mi: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """
        Identify "hidden links" - relationships missed by Pearson correlation
        
        Criteria: High MI (>0.5) AND Low Pearson (<0.3)
        """
        hidden_links = []
        
        for pair_key in mi.keys():
            mi_val = mi[pair_key]
            pearson_val = abs(pearson[pair_key])
            dcor_val = dcor[pair_key]
            
            # High non-linear dependency, not strongly linear
            # Changed: mi > 0.4 (from 0.5) and abs(pearson) < 0.6 (from 0.3)
            if mi_val > 0.4 and abs(pearson_val) < 0.6:
                col1, col2 = pair_key.split('__')
                
                hidden_links.append({
                    'column_1': col1,
                    'column_2': col2,
                    'mutual_information': round(mi_val, 3),
                    'distance_correlation': round(dcor_val, 3),
                    'pearson_correlation': round(pearson_val, 3),
                    'type': self._classify_relationship_type(mi_val, dcor_val, pearson_val)
                })
        
        # Sort by MI (strongest first)
        hidden_links.sort(key=lambda x: x['mutual_information'], reverse=True)
        
        return hidden_links
    
    def _classify_relationship_type(self, mi: float, dcor: float, pearson: float) -> str:
        """Classify the type of non-linear relationship"""
        if mi > 0.7 and abs(pearson) < 0.2:
            return "Highly Non-Linear"
        elif mi > 0.5 and abs(pearson) < 0.3:
            return "Moderately Non-Linear"
        else:
            return "Weak Non-Linear"
    
    def _generate_insights(
        self,
        cols: List[str],
        pearson: Dict[str, float],
        dcor: Dict[str, float],
        mi: Dict[str, float],
        hidden_links: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate business-friendly insights"""
        insights = []
        
        # Summary
        insights.append(
            f"📊 **Chaos Analysis Complete**: Analyzed {len(cols)} numeric variables "
            f"to detect non-linear relationships invisible to standard correlation."
        )
        
        # Hidden links
        if hidden_links:
            insights.append(
                f"🔗 **{len(hidden_links)} Hidden Relationship(s) Discovered**: "
                f"These exhibit strong non-linear dependencies that linear models would miss."
            )
            
            # Detail top 3 hidden links
            for i, link in enumerate(hidden_links[:3], 1):
                col1 = link['column_1']
                col2 = link['column_2']
                mi_val = link['mutual_information']
                pearson_val = link['pearson_correlation']
                rel_type = link['type']
                
                insights.append(
                    f"   {i}. **'{col1}' ↔ '{col2}'**: {rel_type} pattern detected. "
                    f"Mutual Information: {mi_val:.2f} (strong), "
                    f"Pearson Correlation: {pearson_val:.2f} (weak). "
                    f"This suggests a complex behavioral pattern like threshold effects, "
                    f"cyclical dynamics, or exponential relationships."
                )
        else:
            insights.append(
                "✓ **No Hidden Links Found**: All relationships are approximately linear. "
                "Standard correlation analysis is sufficient for this dataset."
            )
        
        # Strategic recommendation
        if hidden_links:
            insights.append(
                "💡 **Strategic Insight**: Consider non-linear models (Random Forest, "
                "Gradient Boosting, Neural Networks) to capture these complex patterns. "
                "Linear regression will underperform on this data."
            )
        
        return insights

    # =========================================================================
    # PREMIUM OUTPUT: Standardized PremiumResult format for unified UI
    # =========================================================================
    
    def run_premium(
        self, 
        df: pd.DataFrame, 
        config: Optional[Dict[str, Any]] = None
    ) -> PremiumResult:
        """
        Run Chaos analysis and return standardized PremiumResult.
        
        This wrapper converts the internal analyze() output to the unified
        PremiumResult format used by the frontend predictions.html.
        
        Args:
            df: Input DataFrame
            config: Configuration overrides
            
        Returns:
            PremiumResult with all components for unified UI
        """
        import time
        start_time = time.time()
        
        # Get raw analysis results
        config = config or {}
        raw_result = self.analyze(df, config)
        
        # Handle errors
        if 'error' in raw_result:
            return self._error_to_premium_result(raw_result, df, config, start_time)
        
        # Convert hidden links to variants
        variants = self._convert_hidden_links_to_variants(raw_result.get('hidden_links', []))
        
        # Get best variant
        best_variant = variants[0] if variants else self._create_fallback_variant()
        
        # Build feature importance from correlations
        feature_importance = self._build_feature_importance_from_correlations(
            raw_result.get('numeric_columns', []),
            raw_result.get('mutual_information', {}),
            raw_result.get('pearson_correlation', {})
        )
        
        # Build plain English summary
        summary = self._build_summary(raw_result, len(variants))
        
        # Build technical explanation
        explanation = self._build_explanation()
        
        # Build config schema
        config_schema = self._build_config_schema()
        
        execution_time = time.time() - start_time
        
        return PremiumResult(
            engine_name="chaos",
            engine_display_name="Chaos Detector",
            engine_icon="🌀",
            task_type=TaskType.DETECTION,
            target_column=None,  # Chaos doesn't target a specific column
            columns_analyzed=raw_result.get('numeric_columns', []),
            row_count=len(df),
            variants=variants,
            best_variant=best_variant,
            feature_importance=feature_importance,
            summary=summary,
            explanation=explanation,
            holdout=None,  # Chaos doesn't use holdout validation
            config_used=config,
            config_schema=config_schema,
            execution_time_seconds=execution_time,
            warnings=[]
        )
    
    def _convert_hidden_links_to_variants(self, hidden_links: List[Dict]) -> List[Variant]:
        """Convert hidden links to Variant objects"""
        variants = []
        
        for i, link in enumerate(hidden_links):
            # Score based on MI strength
            mi_score = link.get('mutual_information', 0)
            gemma_score = int(min(100, mi_score * 120))  # Scale MI to 1-100
            
            col1 = link.get('column_1', 'Unknown')
            col2 = link.get('column_2', 'Unknown')
            rel_type = link.get('type', 'Non-Linear')
            
            variant = Variant(
                rank=i + 1,
                gemma_score=gemma_score,
                cv_score=mi_score,
                variant_type="hidden_relationship",
                model_name="Distance Correlation + Mutual Information",
                features_used=[col1, col2],
                interpretation=f"{rel_type} relationship: {col1} ↔ {col2} (MI: {mi_score:.2f})",
                details={
                    'column_1': col1,
                    'column_2': col2,
                    'mutual_information': link.get('mutual_information', 0),
                    'distance_correlation': link.get('distance_correlation', 0),
                    'pearson_correlation': link.get('pearson_correlation', 0),
                    'relationship_type': rel_type
                }
            )
            variants.append(variant)
        
        # If no hidden links found, create a "no findings" variant
        if not variants:
            variants.append(Variant(
                rank=1,
                gemma_score=75,  # Good - no hidden complexity
                cv_score=0.0,
                variant_type="no_hidden_links",
                model_name="All Relationships Linear",
                features_used=[],
                interpretation="No hidden non-linear relationships detected. Linear models are appropriate.",
                details={'status': 'linear_data'}
            ))
        
        return variants
    
    def _build_feature_importance_from_correlations(
        self,
        columns: List[str],
        mi_matrix: Dict[str, float],
        pearson_matrix: Dict[str, float]
    ) -> List[FeatureImportance]:
        """Build feature importance based on correlation involvement"""
        translator = BusinessTranslator()
        
        # Count how many strong correlations each column is involved in
        col_involvement = {col: {'mi_sum': 0, 'count': 0} for col in columns}
        
        for pair_key, mi_val in mi_matrix.items():
            col1, col2 = pair_key.split('__')
            if col1 in col_involvement:
                col_involvement[col1]['mi_sum'] += mi_val
                col_involvement[col1]['count'] += 1
            if col2 in col_involvement:
                col_involvement[col2]['mi_sum'] += mi_val
                col_involvement[col2]['count'] += 1
        
        features = []
        for col in columns:
            data = col_involvement[col]
            avg_mi = data['mi_sum'] / max(1, data['count'])
            
            # Determine if this column is involved in hidden links
            if avg_mi > 0.4:
                explanation = f"{col} has complex non-linear relationships with other variables"
                impact = "mixed"
            elif avg_mi > 0.2:
                explanation = f"{col} shows moderate dependency patterns"
                impact = "positive"
            else:
                explanation = f"{col} has mostly independent behavior"
                impact = "positive"
            
            features.append(FeatureImportance(
                name=col,
                stability=avg_mi * 100,
                importance=avg_mi,
                impact=impact,
                explanation=explanation
            ))
        
        # Sort by importance
        features.sort(key=lambda f: f.importance, reverse=True)
        return features
    
    def _build_summary(self, raw_result: Dict, num_hidden: int) -> PlainEnglishSummary:
        """Build plain English summary"""
        num_cols = len(raw_result.get('numeric_columns', []))
        hidden_links = raw_result.get('hidden_links', [])
        
        # Use actual hidden_links length, not num_hidden parameter (which could be variants count)
        actual_hidden = len(hidden_links)
        
        if actual_hidden > 0 and hidden_links:
            # Found hidden relationships
            top_link = hidden_links[0]
            headline = f"Found {actual_hidden} hidden non-linear relationship(s) in your data"
            explanation = (
                f"Your data contains complex patterns that standard correlation analysis would miss. "
                f"The strongest hidden link is between '{top_link.get('column_1', 'Unknown')}' and '{top_link.get('column_2', 'Unknown')}' "
                f"(Mutual Information: {top_link.get('mutual_information', 0):.2f}). "
                f"Linear models will underperform on this data."
            )
            recommendation = "Use non-linear models like Random Forest or Gradient Boosting to capture these patterns."
            confidence = Confidence.HIGH if actual_hidden >= 3 else Confidence.MEDIUM
        else:
            headline = "Your data relationships are linear - no hidden complexity found"
            explanation = (
                f"Analyzed {num_cols} numeric variables and found no significant non-linear patterns. "
                f"All relationships can be captured by standard correlation analysis. "
                f"Linear models will work well on this data."
            )
            recommendation = "Linear regression or similar simple models are appropriate for your data."
            confidence = Confidence.HIGH
        
        return PlainEnglishSummary(
            headline=headline,
            explanation=explanation,
            recommendation=recommendation,
            confidence=confidence
        )
    
    def _build_explanation(self) -> TechnicalExplanation:
        """Build technical explanation"""
        return TechnicalExplanation(
            methodology_name="Distance Correlation + Mutual Information Analysis",
            methodology_url="https://en.wikipedia.org/wiki/Distance_correlation",
            steps=[
                ExplanationStep(1, "Data Selection", "Identified all numeric columns for analysis"),
                ExplanationStep(2, "Pearson Correlation", "Calculated linear correlations between all pairs"),
                ExplanationStep(3, "Distance Correlation", "Computed dCor to detect ANY dependency (linear + non-linear)"),
                ExplanationStep(4, "Mutual Information", "Used information-theoretic measure to quantify dependency"),
                ExplanationStep(5, "Hidden Link Detection", "Found pairs with high MI/dCor but low Pearson (hidden non-linear patterns)")
            ],
            limitations=[
                "Requires at least 50 samples for reliable results",
                "Computationally intensive for many columns",
                "May detect spurious relationships in small samples"
            ]
        )
    
    def _build_config_schema(self) -> List[ConfigParameter]:
        """Convert CHAOS_CONFIG_SCHEMA to list of ConfigParameter objects"""
        params = []
        for name, schema in CHAOS_CONFIG_SCHEMA.items():
            param_range = None
            if 'min' in schema and 'max' in schema:
                param_range = [schema['min'], schema['max']]
            
            params.append(ConfigParameter(
                name=name,
                type=schema.get('type', 'float'),
                default=schema.get('default'),
                range=param_range,
                description=schema.get('description', '')
            ))
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
            details={"error": "No results generated"}
        )
    
    def _error_to_premium_result(
        self, 
        raw_result: Dict, 
        df: pd.DataFrame, 
        config: Dict,
        start_time: float
    ) -> PremiumResult:
        """Convert error result to PremiumResult"""
        import time
        
        return PremiumResult(
            engine_name="chaos",
            engine_display_name="Chaos Detector",
            engine_icon="🌀",
            task_type=TaskType.DETECTION,
            target_column=None,
            columns_analyzed=list(df.columns),
            row_count=len(df),
            variants=[],
            best_variant=self._create_fallback_variant(),
            feature_importance=[],
            summary=PlainEnglishSummary(
                headline="Analysis could not be completed",
                explanation=raw_result.get('message', 'Unknown error occurred'),
                recommendation="Please check your data format and try again.",
                confidence=Confidence.LOW
            ),
            explanation=TechnicalExplanation(
                methodology_name="Distance Correlation + Mutual Information",
                methodology_url=None,
                steps=[],
                limitations=["Analysis failed - see error message"]
            ),
            holdout=None,
            config_used=config,
            config_schema=self._build_config_schema(),
            execution_time_seconds=time.time() - start_time,
            warnings=[raw_result.get('message', 'Error occurred')]
        )


# CLI entry point
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python chaos_engine.py <csv_file>")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    df = pd.read_csv(csv_file)
    
    engine = ChaosEngine()
    result = engine.analyze(df)
    
    print(f"\n{'='*60}")
    print(f"CHAOS ENGINE RESULTS: {csv_file}")
    print(f"{'='*60}\n")
    
    for insight in result['insights']:
        print(insight)
    
    print(f"\n{'='*60}")
    print("HIDDEN LINKS:")
    print(f"{'='*60}\n")
    
    for link in result['hidden_links']:
        print(f"{link['column_1']} <-> {link['column_2']}")
        print(f"  MI: {link['mutual_information']:.3f} | dCor: {link['distance_correlation']:.3f} | Pearson: {link['pearson_correlation']:.3f}")
        print(f"  Type: {link['type']}\n")
