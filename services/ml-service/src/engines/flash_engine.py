"""
Flash Counterfactual Engine - Actionable Explanations

Generates "what-if" scenarios to answer: "What changes would flip this prediction?"

Core technique: DiCE (Diverse Counterfactual Explanations)
- Finds minimal changes to input features that change the outcome
- Provides actionable recommendations for interventions
- Ensures diversity in suggested changes

Author: Enterprise Analytics Team
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

import dice_ml
from dice_ml import Dice
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from core import (
    ColumnProfiler,
    ColumnMapper,
    SemanticType,
    BusinessEntity,
    EngineRequirements,
    DatasetClassifier
)

# Import premium models
from core.premium_models import (
    PremiumResult, Variant, PlainEnglishSummary, TechnicalExplanation,
    ExplanationStep, FeatureImportance, ConfigParameter, TaskType, Confidence
)

FLASH_CONFIG_SCHEMA = {
    "num_cfs": {"type": "int", "min": 1, "max": 20, "default": 5, "description": "Number of counterfactuals per query"},
    "desired_class": {"type": "select", "options": ["opposite", "1", "0"], "default": "opposite", "description": "Target class for counterfactuals"},
}


class FlashEngine:
    """
    Flash Counterfactual Engine: Actionable "What-If" Recommendations
    
    Uses DiCE (Diverse Counterfactual Explanations) to generate
    minimal changes that would flip a prediction from negative to positive.
    
    Key insight: Don't just predict - show HOW to change the outcome
    """
    
    def __init__(self):
        self.name = "Flash Counterfactual Engine"
        self.profiler = ColumnProfiler()
        self.mapper = ColumnMapper()
        
    def get_requirements(self) -> EngineRequirements:
        """Define what this engine needs to run"""
        return EngineRequirements(
            required_semantics=[SemanticType.NUMERIC_CONTINUOUS],
            optional_semantics={'target': [SemanticType.CATEGORICAL, SemanticType.NUMERIC_CONTINUOUS]},
            required_entities=[],
            preferred_domains=[],
            applicable_tasks=['classification', 'counterfactual'],
            min_rows=100,
            min_numeric_cols=2
        )
    
    def analyze(self, df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Main analysis method
        
        Args:
            df: Input dataset
            config: {
                'target_column': Optional[str] - name of target column
                'num_counterfactuals': int - number of alternatives to generate (default: 3)
                'desired_outcome': int/str - target outcome to achieve (default: 1)
                'query_instances': int - number of random instances to explain (default: 5)
            }
        
        Returns:
            {
                'target_column': str,
                'counterfactuals': [...],
                'insights': [...]
            }
        """
        config = config or {}
        
        # 1. Profile dataset
        if not config.get('skip_profiling', False):
            profiles = self.profiler.profile_dataset(df)
        else:
            profiles = {}
        
        # 2. Detect target column
        target_col = self._detect_target_column(df, profiles, config.get('target_column'))
        
        if not target_col:
            return {
                'error': 'No target column found',
                'message': 'Flash Engine requires a target column for counterfactuals',
                'insights': ['💡 Specify target_column in config']
            }
        
        # 3. Detect task type
        task_type = self._detect_task_type(df, target_col)
        
        if task_type != 'classification':
            return {
                'error': 'Only classification supported',
                'message': 'Flash Engine currently supports binary/multiclass classification',
                'insights': ['💡 Target must be categorical or binary (0/1)']
            }
        
        # 4. Prepare data
        X, y, feature_names, continuous_features = self._prepare_data(df, target_col)
        
        if X.shape[1] < 1:
            return {
                'error': 'No features found',
                'message': 'Need at least 1 feature for counterfactuals',
                'insights': []
            }
        
        # 5. Train model
        model, accuracy = self._train_model(X, y)
        
        # 6. Generate counterfactuals
        num_cfs = config.get('num_counterfactuals', 3)
        desired_outcome = config.get('desired_outcome', 1)
        query_instances = config.get('query_instances', 5)
        
        counterfactuals = self._generate_counterfactuals(
            df, target_col, feature_names, continuous_features,
            model, num_cfs, desired_outcome, query_instances
        )
        
        # 7. Generate insights
        insights = self._generate_insights(
            target_col, counterfactuals, desired_outcome, accuracy
        )
        
        return {
            'engine': self.name,
            'target_column': target_col,
            'model_accuracy': accuracy,
            'desired_outcome': desired_outcome,
            'counterfactuals_generated': len(counterfactuals),
            'counterfactuals': counterfactuals,
            'insights': insights
        }
    
    def _detect_target_column(self, df: pd.DataFrame, profiles: Dict, hint: Optional[str]) -> Optional[str]:
        """Detect target column"""
        if hint and hint in df.columns:
            return hint
        
        # Common target keywords
        target_keywords = ['target', 'label', 'class', 'y', 'survived', 'outcome', 'churn']
        for col in df.columns:
            if any(kw in col.lower() for kw in target_keywords):
                return col
        
        # Last column
        if len(df.columns) > 2:
            return df.columns[-1]
        
        return None
    
    def _detect_task_type(self, df: pd.DataFrame, target_col: str) -> str:
        """Detect if classification or regression"""
        target = df[target_col]
        
        # Check if categorical
        if not pd.api.types.is_numeric_dtype(target):
            return 'classification'
        
        # Check uniqueness
        uniqueness = target.nunique() / len(target)
        if uniqueness < 0.05 or target.nunique() < 20:
            return 'classification'
        
        return 'regression'
    
    def _prepare_data(self, df: pd.DataFrame, target_col: str) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """Prepare feature matrix and target vector"""
        feature_cols = [c for c in df.columns if c != target_col]
        
        # Select numeric features (including booleans converted to int)
        numeric_features = []
        for col in feature_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_features.append(col)
            elif pd.api.types.is_bool_dtype(df[col]) or df[col].dtype == 'bool':
                # Convert boolean to int
                df[col] = df[col].astype(int)
                numeric_features.append(col)
        
        # Also handle object columns that are actually boolean strings
        for col in feature_cols:
            if col not in numeric_features and df[col].dtype == 'object':
                unique_vals = df[col].dropna().unique()
                if len(unique_vals) <= 2 and set(str(v).lower() for v in unique_vals) <= {'true', 'false', '1', '0', 'yes', 'no'}:
                    df[col] = df[col].map(lambda x: 1 if str(x).lower() in ['true', '1', 'yes'] else 0)
                    numeric_features.append(col)
        
        X = df[numeric_features].fillna(df[numeric_features].median()).values
        y = df[target_col].values
        
        # Convert target to integer if needed
        if y.dtype == 'object':
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y = le.fit_transform(y)
        elif y.dtype == 'bool':
            y = y.astype(int)
        
        return X, y, numeric_features, numeric_features  # All numeric are continuous
    
    def _train_model(self, X: np.ndarray, y: np.ndarray) -> Tuple[Any, float]:
        """Train classification model"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return model, accuracy
    
    def _generate_counterfactuals(
        self, df: pd.DataFrame, target_col: str,
        feature_names: List[str], continuous_features: List[str],
        model: Any, num_cfs: int, desired_outcome: int,
        query_instances: int
    ) -> List[Dict[str, Any]]:
        """
        Generate counterfactual explanations using DiCE
        """
        # Prepare data for DiCE - make a copy and convert booleans
        feature_df = df[feature_names + [target_col]].copy()
        
        # Convert boolean columns to int (DiCE requires numeric types)
        for col in feature_df.columns:
            if feature_df[col].dtype == 'bool' or pd.api.types.is_bool_dtype(feature_df[col]):
                feature_df[col] = feature_df[col].astype(int)
        
        # Create DiCE data object
        d = dice_ml.Data(
            dataframe=feature_df,
            continuous_features=continuous_features,
            outcome_name=target_col
        )
        
        # Create DiCE model
        m = dice_ml.Model(model=model, backend='sklearn')
        
        # Create DiCE explainer
        exp = Dice(d, m, method='random')
        
        # Select query instances (instances with undesired outcome)
        undesired_mask = df[target_col] != desired_outcome
        if undesired_mask.sum() == 0:
            return []  # All instances already have desired outcome
        
        undesired_indices = df[undesired_mask].index[:query_instances].tolist()
        
        counterfactuals = []
        
        for idx in undesired_indices:
            try:
                # Get original instance
                query_instance = feature_df.loc[[idx], feature_names]
                
                # Generate counterfactuals
                dice_exp = exp.generate_counterfactuals(
                    query_instance,
                    total_CFs=num_cfs,
                    desired_class=desired_outcome
                )
                
                # Extract counterfactuals
                cf_df = dice_exp.cf_examples_list[0].final_cfs_df
                
                if cf_df is not None and len(cf_df) > 0:
                    # Calculate changes
                    original = query_instance.iloc[0]
                    
                    for cf_idx in range(len(cf_df)):
                        cf = cf_df.iloc[cf_idx]
                        changes = {}
                        
                        for feat in feature_names:
                            if original[feat] != cf[feat]:
                                changes[feat] = {
                                    'from': float(original[feat]),
                                    'to': float(cf[feat]),
                                    'delta': float(cf[feat] - original[feat])
                                }
                        
                        if changes:  # Only add if there are actual changes
                            counterfactuals.append({
                                'original_index': int(idx),
                                'counterfactual_id': cf_idx,
                                'changes': changes,
                                'num_features_changed': len(changes)
                            })
            
            except Exception as e:
                continue  # Skip problematic instances
        
        return counterfactuals
    
    def _generate_insights(
        self, target_col: str, counterfactuals: List[Dict],
        desired_outcome: int, accuracy: float
    ) -> List[str]:
        """Generate business-friendly insights"""
        insights = []
        
        # Summary
        insights.append(
            f"📊 **Flash Counterfactual Analysis Complete**: Generated actionable "
            f"'what-if' recommendations to achieve '{target_col}' = {desired_outcome}."
        )
        
        # Model performance
        insights.append(
            f"✅ **Model Accuracy**: {accuracy*100:.1f}%. Counterfactuals based on "
            f"Random Forest predictions."
        )
        
        # Counterfactuals found
        if counterfactuals:
            insights.append(
                f"🔄 **{len(counterfactuals)} Counterfactual(s) Discovered**: "
                f"These show minimal changes needed to achieve desired outcome."
            )
            
            # Analyze common changes
            all_changed_features = {}
            for cf in counterfactuals:
                for feat in cf['changes'].keys():
                    all_changed_features[feat] = all_changed_features.get(feat, 0) + 1
            
            # Most frequently changed features
            sorted_features = sorted(all_changed_features.items(), key=lambda x: x[1], reverse=True)
            
            insights.append(
                f"📈 **Key Intervention Levers**: Most frequently changed features across counterfactuals:"
            )
            for i, (feat, count) in enumerate(sorted_features[:3], 1):
                pct = count / len(counterfactuals) * 100
                insights.append(f"   {i}. '{feat}': Changed in {pct:.0f}% of counterfactuals")
            
            # Example counterfactual
            if len(counterfactuals) > 0:
                example = counterfactuals[0]
                insights.append(
                    f"\n💡 **Example Actionable Recommendation** (Instance #{example['original_index']}):"
                )
                for feat, change in list(example['changes'].items())[:3]:
                    insights.append(
                        f"   • Change '{feat}' from {change['from']:.2f} → {change['to']:.2f} "
                        f"(Δ{change['delta']:+.2f})"
                    )
                
                insights.append(
                    f"   ➜ Result: '{target_col}' changes to {desired_outcome}"
                )
        else:
            insights.append(
                "✓ **No Counterfactuals Needed**: All instances already achieve desired outcome, "
                "or no feasible counterfactuals found."
            )
        
        # Strategic value
        if counterfactuals:
            insights.append(
                "🎯 **Strategic Insight**: These are minimal, actionable interventions. "
                "Unlike feature importance, counterfactuals show EXACTLY what to change "
                "for each individual case. Use for personalized recommendations."
            )
        
        return insights

    # =========================================================================
    # PREMIUM OUTPUT
    # =========================================================================
    
    def run_premium(self, df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> PremiumResult:
        """Run Flash counterfactual analysis and return PremiumResult."""
        import time
        start_time = time.time()
        
        config = config or {}
        raw = self.analyze(df, config)
        
        if 'error' in raw:
            return self._error_to_premium_result(raw, df, config, start_time)
        
        cfs = raw.get('counterfactuals', [])
        variants = []
        for i, cf in enumerate(cfs[:10]):
            variants.append(Variant(
                rank=i + 1, gemma_score=90 - i * 5, cv_score=1 / (cf.get('num_features_changed', 1) + 1),
                variant_type="counterfactual", model_name="DiCE",
                features_used=list(cf.get('changes', {}).keys()),
                interpretation=f"Change {cf.get('num_features_changed', 0)} feature(s) to flip outcome",
                details=cf
            ))
        
        if not variants:
            variants.append(Variant(1, 50, 0.0, "none", "DiCE", [], "No counterfactuals found", {}))
        
        # Feature importance from change frequency
        change_counts = {}
        for cf in cfs:
            for feat in cf.get('changes', {}).keys():
                change_counts[feat] = change_counts.get(feat, 0) + 1
        
        features = [FeatureImportance(
            name=f, stability=80.0, importance=count / max(1, len(cfs)),
            impact="mixed", explanation=f"{f} changed in {count} counterfactual(s)"
        ) for f, count in sorted(change_counts.items(), key=lambda x: -x[1])]
        
        return PremiumResult(
            engine_name="flash", engine_display_name="Flash What-If", engine_icon="⚡",
            task_type=TaskType.EXPLANATION, target_column=raw.get('target_column'),
            columns_analyzed=raw.get('features', []), row_count=len(df),
            variants=variants, best_variant=variants[0], feature_importance=features,
            summary=PlainEnglishSummary(
                f"Found {len(cfs)} ways to change outcomes",
                f"DiCE generated counterfactual explanations showing minimal changes to flip predictions.",
                "Review suggested changes for actionable interventions.",
                Confidence.HIGH if len(cfs) > 5 else Confidence.MEDIUM
            ),
            explanation=TechnicalExplanation(
                "Diverse Counterfactual Explanations (DiCE)",
                "https://github.com/interpretml/DiCE",
                [ExplanationStep(1, "Model Training", "Trained classifier on data"),
                 ExplanationStep(2, "Query Selection", "Selected instances for counterfactual generation"),
                 ExplanationStep(3, "Optimization", "Found minimal feature changes to flip outcome")],
                ["May suggest infeasible changes", "Assumes feature independence"]
            ),
            holdout=None, config_used=config,
            config_schema=[ConfigParameter(k, v['type'], v['default'], 
                          v.get('options') or ([v.get('min'), v.get('max')] if 'min' in v else None), 
                          v.get('description', '')) for k, v in FLASH_CONFIG_SCHEMA.items()],
            execution_time_seconds=time.time() - start_time, warnings=[]
        )
    
    def _error_to_premium_result(self, raw, df, config, start):
        import time
        return PremiumResult(
            engine_name="flash", engine_display_name="Flash What-If", engine_icon="⚡",
            task_type=TaskType.EXPLANATION, target_column=None, columns_analyzed=list(df.columns),
            row_count=len(df), variants=[],
            best_variant=Variant(1, 0, 0.0, "error", "None", [], raw.get('message', 'Error'), {}),
            feature_importance=[],
            summary=PlainEnglishSummary("What-if analysis failed", raw.get('message', 'Error'), 
                                       "Provide classification data.", Confidence.LOW),
            explanation=TechnicalExplanation("DiCE", None, [], ["Analysis failed"]),
            holdout=None, config_used=config, config_schema=[],
            execution_time_seconds=time.time() - start, warnings=[raw.get('message', '')]
        )


# CLI entry point
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python flash_engine.py <csv_file> [--target COLUMN] [--outcome VALUE]")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    
    target_col = None
    desired_outcome = 1
    
    if '--target' in sys.argv:
        target_idx = sys.argv.index('--target')
        if len(sys.argv) > target_idx + 1:
            target_col = sys.argv[target_idx + 1]
    
    if '--outcome' in sys.argv:
        outcome_idx = sys.argv.index('--outcome')
        if len(sys.argv) > outcome_idx + 1:
            desired_outcome = int(sys.argv[outcome_idx + 1])
    
    df = pd.read_csv(csv_file)
    
    engine = FlashEngine()
    config = {
        'target_column': target_col,
        'desired_outcome': desired_outcome,
        'num_counterfactuals': 3,
        'query_instances': 5
    }
    result = engine.analyze(df, config)
    
    print(f"\n{'='*60}")
    print(f"FLASH ENGINE RESULTS: {csv_file}")
    print(f"{'='*60}\n")
    
    if 'error' in result:
        print(f"❌ Error: {result['message']}")
        for insight in result.get('insights', []):
            print(f"   {insight}")
    else:
        for insight in result['insights']:
            print(insight)
        
        if result.get('counterfactuals'):
            print(f"\n{'='*60}")
            print("DETAILED COUNTERFACTUALS:")
            print(f"{'='*60}\n")
            
            for cf in result['counterfactuals'][:5]:  # Show first 5
                print(f"Instance #{cf['original_index']} CF#{cf['counterfactual_id']}:")
                print(f"  Features to change: {cf['num_features_changed']}")
                for feat, change in cf['changes'].items():
                    print(f"    • {feat}: {change['from']:.2f} → {change['to']:.2f} (Δ{change['delta']:+.2f})")
                print()
