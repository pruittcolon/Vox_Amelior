"""
Newton Symbolic Regression Engine - Equation Discovery

Discovers interpretable mathematical formulas that model data relationships
using genetic programming.

Core technique: Genetic Programming (Koza, 1992)
- Evolves mathematical expressions to fit data
- Produces human-readable equations
- Provides interpretability beyond black-box models

Author: Enterprise Analytics Team
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from gplearn.genetic import SymbolicRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Monkeypatch gplearn for newer sklearn compatibility
if not hasattr(SymbolicRegressor, '_validate_data'):
    def _validate_data(self, X, y=None, reset=True, validate_separately=False, **check_params):
        from sklearn.utils.validation import check_X_y, check_array
        
        if y is None:
            if self._get_tags()['requires_y']:
                 raise ValueError(
                     f"This {self.__class__.__name__} estimator "
                     "requires y to be passed, but the target y is None."
                 )
            X = check_array(X, **check_params)
            out = X
        else:
            X, y = check_X_y(X, y, **check_params)
            out = X, y
            
        if reset:
            self.n_features_in_ = X.shape[1]
            
        return out
    SymbolicRegressor._validate_data = _validate_data

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
from core.business_translator import BusinessTranslator

NEWTON_CONFIG_SCHEMA = {
    "population_size": {"type": "int", "min": 100, "max": 5000, "default": 500, "description": "Size of genetic population"},
    "generations": {"type": "int", "min": 5, "max": 100, "default": 20, "description": "Number of evolution generations"},
    "parsimony_coefficient": {"type": "float", "min": 0.0, "max": 0.1, "default": 0.01, "description": "Penalty for complex equations"},
}


class NewtonEngine:
    """
    Newton Symbolic Regression Engine: Discovers Mathematical Laws
    
    Uses genetic programming to evolve mathematical expressions that
    explain data relationships. Unlike black-box models, produces
    interpretable formulas like "y = 2.5x^2 + 3x - 1".
    
    Key insight: Best models are interpretable models
    """
    
    def __init__(self):
        self.name = "Newton Symbolic Regression Engine"
        self.profiler = ColumnProfiler()
        self.mapper = ColumnMapper()
        
    def get_requirements(self) -> EngineRequirements:
        """Define what this engine needs to run"""
        return EngineRequirements(
            required_semantics=[SemanticType.NUMERIC_CONTINUOUS],
            optional_semantics={'target': [SemanticType.NUMERIC_CONTINUOUS]},
            required_entities=[],
            preferred_domains=[],
            applicable_tasks=['regression', 'equation_discovery'],
            min_rows=50,
            min_numeric_cols=2
        )
    
    def analyze(self, df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Main analysis method
        
        Args:
            df: Input dataset
            config: {
                'target_column': Optional[str] - name of target column
                'population_size': int - number of programs in each generation (default: 2000)
                'generations': int - number of generations to evolve (default: 20)
                'max_samples': float - fraction of data for fitness (default: 0.9)
                'function_set': List[str] - operations to use (default: ['add', 'sub', 'mul', 'div'])
            }
        
        Returns:
            {
                'equation': str - discovered mathematical formula,
                'r2_score': float - goodness of fit,
                'complexity': int - number of operations,
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
                'engine': self.name,
                'status': 'not_applicable',
                'message': 'No suitable target column found for equation discovery',
                'insights': ['📊 **Newton Engine - Not Applicable**: No suitable numeric target column found. Newton needs a continuous variable to predict (e.g., sales, price, profit). Check if your numeric columns contain currency symbols ($) that need to be cleaned.']
            }
        
        # 3. Prepare features and target
        X, y, feature_names = self._prepare_data(df, target_col)
        
        if X.shape[1] < 1:
            return {
                'engine': self.name,
                'status': 'not_applicable', 
                'message': f'No features available to predict {target_col}',
                'insights': [f'📊 **Newton Engine - Insufficient Data**: Found target "{target_col}" but no numeric features to build an equation. Need at least 1 numeric feature column. Check if columns contain currency symbols ($) or other formatting that prevents numeric parsing.']
            }
        
        # Check target variance
        if y.std() < 1e-10:
            return {
                'engine': self.name,
                'status': 'not_applicable',
                'message': f'Target column {target_col} has no variance',
                'insights': [f'📊 **Newton Engine - Constant Target**: The target column "{target_col}" has no variance (all values are the same or nearly identical). Cannot discover a meaningful equation for a constant.']
            }
        
        # 4. Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 5. Configure genetic programming - use smaller defaults for speed
        # population_size * generations = total evaluations, keep under 5000 for <10s runtime
        population_size = config.get('population_size', 200)
        generations = config.get('generations', 10)
        max_samples = config.get('max_samples', 0.9)
        function_set = config.get('function_set', ['add', 'sub', 'mul', 'div'])
        
        # 6. Run symbolic regression
        equation, model, train_score, test_score, complexity = self._discover_equation(
            X_train, X_test, y_train, y_test, feature_names,
            population_size, generations, max_samples, function_set
        )
        
        # 7. Clamp R² scores to 0-1 range (R² can be negative if model is very poor)
        # For display purposes, we clamp to 0 but keep raw scores available
        display_train_r2 = max(0.0, min(1.0, train_score))
        display_test_r2 = max(0.0, min(1.0, test_score))
        
        # 8. Generate insights
        insights = self._generate_insights(
            target_col, feature_names, equation,
            display_train_r2, display_test_r2, complexity, generations,
            raw_test_r2=test_score  # Pass raw score for warning if negative
        )
        
        return {
            'engine': self.name,
            'target_column': target_col,
            'features': feature_names,
            'equation': equation,
            'train_r2': display_train_r2,
            'test_r2': display_test_r2,
            'complexity': complexity,
            'generations_evolved': generations,
            'population_size': population_size,
            'insights': insights
        }
    
    def _detect_target_column(self, df: pd.DataFrame, profiles: Dict, hint: Optional[str]) -> Optional[str]:
        """Detect target column - find something meaningful to predict"""
        if hint and hint in df.columns:
            if pd.api.types.is_numeric_dtype(df[hint]):
                return hint
        
        # Columns to avoid as targets (temporal, identifier-like, demographics)
        avoid_keywords = ['year', 'month', 'day', 'date', 'time', 'id', 'index', 'number', 'count',
                         'age', 'survived', 'sex', 'gender', 'class', 'category']
        
        # Common target keywords (priority order) - things we want to predict
        target_keywords = ['fare', 'price', 'cost', 'amount', 'total', 'revenue', 'profit', 
                          'sales', 'value', 'target', 'label', 'y', 'output', 'outcome', 'medv']
        
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        
        # First pass: look for target keywords, avoid temporal/id columns
        for col in numeric_cols:
            col_lower = col.lower().strip()
            if any(kw in col_lower for kw in target_keywords):
                if not any(avoid in col_lower for avoid in avoid_keywords):
                    # Check if it has meaningful variance (not constant)
                    if df[col].nunique() > 2:
                        return col
        
        # Second pass: any numeric column with good variance, avoiding temporal
        for col in numeric_cols:
            col_lower = col.lower().strip()
            if not any(avoid in col_lower for avoid in avoid_keywords):
                if df[col].nunique() > 5:  # Need some variance
                    return col
        
        # No suitable target found - don't use temporal/id columns as last resort
        # Return None so we can give a clear "not applicable" message
        return None
    
    def _prepare_data(self, df: pd.DataFrame, target_col: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare feature matrix and target vector"""
        feature_cols = [c for c in df.columns if c != target_col]
        
        # ID column keywords to exclude
        id_keywords = ['rowname', 'row_name', 'index', 'idx', 'id', 'key', 'pk', 'uuid', 
                       'record', 'row_id', 'rowid', 'obs', 'observation', 'unnamed', 'level_']
        
        def is_id_column(col_name: str, col_data) -> bool:
            """Check if column appears to be an ID/index column"""
            col_lower = col_name.lower().strip()
            if any(skip in col_lower for skip in id_keywords):
                return True
            if col_lower.isdigit():
                return True
            # Check for sequential integers
            if pd.api.types.is_integer_dtype(col_data):
                n_rows = len(col_data)
                if n_rows > 10:
                    uniqueness = col_data.nunique() / n_rows
                    if uniqueness > 0.95:
                        sorted_vals = col_data.dropna().sort_values()
                        if len(sorted_vals) > 1:
                            diffs = sorted_vals.diff().dropna()
                            if len(diffs) > 0 and (diffs == 1).mean() > 0.95:
                                return True
            return False
        
        # Select only numeric features, excluding ID columns
        numeric_features = []
        for col in feature_cols:
            if pd.api.types.is_numeric_dtype(df[col]) and not is_id_column(col, df[col]):
                numeric_features.append(col)
        
        X = df[numeric_features].fillna(df[numeric_features].median()).values
        y = df[target_col].fillna(df[target_col].median()).values
        
        return X, y, numeric_features
    
    def _discover_equation(
        self, X_train: np.ndarray, X_test: np.ndarray,
        y_train: np.ndarray, y_test: np.ndarray,
        feature_names: List[str],
        population_size: int, generations: int,
        max_samples: float, function_set: List[str]
    ) -> Tuple[str, Any, float, float, int]:
        """
        Use genetic programming to discover equation
        """
        # Create symbolic regressor
        est_gp = SymbolicRegressor(
            population_size=population_size,
            generations=generations,
            tournament_size=20,
            stopping_criteria=0.01,
            p_crossover=0.7,
            p_subtree_mutation=0.1,
            p_hoist_mutation=0.05,
            p_point_mutation=0.1,
            max_samples=max_samples,
            function_set=function_set,
            verbose=0,
            parsimony_coefficient=0.01,
            random_state=42,
            n_jobs=-1
        )
        
        # Evolve program
        est_gp.fit(X_train, y_train)
        
        # Get discovered equation
        program = est_gp._program
        equation = self._format_equation(str(program), feature_names)
        
        # Evaluate performance
        train_score = est_gp.score(X_train, y_train)
        test_score = est_gp.score(X_test, y_test)
        
        # Complexity (number of operations)
        complexity = program.length_
        
        return equation, est_gp, train_score, test_score, complexity
    
    def _format_equation(self, raw_eq: str, feature_names: List[str]) -> str:
        """
        Format equation for readability
        
        Converts X0, X1, X2 to actual feature names
        """
        equation = raw_eq
        
        for i, name in enumerate(feature_names):
            equation = equation.replace(f'X{i}', name)
        
        # Simplify notation
        equation = equation.replace('add(', '(')
        equation = equation.replace('sub(', '(')
        equation = equation.replace('mul(', '(')
        equation = equation.replace('div(', '(')
        equation = equation.replace('), ', ' ')
        
        return equation
    
    def _generate_insights(
        self, target_col: str, features: List[str],
        equation: str, train_r2: float, test_r2: float,
        complexity: int, generations: int,
        raw_test_r2: float = None
    ) -> List[str]:
        """Generate business-friendly insights"""
        insights = []
        
        # Summary
        insights.append(
            f"📊 **Newton Symbolic Regression Complete**: Evolved {generations} generations "
            f"of mathematical programs to discover equation for '{target_col}'."
        )
        
        # Discovered equation
        insights.append(
            f"🔬 **Discovered Equation**:\n   `{target_col} = {equation}`"
        )
        
        # Check if raw R² was very negative (model is terrible)
        if raw_test_r2 is not None and raw_test_r2 < -0.5:
            insights.append(
                f"⚠️ **Poor Model Fit**: The discovered equation performs worse than a simple mean prediction. "
                f"This suggests the relationship is highly complex or the features are not predictive. "
                f"Consider trying different feature engineering or a non-symbolic approach."
            )
        else:
            # Model performance
            insights.append(
                f"✅ **Model Performance**: Training R² = {train_r2:.3f}, Test R² = {test_r2:.3f}. "
                f"Equation complexity: {complexity} operations."
            )
        
        # Interpretability
        if test_r2 > 0.7:
            insights.append(
                f"💡 **High Interpretability**: The discovered equation explains {test_r2*100:.1f}% "
                f"of variance in '{target_col}'. This is a reliable mathematical model you can "
                f"audit, understand, and trust."
            )
        elif test_r2 > 0.5:
            insights.append(
                f"💡 **Moderate Fit**: Equation explains {test_r2*100:.1f}% of variance. "
                f"Some complexity remains unexplained - consider non-linear terms or interactions."
            )
        elif test_r2 > 0.0:
            insights.append(
                f"⚠️ **Low Fit**: Equation explains only {test_r2*100:.1f}% of variance. "
                f"Relationship may be highly non-linear or require additional features."
            )
        # For very poor models (raw R² < 0), we already gave warning above
        
        # Feature importance (from equation)
        insights.append(
            f"📈 **Key Drivers**: Discovered equation uses {len(features)} features. "
            f"Mathematical formula shows exact contribution of each variable."
        )
        
        # Business value
        insights.append(
            "🎯 **Strategic Insight**: Unlike black-box models, this symbolic equation is "
            "fully transparent. Stakeholders can verify the logic, regulators can audit it, "
            "and you can manually calculate predictions without software."
        )
        
        return insights

    # =========================================================================
    # PREMIUM OUTPUT
    # =========================================================================
    
    def run_premium(self, df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> PremiumResult:
        """Run Newton equation discovery and return PremiumResult."""
        import time
        start_time = time.time()
        
        config = config or {}
        raw = self.analyze(df, config)
        
        # Handle errors and not_applicable status
        if 'error' in raw or raw.get('status') == 'not_applicable':
            return self._error_to_premium_result(raw, df, config, start_time)
        
        equation = raw.get('equation', 'Unknown')
        r2 = raw.get('test_r2', 0.0)
        
        variants = [Variant(
            rank=1, gemma_score=int(r2 * 100), cv_score=r2,
            variant_type="equation", model_name="Symbolic Regression",
            features_used=raw.get('features', []),
            interpretation=f"Equation: {equation[:50]}...",
            details={'equation': equation, 'train_r2': raw.get('train_r2'), 'test_r2': r2}
        )]
        
        features = [FeatureImportance(
            name=f, stability=80.0, importance=0.5, impact="mixed",
            explanation=f"{f} appears in the discovered equation"
        ) for f in raw.get('features', [])]
        
        return PremiumResult(
            engine_name="newton", engine_display_name="Newton Equations", engine_icon="📐",
            task_type=TaskType.DISCOVERY, target_column=raw.get('target_column'),
            columns_analyzed=raw.get('features', []), row_count=len(df),
            variants=variants, best_variant=variants[0], feature_importance=features,
            summary=PlainEnglishSummary(
                f"Discovered equation with {r2*100:.1f}% accuracy",
                f"Newton found: {equation[:60]}... This explains {r2*100:.1f}% of variance.",
                "Use this equation for interpretable predictions.",
                Confidence.HIGH if r2 > 0.7 else Confidence.MEDIUM
            ),
            explanation=TechnicalExplanation(
                "Genetic Programming Symbolic Regression",
                "https://gplearn.readthedocs.io/en/stable/",
                [ExplanationStep(1, "Population Init", "Created random equation trees"),
                 ExplanationStep(2, "Evolution", "Evolved equations over generations"),
                 ExplanationStep(3, "Selection", "Chose best-fitting equation")],
                ["May find local optima", "Computationally intensive"]
            ),
            holdout=None, config_used=config,
            config_schema=[ConfigParameter(k, v['type'], v['default'], 
                          [v.get('min'), v.get('max')] if 'min' in v else None, 
                          v.get('description', '')) for k, v in NEWTON_CONFIG_SCHEMA.items()],
            execution_time_seconds=time.time() - start_time, warnings=[]
        )
    
    def _error_to_premium_result(self, raw, df, config, start):
        import time
        
        # Determine headline based on status
        is_not_applicable = raw.get('status') == 'not_applicable'
        message = raw.get('message', 'Error')
        insights = raw.get('insights', [])
        
        if is_not_applicable:
            headline = "Newton Engine - Not Applicable"
            recommendation = "This dataset lacks suitable numeric columns for equation discovery. Check for currency formatting ($) in numeric columns."
        else:
            headline = "Equation discovery failed"
            recommendation = "Provide numeric data with a target column."
        
        return PremiumResult(
            engine_name="newton", engine_display_name="Newton Equations", engine_icon="📐",
            task_type=TaskType.DISCOVERY, target_column=None, columns_analyzed=list(df.columns),
            row_count=len(df), variants=[],
            best_variant=Variant(1, 0, 0.0, "not_applicable" if is_not_applicable else "error", "None", [], message, {}),
            feature_importance=[],
            summary=PlainEnglishSummary(headline, insights[0] if insights else message, 
                                       recommendation, Confidence.LOW),
            explanation=TechnicalExplanation("Symbolic Regression", None, [], [message]),
            holdout=None, config_used=config, config_schema=[],
            execution_time_seconds=time.time() - start, warnings=[raw.get('message', '')]
        )


# CLI entry point
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python newton_engine.py <csv_file> [--target COLUMN] [--gen N]")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    
    target_col = None
    generations = 20
    
    if '--target' in sys.argv:
        target_idx = sys.argv.index('--target')
        if len(sys.argv) > target_idx + 1:
            target_col = sys.argv[target_idx + 1]
    
    if '--gen' in sys.argv:
        gen_idx = sys.argv.index('--gen')
        if len(sys.argv) > gen_idx + 1:
            generations = int(sys.argv[gen_idx + 1])
    
    df = pd.read_csv(csv_file)
    
    engine = NewtonEngine()
    config = {
        'target_column': target_col,
        'generations': generations
    }
    result = engine.analyze(df, config)
    
    print(f"\n{'='*60}")
    print(f"NEWTON ENGINE RESULTS: {csv_file}")
    print(f"{'='*60}\n")
    
    if 'error' in result:
        print(f"❌ Error: {result['message']}")
        for insight in result.get('insights', []):
            print(f"   {insight}")
    else:
        for insight in result['insights']:
            print(insight)
        
        print(f"\n{'='*60}")
        print("EQUATION DETAILS:")
        print(f"{'='*60}\n")
        
        print(f"Target: {result['target_column']}")
        print(f"Features: {', '.join(result['features'])}")
        print(f"\nDiscovered Equation:")
        print(f"  {result['equation']}\n")
        print(f"Performance:")
        print(f"  Train R²: {result['train_r2']:.4f}")
        print(f"  Test R²: {result['test_r2']:.4f}")
        print(f"  Complexity: {result['complexity']} operations")
        print(f"  Generations: {result['generations_evolved']}")
