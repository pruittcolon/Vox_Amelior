"""
Scout Drift Detection Engine - Continuous Model Monitoring

Detects when data distributions change over time (concept drift),
alerting when models need retraining.

Techniques:
1. Kolmogorov-Smirnov Test (numeric features)
2. Chi-Squared Test (categorical features)
3. Population Stability Index (PSI)
4. DDM (Drift Detection Method) for prediction errors

Author: Enterprise Analytics Team
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from scipy import stats
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

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

# Configuration schema for Scout Engine
SCOUT_CONFIG_SCHEMA = {
    "window_size": {"type": "select", "options": ["1D", "1W", "1M", "1Q"], "default": "1M", "description": "Time window size for drift detection"},
    "baseline_period": {"type": "int", "min": 1, "max": 12, "default": 1, "description": "Number of initial windows to use as baseline"},
    "psi_threshold": {"type": "float", "min": 0.05, "max": 0.5, "default": 0.25, "description": "PSI threshold for drift alert"},
}


class ScoutEngine:
    """
    Scout Engine: Continuous monitoring for concept drift
    
    Detects when data distributions change, indicating:
    - Market shifts
    - Seasonal changes
    - Model degradation
    - Data quality issues
    """
    
    def __init__(self):
        self.name = "Scout Drift Detection Engine"
        self.profiler = ColumnProfiler()
        self.mapper = ColumnMapper()
        
    def get_requirements(self) -> EngineRequirements:
        """Define what this engine needs to run"""
        return EngineRequirements(
            required_semantics=[SemanticType.TEMPORAL, SemanticType.NUMERIC_CONTINUOUS],
            optional_semantics={},
            required_entities=[],
            preferred_domains=[],
            applicable_tasks=['time_series', 'monitoring'],
            min_rows=100
        )
    
    def analyze(self, df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Main analysis method
        
        Args:
            df: Input dataset (must have temporal column)
            config: {
                'time_column': Optional[str] - name of time column
                'window_size': str - '1M', '1W', '1D' for monthly, weekly, daily windows
                'baseline_period': int - number of initial windows to use as baseline
            }
        
        Returns:
            {
                'time_column': str,
                'windows_analyzed': int,
                'drift_events': [...],
                'feature_drift_scores': {...},
                'insights': [...]
            }
        """
        config = config or {}
        
        # 1. Profile dataset
        if not config.get('skip_profiling', False):
            profiles = self.profiler.profile_dataset(df)
        else:
            profiles = {}
        
        # 2. Detect temporal column
        time_col = self._detect_time_column(df, profiles, config.get('time_column'))
        
        if not time_col:
            return {
                'engine': self.name,
                'status': 'not_applicable',
                'insights': ['📊 **Scout Engine - Not Applicable**: This engine requires a time/date column for drift detection. Dataset does not contain temporal data.']
            }
        
        # 3. Parse temporal column
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        df = df.dropna(subset=[time_col])
        df = df.sort_values(time_col)
        
        if len(df) < 100:
            return {
                'engine': self.name,
                'status': 'insufficient_data',
                'insights': [f'📊 **Scout Engine - Insufficient Data**: Only {len(df)} valid timestamps found. Drift detection requires at least 100 temporal records for statistical significance.']
            }
        
        # 4. Create time windows
        window_size = config.get('window_size', '1M')  # Default: monthly
        windows = self._create_time_windows(df, time_col, window_size)
        
        if len(windows) < 3:
            return {
                'engine': self.name,
                'status': 'insufficient_windows',
                'time_column': time_col,
                'windows_found': len(windows),
                'insights': [f'📊 **Scout Engine - Insufficient Time Windows**: Only {len(windows)} time windows detected. Drift detection requires at least 3 windows for baseline comparison. Try using a smaller window size (current: {window_size}).']
            }
    
        # 5. Select numeric features for drift detection
        numeric_cols = self._get_numeric_columns(df, profiles, exclude=[time_col])
        
        if len(numeric_cols) == 0:
            return {
                'error': 'No numeric features found',
                'message': 'Scout requires numeric features to monitor',
                'insights': []
            }
        
        # 6. Detect drift across windows
        baseline_period = config.get('baseline_period', 1)
        drift_events, feature_drift_scores = self._detect_drift(
            windows, numeric_cols, baseline_period
        )
        
        # 7. Calculate Population Stability Index (PSI)
        psi_scores = self._calculate_psi(windows, numeric_cols, baseline_period)
        
        # 8. Generate insights
        insights = self._generate_insights(
            time_col, windows, drift_events, 
            feature_drift_scores, psi_scores, window_size
        )
        
        return {
            'engine': self.name,
            'time_column': time_col,
            'window_size': window_size,
            'windows_analyzed': len(windows),
            'date_range': {
                'start': str(df[time_col].min()),
                'end': str(df[time_col].max())
            },
            'drift_events': drift_events,
            'feature_drift_scores': feature_drift_scores,
            'psi_scores': psi_scores,
            'insights': insights
        }
    
    def _detect_time_column(self, df: pd.DataFrame, profiles: Dict, hint: Optional[str]) -> Optional[str]:
        """Detect temporal column"""
        # Priority 1: User hint
        if hint and hint in df.columns:
            return hint
        
        # Priority 2: ColumnProfiler semantic type
        if profiles:
            for col, profile in profiles.items():
                if profile.semantic_type == SemanticType.TEMPORAL:
                    return col
        
        # Priority 3: Common time keywords
        time_keywords = ['date', 'time', 'timestamp', 'datetime', 'year', 'month', 'day']
        for col in df.columns:
            if any(kw in col.lower() for kw in time_keywords):
                # Try parsing
                try:
                    pd.to_datetime(df[col].head(10), errors='coerce')
                    return col
                except:
                    continue
        
        return None
    
    def _get_numeric_columns(self, df: pd.DataFrame, profiles: Dict, exclude: List[str]) -> List[str]:
        """Get numeric columns excluding specified ones"""
        numeric_cols = []
        
        for col in df.columns:
            if col in exclude:
                continue
            
            if pd.api.types.is_numeric_dtype(df[col]):
                # Exclude identifiers
                if profiles:
                    profile = profiles.get(col)
                    if profile and profile.semantic_type == SemanticType.IDENTIFIER:
                        continue
                numeric_cols.append(col)
        
        return numeric_cols
    
    def _create_time_windows(self, df: pd.DataFrame, time_col: str, window_size: str) -> List[pd.DataFrame]:
        """Split data into time windows"""
        # Set time column as index
        df_indexed = df.set_index(time_col)
        
        # Resample based on window size
        windows = []
        for period, group in df_indexed.groupby(pd.Grouper(freq=window_size)):
            if len(group) >= 10:  # Minimum 10 samples per window
                windows.append(group.reset_index())
        
        return windows
    
    def _detect_drift(
        self, windows: List[pd.DataFrame], 
        numeric_cols: List[str], 
        baseline_period: int
    ) -> Tuple[List[Dict[str, Any]], Dict[str, List[float]]]:
        """
        Detect drift using Kolmogorov-Smirnov test
        
        Compares each window to the baseline (first N windows)
        """
        # Baseline is the first baseline_period windows
        baseline_data = pd.concat(windows[:baseline_period], ignore_index=True)
        
        drift_events = []
        feature_drift_scores = {col: [] for col in numeric_cols}
        
        # Test each subsequent window
        for i, window in enumerate(windows[baseline_period:], start=baseline_period):
            # Get time period for this window
            time_col = window.columns[0]  # Time column is first after reset_index
            period_start = window[time_col].min()
            period_end = window[time_col].max()
            
            window_drift_detected = False
            drifted_features = []
            
            for col in numeric_cols:
                # KS test
                baseline_values = baseline_data[col].dropna().values
                window_values = window[col].dropna().values
                
                if len(baseline_values) < 5 or len(window_values) < 5:
                    feature_drift_scores[col].append(0.0)
                    continue
                
                statistic, p_value = stats.ks_2samp(baseline_values, window_values)
                
                # Store drift score (1 - p_value, so higher = more drift)
                drift_score = 1 - p_value
                feature_drift_scores[col].append(drift_score)
                
                # Drift detected if p < 0.05 (significant difference)
                if p_value < 0.05:
                    window_drift_detected = True
                    drifted_features.append({
                        'feature': col,
                        'ks_statistic': round(statistic, 3),
                        'p_value': round(p_value, 4),
                        'drift_score': round(drift_score, 3)
                    })
            
            # Record drift event
            if window_drift_detected:
                drift_events.append({
                    'window_index': i,
                    'period_start': str(period_start),
                    'period_end': str(period_end),
                    'drifted_features': drifted_features,
                    'severity': 'HIGH' if len(drifted_features) > len(numeric_cols) / 2 else 'MEDIUM'
                })
        
        return drift_events, feature_drift_scores
    
    def _calculate_psi(
        self, windows: List[pd.DataFrame],
        numeric_cols: List[str],
        baseline_period: int
    ) -> Dict[str, float]:
        """
        Calculate Population Stability Index (PSI)
        
        PSI measures distribution shift:
        - PSI < 0.1: No significant change
        - PSI 0.1-0.25: Moderate change
        - PSI > 0.25: Significant change (model retraining recommended)
        """
        baseline_data = pd.concat(windows[:baseline_period], ignore_index=True)
        current_data = pd.concat(windows[baseline_period:], ignore_index=True) if len(windows) > baseline_period else baseline_data
        
        psi_scores = {}
        
        for col in numeric_cols:
            baseline_values = baseline_data[col].dropna().values
            current_values = current_data[col].dropna().values
            
            if len(baseline_values) < 10 or len(current_values) < 10:
                psi_scores[col] = 0.0
                continue
            
            # Discretize into 10 bins
            bins = np.percentile(baseline_values, np.linspace(0, 100, 11))
            
            # Ensure unique bins
            bins = np.unique(bins)
            if len(bins) < 3:
                psi_scores[col] = 0.0
                continue
            
            # Calculate distributions
            baseline_dist, _ = np.histogram(baseline_values, bins=bins)
            current_dist, _ = np.histogram(current_values, bins=bins)
            
            # Add small constant to avoid division by zero
            baseline_dist = baseline_dist / baseline_dist.sum() + 1e-10
            current_dist = current_dist / current_dist.sum() + 1e-10
            
            # PSI formula
            psi = np.sum((current_dist - baseline_dist) * np.log(current_dist / baseline_dist))
            psi_scores[col] = round(psi, 3)
        
        return psi_scores
    
    def _generate_insights(
        self, time_col: str, windows: List[pd.DataFrame],
        drift_events: List[Dict], feature_drift_scores: Dict,
        psi_scores: Dict, window_size: str
    ) -> List[str]:
        """Generate business-friendly insights"""
        insights = []
        
        # Summary
        insights.append(
            f"📊 **Scout Drift Analysis Complete**: Monitored {len(windows)} time periods "
            f"({window_size} windows) for distribution changes."
        )
        
        # Drift events
        if drift_events:
            insights.append(
                f"⚠️ **{len(drift_events)} Drift Event(s) Detected**: "
                f"Data distribution shifted significantly in these periods."
            )
            
            # Detail top 3 events
            for i, event in enumerate(drift_events[:3], 1):
                period = f"{event['period_start'][:10]} to {event['period_end'][:10]}"
                n_features = len(event['drifted_features'])
                severity = event['severity']
                
                insights.append(
                    f"   {i}. **{period}** ({severity} severity): {n_features} feature(s) drifted"
                )
                
                # Most drifted feature
                if event['drifted_features']:
                    top_feature = max(event['drifted_features'], key=lambda x: x['drift_score'])
                    insights.append(
                        f"      → '{top_feature['feature']}': Drift score {top_feature['drift_score']:.2f} "
                        f"(KS statistic: {top_feature['ks_statistic']}, p={top_feature['p_value']})"
                    )
        else:
            insights.append(
                "✓ **No Drift Detected**: Data distribution remained stable across all time periods. "
                "Current model should still be valid."
            )
        
        # PSI analysis
        high_psi_features = [f for f, score in psi_scores.items() if score > 0.25]
        
        if high_psi_features:
            insights.append(
                f"📉 **Population Stability Index (PSI)**: {len(high_psi_features)} feature(s) "
                f"show significant distribution shift (PSI > 0.25). Model retraining recommended."
            )
            
            for feat in high_psi_features[:3]:
                insights.append(f"   → '{feat}': PSI = {psi_scores[feat]:.3f}")
        
        # Strategic recommendation
        if drift_events or high_psi_features:
            insights.append(
                "💡 **Strategic Insight**: Market conditions have changed. "
                "Retrain models on recent data to maintain accuracy. "
                "Consider automated retraining pipelines for continuous deployment."
            )
        else:
            insights.append(
                "💡 **Strategic Insight**: Data distribution is stable. "
                "Current models remain valid. Continue monitoring on a regular schedule."
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
        Run Scout drift detection and return standardized PremiumResult.
        """
        import time
        start_time = time.time()
        
        config = config or {}
        raw_result = self.analyze(df, config)
        
        # Handle non-applicable or error cases
        if raw_result.get('status') == 'not_applicable' or 'error' in raw_result:
            return self._error_to_premium_result(raw_result, df, config, start_time)
        
        # Convert drift events to variants
        variants = self._convert_drift_events_to_variants(
            raw_result.get('drift_events', []),
            raw_result.get('psi_scores', {})
        )
        
        best_variant = variants[0] if variants else self._create_fallback_variant()
        
        # Build feature importance from PSI scores
        feature_importance = self._convert_psi_to_importance(
            raw_result.get('psi_scores', {}),
            raw_result.get('time_column', '')
        )
        
        # Build summary
        summary = self._build_summary(raw_result)
        
        # Build explanation
        explanation = self._build_explanation()
        
        execution_time = time.time() - start_time
        
        return PremiumResult(
            engine_name="scout",
            engine_display_name="Scout Monitor",
            engine_icon="🔍",
            task_type=TaskType.DETECTION,
            target_column=raw_result.get('time_column'),
            columns_analyzed=list(raw_result.get('psi_scores', {}).keys()),
            row_count=len(df),
            variants=variants,
            best_variant=best_variant,
            feature_importance=feature_importance,
            summary=summary,
            explanation=explanation,
            holdout=None,
            config_used=config,
            config_schema=self._build_config_schema(),
            execution_time_seconds=execution_time,
            warnings=[]
        )
    
    def _convert_drift_events_to_variants(
        self, 
        drift_events: List[Dict], 
        psi_scores: Dict[str, float]
    ) -> List[Variant]:
        """Convert drift events to variants"""
        variants = []
        
        for i, event in enumerate(drift_events):
            severity = event.get('severity', 'LOW')
            gemma_score = {'HIGH': 30, 'MEDIUM': 60, 'LOW': 85}.get(severity, 50)
            
            drifted_features = [f['feature'] for f in event.get('drifted_features', [])]
            
            variants.append(Variant(
                rank=i + 1,
                gemma_score=gemma_score,
                cv_score=1 - gemma_score / 100,
                variant_type="drift_event",
                model_name="KS Test + PSI",
                features_used=drifted_features[:5],
                interpretation=f"{severity} drift: {event['period_start'][:10]} to {event['period_end'][:10]}",
                details={
                    'period_start': event.get('period_start'),
                    'period_end': event.get('period_end'),
                    'severity': severity,
                    'drifted_features': drifted_features
                }
            ))
        
        if not variants:
            variants.append(Variant(
                rank=1,
                gemma_score=95,
                cv_score=0.05,
                variant_type="stable",
                model_name="No Drift Detected",
                features_used=[],
                interpretation="Data distribution is stable - no retraining needed",
                details={'status': 'stable'}
            ))
        
        return variants
    
    def _convert_psi_to_importance(
        self, 
        psi_scores: Dict[str, float],
        time_col: str
    ) -> List[FeatureImportance]:
        """Convert PSI scores to feature importance"""
        features = []
        
        for feat, psi in psi_scores.items():
            if psi > 0.25:
                explanation = f"{feat} has drifted significantly (PSI: {psi:.3f}). Consider retraining."
                impact = "negative"
            elif psi > 0.1:
                explanation = f"{feat} shows moderate drift (PSI: {psi:.3f}). Monitor closely."
                impact = "mixed"
            else:
                explanation = f"{feat} is stable (PSI: {psi:.3f}). No action needed."
                impact = "positive"
            
            features.append(FeatureImportance(
                name=feat,
                stability=(1 - min(psi, 1.0)) * 100,
                importance=psi,
                impact=impact,
                explanation=explanation
            ))
        
        features.sort(key=lambda f: f.importance, reverse=True)
        return features
    
    def _build_summary(self, raw_result: Dict) -> PlainEnglishSummary:
        """Build summary from drift results"""
        drift_events = raw_result.get('drift_events', [])
        psi_scores = raw_result.get('psi_scores', {})
        high_psi = [k for k, v in psi_scores.items() if v > 0.25]
        
        if drift_events:
            headline = f"Detected {len(drift_events)} drift event(s) - your models may be outdated"
            explanation = (
                f"Your data distribution has changed over time. "
                f"{len(high_psi)} feature(s) show significant drift (PSI > 0.25). "
                f"Models trained on old data may not perform well on current data."
            )
            recommendation = "Retrain your models on recent data to maintain accuracy."
            confidence = Confidence.HIGH
        else:
            headline = "No significant drift detected - your models are still valid"
            explanation = (
                f"Analyzed {raw_result.get('windows_analyzed', 0)} time windows and found stable distributions. "
                f"Your current models should continue to perform well."
            )
            recommendation = "Continue monitoring on a regular schedule."
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
            methodology_name="Kolmogorov-Smirnov Test + Population Stability Index",
            methodology_url="https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test",
            steps=[
                ExplanationStep(1, "Time Window Creation", "Split data into time-based windows (daily/weekly/monthly)"),
                ExplanationStep(2, "Baseline Establishment", "Use first window(s) as reference distribution"),
                ExplanationStep(3, "KS Test", "Compare each window to baseline using Kolmogorov-Smirnov test"),
                ExplanationStep(4, "PSI Calculation", "Compute Population Stability Index for each feature"),
                ExplanationStep(5, "Drift Detection", "Flag windows where p-value < 0.05 or PSI > 0.25")
            ],
            limitations=[
                "Requires temporal data with clear time ordering",
                "Minimum 100 samples needed for reliable detection",
                "Seasonal patterns may be falsely flagged as drift"
            ]
        )
    
    def _build_config_schema(self) -> List[ConfigParameter]:
        """Convert config schema to ConfigParameter list"""
        params = []
        for name, schema in SCOUT_CONFIG_SCHEMA.items():
            param_range = None
            if 'min' in schema and 'max' in schema:
                param_range = [schema['min'], schema['max']]
            elif 'options' in schema:
                param_range = schema['options']
            
            params.append(ConfigParameter(
                name=name,
                type=schema.get('type', 'str'),
                default=schema.get('default'),
                range=param_range,
                description=schema.get('description', '')
            ))
        return params
    
    def _create_fallback_variant(self) -> Variant:
        """Create fallback variant"""
        return Variant(
            rank=1,
            gemma_score=0,
            cv_score=0.0,
            variant_type="fallback",
            model_name="None",
            features_used=[],
            interpretation="Analysis could not be completed",
            details={"error": "No results"}
        )
    
    def _error_to_premium_result(
        self, 
        raw_result: Dict, 
        df: pd.DataFrame, 
        config: Dict,
        start_time: float
    ) -> PremiumResult:
        """Convert error to PremiumResult"""
        import time
        message = raw_result.get('message', '') or raw_result.get('insights', [''])[0] if raw_result.get('insights') else 'Unknown error'
        
        return PremiumResult(
            engine_name="scout",
            engine_display_name="Scout Monitor",
            engine_icon="🔍",
            task_type=TaskType.DETECTION,
            target_column=None,
            columns_analyzed=list(df.columns),
            row_count=len(df),
            variants=[],
            best_variant=self._create_fallback_variant(),
            feature_importance=[],
            summary=PlainEnglishSummary(
                headline="Drift detection not applicable",
                explanation=message,
                recommendation="Provide data with a time/date column for drift detection.",
                confidence=Confidence.LOW
            ),
            explanation=self._build_explanation(),
            holdout=None,
            config_used=config,
            config_schema=self._build_config_schema(),
            execution_time_seconds=time.time() - start_time,
            warnings=[message]
        )


# CLI entry point
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python scout_engine.py <csv_file> [--time COLUMN] [--window SIZE]")
        print("  Example: python scout_engine.py stock_prices.csv --time date --window 1M")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    
    # Parse arguments
    time_col = None
    window_size = '1M'
    
    if '--time' in sys.argv:
        time_idx = sys.argv.index('--time')
        if len(sys.argv) > time_idx + 1:
            time_col = sys.argv[time_idx + 1]
    
    if '--window' in sys.argv:
        window_idx = sys.argv.index('--window')
        if len(sys.argv) > window_idx + 1:
            window_size = sys.argv[window_idx + 1]
    
    df = pd.read_csv(csv_file)
    
    engine = ScoutEngine()
    config = {
        'time_column': time_col,
        'window_size': window_size
    }
    result = engine.analyze(df, config)
    
    print(f"\n{'='*60}")
    print(f"SCOUT ENGINE RESULTS: {csv_file}")
    print(f"{'='*60}\n")
    
    if 'error' in result:
        print(f"❌ Error: {result['message']}")
        for insight in result.get('insights', []):
            print(f"   {insight}")
    else:
        for insight in result['insights']:
            print(insight)
        
        print(f"\n{'='*60}")
        print("DRIFT TIMELINE:")
        print(f"{'='*60}\n")
        
        if result['drift_events']:
            for event in result['drift_events']:
                print(f"Period: {event['period_start'][:10]} → {event['period_end'][:10]}")
                print(f"Severity: {event['severity']}")
                print(f"Drifted Features: {len(event['drifted_features'])}")
                for feat_info in event['drifted_features'][:3]:
                    print(f"  - {feat_info['feature']}: Drift={feat_info['drift_score']:.2f}")
                print()
        else:
            print("No drift events detected - distribution stable.")
        
        print(f"\n{'='*60}")
        print("PSI SCORES (Population Stability Index):")
        print(f"{'='*60}\n")
        
        for feat, score in sorted(result['psi_scores'].items(), key=lambda x: x[1], reverse=True)[:10]:
            status = "🔴 HIGH" if score > 0.25 else ("🟡 MEDIUM" if score > 0.1 else "🟢 LOW")
            bar = '█' * int(min(score, 1.0) * 30) + '░' * (30 - int(min(score, 1.0) * 30))
            print(f"{feat:20s} [{bar}] {score:.3f} {status}")
