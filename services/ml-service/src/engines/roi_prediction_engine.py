"""
ROI Prediction Engine - Schema-Agnostic Version

Predicts return on investment for initiatives with confidence intervals.
Supports NPV analysis, payback period calculation, and sensitivity analysis.

NOW SCHEMA-AGNOSTIC: Works with any dataset structure via semantic column mapping.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from scipy import stats

# Import Schema Intelligence Layer
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core import (
    ColumnProfiler,
    ColumnMapper,
    SemanticType,
    BusinessEntity,
    EngineRequirements,
    DatasetClassifier
)

from core.premium_models import ConfigParameter

logger = logging.getLogger(__name__)


class ROIPredictionEngine:
    """
    Schema-agnostic ROI prediction and analysis engine.
    
    Automatically detects investment, return, and time columns using semantic intelligence.
    """
    
    def __init__(self):
        self.name = "ROI Prediction Engine (Schema-Agnostic)"
        self.profiler = ColumnProfiler()
        self.mapper = ColumnMapper()
    
    def get_engine_info(self) -> Dict[str, str]:
        """Get engine display information."""
        return {
            "name": "roi_prediction",
            "display_name": "ROI Prediction",
            "icon": "📊",
            "task_type": "prediction"
        }
    
    def get_config_schema(self) -> List[ConfigParameter]:
        """Get configuration schema for UI."""
        return [
            ConfigParameter(
                name="investment_column",
                type="select",
                default=None,
                range=[],
                description="Column containing investment values"
            ),
            ConfigParameter(
                name="return_column",
                type="select",
                default=None,
                range=[],
                description="Column containing return values"
            ),
            ConfigParameter(
                name="discount_rate",
                type="float",
                default=0.1,
                range=[0.0, 0.5],
                description="Discount rate for NPV calculations"
            ),
            ConfigParameter(
                name="confidence_level",
                type="float",
                default=0.95,
                range=[0.8, 0.99],
                description="Confidence level for predictions"
            )
        ]
    
    def get_methodology_info(self) -> Dict[str, Any]:
        """Get methodology documentation."""
        return {
            "name": "ROI Prediction Analysis",
            "url": "https://en.wikipedia.org/wiki/Return_on_investment",
            "steps": [
                {"step_number": 1, "title": "Column Detection", "description": "Identify investment and return columns"},
                {"step_number": 2, "title": "ROI Calculation", "description": "Calculate basic and advanced ROI metrics"},
                {"step_number": 3, "title": "NPV Analysis", "description": "Compute Net Present Value with discount rate"},
                {"step_number": 4, "title": "Sensitivity Analysis", "description": "Assess ROI sensitivity to key variables"}
            ],
            "limitations": ["Historical ROI may not predict future", "Requires accurate cost and revenue data"],
            "assumptions": ["Investment and return data is complete", "Discount rate reflects opportunity cost"]
        }
    
    def get_requirements(self) -> EngineRequirements:
        """
        Define semantic requirements for ROI analysis.
        
        Returns:
            EngineRequirements specifying needed columns
        """
        return EngineRequirements(
            required_semantics=[SemanticType.NUMERIC_CONTINUOUS],
            optional_semantics={
                'investment': [SemanticType.NUMERIC_CONTINUOUS],
                'returns': [SemanticType.NUMERIC_CONTINUOUS],
                'temporal': [SemanticType.TEMPORAL]
            },
            required_entities=[],
            preferred_domains=[],
            applicable_tasks=[],
            min_rows=3,
            min_numeric_cols=2
        )
        
    def analyze(self, df: pd.DataFrame, config: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Run schema-agnostic ROI prediction analysis.
        
        Args:
            df: DataFrame with any structure containing investment/return data
            config: Optional configuration with:
                - investment_column: Hint for investment column
                - return_column: Hint for return column
                - time_column: Hint for time/period column
                - discount_rate: Discount rate for NPV (default: 0.1)
                - confidence_level: Confidence level (default: 0.95)
                - skip_profiling: Skip schema intelligence (default: False)
        
        Returns:
            Dict with ROI predictions, NPV analysis, and visualizations
        """
        config = config or {}
        discount_rate = config.get('discount_rate', 0.1)
        confidence = config.get('confidence_level', 0.95)
        
        results = {
            'summary': {},
            'predictions': [],
            'graphs': [],
            'insights': [],
            'column_mappings': {},
            'profiling_used': not config.get('skip_profiling', False)
        }
        
        try:
            # SCHEMA INTELLIGENCE: Profile and map columns
            if not config.get('skip_profiling', False):
                logger.info("Profiling dataset for ROI analysis")
                profiles = self.profiler.profile_dataset(df)
                
                # Classify dataset
                classifier = DatasetClassifier()
                classification = classifier.classify(profiles, len(df))
                results['summary']['detected_domain'] = classification.domain.value
                
                # SMART COLUMN DETECTION (ROI-specific logic)
                investment_col = self._smart_detect(
                    df, profiles,
                    keywords=['investment', 'cost', 'spend', 'expense', 'capex'],
                    hint=config.get('investment_column')
                )
                
                return_col = self._smart_detect(
                    df, profiles,
                    keywords=['return', 'revenue', 'benefit', 'profit', 'income', 'gain'],
                    hint=config.get('return_column')
                )
                
                time_col = self._smart_detect(
                    df, profiles,
                    keywords=['time', 'period', 'month', 'year', 'date', 'quarter'],
                    hint=config.get('time_column'),
                    semantic_type=SemanticType.TEMPORAL
                )
                
                if investment_col:
                    results['column_mappings']['investment'] = {
                        'column': investment_col,
                        'confidence': 0.9
                    }
                if return_col:
                    results['column_mappings']['returns'] = {
                        'column': return_col,
                        'confidence': 0.9
                    }
                if time_col:
                    results['column_mappings']['temporal'] = {
                        'column': time_col,
                        'confidence': 0.95
                    }
            else:
                # Fallback: use hints or old detection
                investment_col = config.get('investment_column') or self._detect_column(
                    df, ['investment', 'cost', 'spend'], None
                )
                return_col = config.get('return_column') or self._detect_column(
                    df, ['return', 'revenue', 'benefit', 'profit'], None
                )
                time_col = config.get('time_column') or self._detect_column(
                    df, ['time', 'period', 'month', 'year', 'date'], None
                )
                results['profiling_used'] = False
                
            if not (investment_col and return_col):
                missing = []
                if not investment_col:
                    missing.append('investment/cost column')
                if not return_col:
                    missing.append('return/revenue column')
                
                return {
                    'engine': 'ROI Prediction Engine (Schema-Agnostic)',
                    'status': 'not_applicable',
                    'insights': [f'📊 **ROI Prediction - Not Applicable**: This engine requires financial data columns (investment/cost and return/revenue). Missing: {", ".join(missing)}. This dataset does not contain the necessary financial metrics for ROI analysis.'],
                    'column_mappings': results['column_mappings'],
                    'profiling_used': results['profiling_used']
                }
            
            # Store detected columns in summary
            results['summary']['investment_column'] = investment_col
            results['summary']['return_column'] = return_col
            if time_col:
                results['summary']['time_column'] = time_col
            
            # ANALYSIS PIPELINE (unchanged logic, now schema-agnostic)
            # Basic ROI calculation
            basic_roi = self._calculate_basic_roi(df, investment_col, return_col)
            results['basic_roi'] = basic_roi
            results['graphs'].append(basic_roi['graph'])
            
            # Payback period
            if time_col:
                payback = self._calculate_payback_period(df, investment_col, return_col, time_col)
                results['payback'] = payback
                results['graphs'].append(payback['graph'])
            
            # NPV analysis
            if time_col:
                npv_analysis = self._npv_analysis(df, investment_col, return_col, time_col, discount_rate)
                results['npv'] = npv_analysis
                results['graphs'].append(npv_analysis['graph'])
            
            # ROI prediction with confidence intervals
            predictions = self._predict_future_roi(df, investment_col, return_col, confidence)
            if 'error' not in predictions:
                results['predictions'] = predictions
                results['graphs'].append(predictions['graph'])
            
            # Sensitivity analysis
            sensitivity = self._sensitivity_analysis(df, investment_col, return_col, discount_rate)
            results['sensitivity'] = sensitivity
            results['graphs'].append(sensitivity['graph'])
            
            # Risk assessment
            risk = self._assess_risk(df, investment_col, return_col)
            results['risk'] = risk
            
            # Generate insights
            results['insights'] = self._generate_insights(
                results, investment_col, return_col, time_col
            )
            
        except Exception as e:
            logger.error(f"ROI prediction failed: {e}", exc_info=True)
            results['error'] = str(e)
            
        return results
    
    def _smart_detect(self, df: pd.DataFrame, profiles: Dict,
                     keywords: List[str], hint: Optional[str] = None,
                     semantic_type: Optional[SemanticType] = None) -> Optional[str]:
        """
        Smart column detection using profiles + keywords.
        
        Args:
            df: DataFrame
            profiles: Column profiles from ColumnProfiler
            keywords: Keywords to match in column names
            hint: User-provided hint
            semantic_type: Required semantic type (optional)
            
        Returns:
            Detected column name or None
        """
        if hint and hint in df.columns:
            return hint
        
        # Priority 1: Keyword match with correct semantic type
        for col in df.columns:
            if any(kw in col.lower() for kw in keywords):
                prof = profiles.get(col)
                if prof:
                    if semantic_type:
                        if prof.semantic_type == semantic_type:
                            return col
                    elif prof.semantic_type in [SemanticType.NUMERIC_CONTINUOUS, SemanticType.NUMERIC_DISCRETE]:
                        return col
        
        # Priority 2: Semantic type match (no keyword)
        if semantic_type:
            for col, prof in profiles.items():
                if prof.semantic_type == semantic_type:
                    return col
        
        return None
    
    def _detect_column(self, df: pd.DataFrame, keywords: List[str], hint: Optional[str] = None) -> Optional[str]:
        """Detect column by keywords"""
        if hint and hint in df.columns:
            return hint
            
        for col in df.columns:
            if any(kw in col.lower() for kw in keywords):
                if pd.api.types.is_numeric_dtype(df[col]):
                    return col
        return None
    
    def _calculate_basic_roi(self, df: pd.DataFrame, investment_col: str, return_col: str) -> Dict:
        """Calculate basic ROI metrics"""
        df['roi'] = ((df[return_col] - df[investment_col]) / df[investment_col] * 100).replace([np.inf, -np.inf], np.nan)
        
        avg_roi = df['roi'].mean()
        median_roi = df['roi'].median()
        
        graph = {
            'type': 'bar_chart',
            'title': 'ROI Distribution',
            'x_data': df.index.tolist() if len(df) < 50 else list(range(len(df))),
            'y_data': df['roi'].fillna(0).values.tolist(),
            'x_label': 'Initiative',
            'y_label': 'ROI (%)',
            'threshold': 0  # Zero line for breakeven
        }
        
        return {
            'average_roi': float(avg_roi),
            'median_roi': float(median_roi),
            'best_roi': float(df['roi'].max()),
            'worst_roi': float(df['roi'].min()),
            'positive_count': int((df['roi'] > 0).sum()),
            'negative_count': int((df['roi'] < 0).sum()),
            'graph': graph
        }
    
    def _calculate_payback_period(self, df: pd.DataFrame, investment_col: str, return_col: str, time_col: str) -> Dict:
        """Calculate payback period"""
        df_sorted = df.sort_values(time_col)
        
        # Calculate cumulative returns
        df_sorted['cumulative_return'] = df_sorted[return_col].cumsum()
        df_sorted['cumulative_investment'] = df_sorted[investment_col].cumsum()
        
        # Find breakeven point
        breakeven_idx = df_sorted[df_sorted['cumulative_return'] >= df_sorted['cumulative_investment']].index
        payback_period = len(df_sorted) if len(breakeven_idx) == 0 else df_sorted.index.get_loc(breakeven_idx[0]) + 1
        
        graph = {
            'type': 'line_chart',
            'title': 'Payback Period Analysis',
            'x_data': df_sorted[time_col].astype(str).tolist(),
            'y_data': [
                df_sorted['cumulative_investment'].values.tolist(),
                df_sorted['cumulative_return'].values.tolist()
            ],
            'labels': ['Cumulative Investment', 'Cumulative Returns'],
            'breakeven': payback_period
        }
        
        return {
            'payback_period': payback_period,
            'payback_achieved': len(breakeven_idx) > 0,
            'graph': graph
        }
    
    def _npv_analysis(self, df: pd.DataFrame, investment_col: str, return_col: str, time_col: str, discount_rate: float) -> Dict:
        """Calculate Net Present Value"""
        df_sorted = df.sort_values(time_col)
        
        # Calculate NPV
        periods = range(len(df_sorted))
        cash_flows = (df_sorted[return_col] - df_sorted[investment_col]).values
        
        npv = sum(cf / (1 + discount_rate) ** t for t, cf in enumerate(cash_flows))
        
        # Calculate IRR (Internal Rate of Return) approximation
        # Use Newton's method for simple IRR calculation
        irr = self._calculate_irr(cash_flows)
        
        graph = {
            'type': 'waterfall',
            'title': 'NPV Analysis (Discounted Cash Flows)',
            'x_data': df_sorted[time_col].astype(str).tolist(),
            'y_data': [cf / (1 + discount_rate) ** t for t, cf in enumerate(cash_flows)],
            'npv': float(npv)
        }
        
        return {
            'npv': float(npv),
            'irr': float(irr) if irr is not None else None,
            'discount_rate': discount_rate,
            'profitable': npv > 0,
            'graph': graph
        }
    
    def _calculate_irr(self, cash_flows: np.ndarray, max_iterations: int = 100) -> Optional[float]:
        """Calculate Internal Rate of Return using Newton's method"""
        try:
            # Initial guess
            irr = 0.1
            
            for _ in range(max_iterations):
                npv = sum(cf / (1 + irr) ** t for t, cf in enumerate(cash_flows))
                npv_derivative = sum(-t * cf / (1 + irr) ** (t + 1) for t, cf in enumerate(cash_flows))
                
                if abs(npv_derivative) < 1e-10:
                    break
                    
                irr_new = irr - npv / npv_derivative
                
                if abs(irr_new - irr) < 1e-6:
                    return irr_new
                    
                irr = irr_new
            
            return irr
        except:
            return None
    
    def _predict_future_roi(self, df: pd.DataFrame, investment_col: str, return_col: str, confidence: float) -> Dict:
        """Predict future ROI with confidence intervals"""
        df['roi'] = ((df[return_col] - df[investment_col]) / df[investment_col] * 100).replace([np.inf, -np.inf], np.nan)
        
        roi_data = df['roi'].dropna()
        
        if len(roi_data) < 3:
            return {'error': 'Insufficient data for predictions'}
        
        # Calculate statistics
        mean_roi = roi_data.mean()
        std_roi = roi_data.std()
        
        # Confidence intervals
        z_score = stats.norm.ppf((1 + confidence) / 2)
        margin_of_error = z_score * (std_roi / np.sqrt(len(roi_data)))
        
        ci_lower = mean_roi - margin_of_error
        ci_upper = mean_roi + margin_of_error
        
        # Future predictions (simple projection)
        future_periods = 12
        predictions = [mean_roi] * future_periods
        lower_bounds = [ci_lower] * future_periods
        upper_bounds = [ci_upper] * future_periods
        
        graph = {
            'type': 'forecast',
            'title': f'ROI Forecast ({int(confidence*100)}% Confidence Interval)',
            'x_data': [f'Period {i+1}' for i in range(future_periods)],
            'y_data': predictions,
            'lower_bound': lower_bounds,
            'upper_bound': upper_bounds,
            'confidence': confidence
        }
        
        return {
            'predicted_mean_roi': float(mean_roi),
            'confidence_interval_lower': float(ci_lower),
            'confidence_interval_upper': float(ci_upper),
            'confidence_level': confidence,
            'volatility': float(std_roi),
            'graph': graph
        }
    
    def _sensitivity_analysis(self, df: pd.DataFrame, investment_col: str, return_col: str, base_discount_rate: float) -> Dict:
        """Perform sensitivity analysis on ROI"""
        discount_rates = np.linspace(0, 0.3, 7)
        npvs = []
        
        for rate in discount_rates:
            cash_flows = (df[return_col] - df[investment_col]).values
            npv = sum(cf / (1 + rate) ** t for t, cf in enumerate(cash_flows))
            npvs.append(npv)
        
        graph = {
            'type': 'line_chart',
            'title': 'Sensitivity Analysis: NPV vs Discount Rate',
            'x_data': (discount_rates * 100).tolist(),
            'y_data': npvs,
            'x_label': 'Discount Rate (%)',
            'y_label': 'NPV ($)',
            'baseline': base_discount_rate * 100
        }
        
        return {
            'discount_rates': discount_rates.tolist(),
            'npv_values': npvs,
            'graph': graph
        }
    
    def _assess_risk(self, df: pd.DataFrame, investment_col: str, return_col: str) -> Dict:
        """Assess investment risk"""
        df['roi'] = ((df[return_col] - df[investment_col]) / df[investment_col] * 100).replace([np.inf, -np.inf], np.nan)
        
        roi_data = df['roi'].dropna()
        
        if len(roi_data) == 0:
            return {'risk_level': 'Unknown'}
        
        # Calculate risk metrics
        volatility = roi_data.std()
        downside_deviation = roi_data[roi_data < roi_data.mean()].std()
        sharpe_ratio = roi_data.mean() / volatility if volatility > 0 else 0
        
        # Risk classification
        if volatility < 10:
            risk_level = 'Low'
        elif volatility < 25:
            risk_level = 'Medium'
        else:
            risk_level = 'High'
        
        return {
            'risk_level': risk_level,
            'volatility': float(volatility),
            'downside_deviation': float(downside_deviation),
            'sharpe_ratio': float(sharpe_ratio),
            'success_rate': float((roi_data > 0).sum() / len(roi_data) * 100)
        }
    
    def _generate_insights(self, results: Dict, investment_col: str = None, return_col: str = None, time_col: str = None) -> List[str]:
        """Generate insights"""
        insights = []
        
        if 'basic_roi' in results:
            roi = results['basic_roi']
            insights.append(
                f"📊 Average ROI: {roi['average_roi']:.1f}% | "
               f"{roi['positive_count']} profitable, {roi['negative_count']} loss-making"
            )
        
        if 'npv' in results:
            npv = results['npv']
            if npv['profitable']:
                insights.append(f"✅ Positive NPV of ${npv['npv']:,.2f} - investment recommended")
            else:
                insights.append(f"⚠️ Negative NPV of ${npv['npv']:,.2f} - reconsider investment")
        
        if 'payback' in results:
            pb = results['payback']
            if pb['payback_achieved']:
                insights.append(f"💰 Payback achieved in {pb['payback_period']} periods")
            else:
                insights.append(f"⏳ Payback not yet achieved after {pb['payback_period']} periods")
        
        if 'risk' in results:
            risk = results['risk']
            insights.append(
                f"🎯 Risk Level: {risk['risk_level']} | "
                f"Success Rate: {risk['success_rate']:.1f}%"
            )
        
        return insights


__all__ = ['ROIPredictionEngine']
