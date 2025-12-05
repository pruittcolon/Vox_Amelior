"""
Chronos Time Series Forecasting Engine - Advanced Predictions

Generates business forecasts using state-of-the-art time series models.

Core technique: Prophet (Facebook/Meta)
- Automatically detects seasonality and trends
- Handles holidays and special events
- Provides uncertainty intervals
- Robust to missing data and outliers

Author: Enterprise Analytics Team
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import warnings
warnings.filterwarnings('ignore')

from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

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

CHRONOS_CONFIG_SCHEMA = {
    "forecast_periods": {"type": "int", "min": 1, "max": 365, "default": 30, "description": "Number of periods to forecast"},
    "seasonality_mode": {"type": "select", "options": ["additive", "multiplicative"], "default": "additive", "description": "Type of seasonality"},
    "include_history": {"type": "bool", "default": True, "description": "Include historical data in output"},
    "run_all_variants": {"type": "bool", "default": False, "description": "Run all Prophet variants for comparison"},
}

# Prophet variant configurations for comprehensive comparison
PROPHET_VARIANTS = [
    {
        "name": "Prophet Linear + Additive",
        "short_name": "linear_add",
        "growth": "linear",
        "seasonality_mode": "additive",
        "changepoint_prior_scale": 0.05,
        "description": "Standard linear growth with additive seasonality"
    },
    {
        "name": "Prophet Linear + Multiplicative", 
        "short_name": "linear_mult",
        "growth": "linear",
        "seasonality_mode": "multiplicative",
        "changepoint_prior_scale": 0.05,
        "description": "Linear growth with multiplicative seasonality (better for growing trends)"
    },
    {
        "name": "Prophet Flat + Additive",
        "short_name": "flat_add", 
        "growth": "flat",
        "seasonality_mode": "additive",
        "changepoint_prior_scale": 0.05,
        "description": "No trend, pure seasonality patterns"
    },
    {
        "name": "Prophet Flexible Trend",
        "short_name": "flexible",
        "growth": "linear",
        "seasonality_mode": "additive",
        "changepoint_prior_scale": 0.5,
        "description": "Highly flexible changepoint detection for volatile data"
    },
    {
        "name": "Prophet Rigid Trend",
        "short_name": "rigid",
        "growth": "linear", 
        "seasonality_mode": "additive",
        "changepoint_prior_scale": 0.001,
        "description": "Rigid trend line for stable data with minimal changepoints"
    },
    {
        "name": "Prophet Multiplicative Flexible",
        "short_name": "mult_flex",
        "growth": "linear",
        "seasonality_mode": "multiplicative",
        "changepoint_prior_scale": 0.3,
        "description": "Flexible multiplicative model for complex seasonal patterns"
    }
]


class ChronosEngine:
    """
    Chronos Time Series Forecasting Engine: Future Predictions
    
    Uses Prophet (Facebook's forecasting library) to generate
    accurate, interpretable forecasts with uncertainty bounds.
    
    Key insight: Predict the future with statistical confidence
    """
    
    def __init__(self):
        self.name = "Chronos Time Series Forecasting Engine"
        self.profiler = ColumnProfiler()
        self.mapper = ColumnMapper()
        
    def get_requirements(self) -> EngineRequirements:
        """Define what this engine needs to run"""
        return EngineRequirements(
            required_semantics=[SemanticType.TEMPORAL, SemanticType.NUMERIC_CONTINUOUS],
            optional_semantics={},
            required_entities=[],
            preferred_domains=[],
            applicable_tasks=['forecasting', 'time_series'],
            min_rows=100
        )
    
    def analyze(self, df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Main analysis method
        
        Args:
            df: Input dataset with time series data
            config: {
                'time_column': Optional[str] - name of time column
                'target_column': Optional[str] - name of column to forecast
                'forecast_periods': int - number of periods to forecast (default: 30)
                'freq': str - frequency ('D', 'W', 'M', 'Y') (default: 'D')
                'confidence_interval': float - confidence level (default: 0.95)
            }
        
        Returns:
            {
                'forecast': pd.DataFrame - future predictions with confidence intervals,
                'metrics': {...} - validation metrics,
                'insights': [...]
            }
        """
        config = config or {}
        
        # 1. Profile dataset
        if not config.get('skip_profiling', False):
            profiles = self.profiler.profile_dataset(df)
        else:
            profiles = {}
        
        # 2. Detect time and target columns
        time_col = self._detect_time_column(df, profiles, config.get('time_column'))
        target_col = self._detect_target_column(df, profiles, config.get('target_column'), exclude=[time_col])
        
        if not time_col or not target_col:
            return {
                'error': 'Missing required columns',
                'message': f'Need time column and target column. Found: time={time_col}, target={target_col}',
                'insights': ['💡 Specify time_column and target_column in config']
            }
        
        # 3. Prepare data for Prophet
        prophet_df = self._prepare_prophet_data(df, time_col, target_col)
        
        if len(prophet_df) < 10:
            return {
                'error': 'Insufficient data',
                'message': f'Only {len(prophet_df)} valid time points. Need at least 10.',
                'insights': []
            }
        
        # 4. Configure forecast
        forecast_periods = config.get('forecast_periods', 30)
        freq = config.get('freq', 'D')
        confidence_interval = config.get('confidence_interval', 0.95)
        run_all_variants = config.get('run_all_variants', False)
        
        # 5. Run ALL variants if requested for comprehensive comparison
        if run_all_variants:
            return self._run_all_variants(
                prophet_df, time_col, target_col, forecast_periods, 
                freq, confidence_interval
            )
        
        # 6. Train model and generate forecast (single default variant)
        model, forecast, metrics = self._generate_forecast(
            prophet_df, forecast_periods, freq, confidence_interval
        )
        
        # 6. Generate insights
        insights = self._generate_insights(
            time_col, target_col, prophet_df, forecast, metrics, forecast_periods
        )
        
        return {
            'engine': self.name,
            'time_column': time_col,
            'target_column': target_col,
            'historical_periods': len(prophet_df),
            'forecast_periods': forecast_periods,
            'frequency': freq,
            'confidence_interval': confidence_interval,
            'forecast': forecast,
            'metrics': metrics,
            'insights': insights
        }
    
    def _detect_time_column(self, df: pd.DataFrame, profiles: Dict, hint: Optional[str]) -> Optional[str]:
        """Detect temporal column"""
        if hint and hint in df.columns:
            return hint
        
        if profiles:
            for col, profile in profiles.items():
                if profile.semantic_type == SemanticType.TEMPORAL:
                    return col
        
        time_keywords = ['date', 'time', 'timestamp', 'datetime', 'year', 'month', 'day', 'rownames']
        for col in df.columns:
            if any(kw in col.lower() for kw in time_keywords):
                return col
        
        return None
    
    def _detect_target_column(self, df: pd.DataFrame, profiles: Dict, hint: Optional[str], exclude: List[str]) -> Optional[str]:
        """Detect target column to forecast"""
        if hint and hint in df.columns:
            return hint
        
        # Common target keywords
        target_keywords = ['value', 'sales', 'revenue', 'price', 'count', 'volume', 'close']
        for col in df.columns:
            if col in exclude:
                continue
            if any(kw in col.lower() for kw in target_keywords):
                if pd.api.types.is_numeric_dtype(df[col]):
                    return col
        
        # First numeric column (excluding time)
        for col in df.columns:
            if col in exclude:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                return col
        
        return None
    
    def _run_all_variants(
        self, prophet_df: pd.DataFrame, time_col: str, target_col: str,
        forecast_periods: int, freq: str, confidence_interval: float
    ) -> Dict[str, Any]:
        """
        Run all Prophet variant configurations and return comprehensive comparison.
        Returns high-precision metrics for each variant.
        """
        import time as time_module
        
        all_variants = []
        best_variant = None
        best_mape = float('inf')
        
        for variant_config in PROPHET_VARIANTS:
            start_time = time_module.time()
            try:
                model, forecast, metrics = self._generate_forecast(
                    prophet_df, forecast_periods, freq, confidence_interval,
                    variant_config=variant_config
                )
                execution_time = time_module.time() - start_time
                
                variant_result = {
                    'name': variant_config['name'],
                    'short_name': variant_config['short_name'],
                    'description': variant_config['description'],
                    'config': {
                        'growth': variant_config['growth'],
                        'seasonality_mode': variant_config['seasonality_mode'],
                        'changepoint_prior_scale': variant_config['changepoint_prior_scale']
                    },
                    'metrics': {
                        'mae': float(metrics['mae']),
                        'rmse': float(metrics['rmse']),
                        'mape': float(metrics['mape']),
                        'accuracy': max(0.0, 100.0 - float(metrics['mape'])),
                        'train_size': metrics['train_size'],
                        'test_size': metrics['test_size']
                    },
                    'execution_time_seconds': execution_time,
                    'forecast': forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_dict('records') if hasattr(forecast, 'to_dict') else [],
                    'success': True
                }
                
                # Convert timestamps to strings in forecast
                for f in variant_result['forecast']:
                    if hasattr(f['ds'], 'isoformat'):
                        f['ds'] = f['ds'].isoformat()
                
                all_variants.append(variant_result)
                
                # Track best variant by lowest MAPE
                if metrics['mape'] < best_mape:
                    best_mape = metrics['mape']
                    best_variant = variant_result
                    
            except Exception as e:
                variant_result = {
                    'name': variant_config['name'],
                    'short_name': variant_config['short_name'],
                    'description': variant_config['description'],
                    'config': variant_config,
                    'metrics': None,
                    'execution_time_seconds': time_module.time() - start_time,
                    'forecast': [],
                    'success': False,
                    'error': str(e)
                }
                all_variants.append(variant_result)
        
        # Sort by accuracy (descending) = lowest MAPE first
        successful_variants = [v for v in all_variants if v['success']]
        successful_variants.sort(key=lambda x: x['metrics']['mape'])
        
        # Assign ranks
        for i, v in enumerate(successful_variants):
            v['rank'] = i + 1
        
        return {
            'engine': self.name,
            'mode': 'all_variants',
            'time_column': time_col,
            'target_column': target_col,
            'historical_periods': len(prophet_df),
            'forecast_periods': forecast_periods,
            'frequency': freq,
            'confidence_interval': confidence_interval,
            'variants': all_variants,
            'best_variant': best_variant,
            'variant_count': len(all_variants),
            'successful_count': len(successful_variants),
            'insights': [
                f"📊 **Multi-Variant Analysis**: Ran {len(all_variants)} Prophet configurations",
                f"🏆 **Best Model**: {best_variant['name']} with {best_variant['metrics']['accuracy']:.10f}% accuracy" if best_variant else "No successful variants",
                f"⏱️ **Total Execution**: {sum(v['execution_time_seconds'] for v in all_variants):.3f}s"
            ]
        }
    
    def _prepare_prophet_data(self, df: pd.DataFrame, time_col: str, target_col: str) -> pd.DataFrame:
        """Prepare data in Prophet format (ds, y)"""
        prophet_df = df[[time_col, target_col]].copy()
        prophet_df.columns = ['ds', 'y']
        
        # Convert to datetime
        prophet_df['ds'] = pd.to_datetime(prophet_df['ds'], errors='coerce')
        
        # Convert target to numeric, coercing errors
        prophet_df['y'] = pd.to_numeric(prophet_df['y'], errors='coerce')
        
        # Remove infinity values
        prophet_df = prophet_df.replace([np.inf, -np.inf], np.nan)
        
        # Remove missing values (including NaN from infinity replacement)
        prophet_df = prophet_df.dropna()
        
        # Clip extreme values to prevent overflow (99.9th percentile)
        if len(prophet_df) > 0:
            q_low = prophet_df['y'].quantile(0.001)
            q_high = prophet_df['y'].quantile(0.999)
            prophet_df['y'] = prophet_df['y'].clip(lower=q_low, upper=q_high)
        
        # If there are duplicate dates, aggregate by taking the mean
        if prophet_df['ds'].duplicated().any():
            prophet_df = prophet_df.groupby('ds', as_index=False).agg({'y': 'mean'})
        
        # Sort by time
        prophet_df = prophet_df.sort_values('ds').reset_index(drop=True)
        
        return prophet_df
    
    def _generate_forecast(
        self, prophet_df: pd.DataFrame, forecast_periods: int,
        freq: str, confidence_interval: float,
        variant_config: Optional[Dict] = None
    ) -> tuple:
        """
        Train Prophet model and generate forecast
        
        Args:
            variant_config: Optional Prophet configuration {
                'growth': 'linear'|'flat',
                'seasonality_mode': 'additive'|'multiplicative',
                'changepoint_prior_scale': float
            }
        """
        # Split into train/test for validation
        train_size = int(len(prophet_df) * 0.8)
        train_df = prophet_df[:train_size]
        test_df = prophet_df[train_size:]
        
        # Apply variant configuration or defaults
        growth = 'linear'
        seasonality_mode = 'additive'
        changepoint_prior_scale = 0.05
        
        if variant_config:
            growth = variant_config.get('growth', growth)
            seasonality_mode = variant_config.get('seasonality_mode', seasonality_mode)
            changepoint_prior_scale = variant_config.get('changepoint_prior_scale', changepoint_prior_scale)
        
        # Create and train Prophet model with variant configuration
        model = Prophet(
            growth=growth,
            seasonality_mode=seasonality_mode,
            changepoint_prior_scale=changepoint_prior_scale,
            interval_width=confidence_interval,
            daily_seasonality='auto',
            weekly_seasonality='auto',
            yearly_seasonality='auto'
        )
        model.fit(train_df)
        
        # Generate future dates
        future = model.make_future_dataframe(
            periods=forecast_periods,
            freq=freq
        )
        
        # Predict
        forecast = model.predict(future)
        
        # Calculate validation metrics (on test set)
        if len(test_df) > 0:
            # Merge test set with predictions to ensure alignment
            test_merged = test_df.merge(
                forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], 
                on='ds', 
                how='inner'
            )
            if len(test_merged) > 0:
                mae = mean_absolute_error(test_merged['y'], test_merged['yhat'])
                rmse = np.sqrt(mean_squared_error(test_merged['y'], test_merged['yhat']))
                # MAPE with protection against division by zero
                non_zero_mask = test_merged['y'] != 0
                if non_zero_mask.any():
                    mape = np.mean(np.abs((test_merged.loc[non_zero_mask, 'y'].values - test_merged.loc[non_zero_mask, 'yhat'].values) / test_merged.loc[non_zero_mask, 'y'].values)) * 100
                else:
                    mape = 0
            else:
                mae, rmse, mape = 0, 0, 0
        else:
            mae, rmse, mape = 0, 0, 0
        
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'train_size': len(train_df),
            'test_size': len(test_df)
        }
        
        return model, forecast, metrics
    
    def _generate_insights(
        self, time_col: str, target_col: str,
        historical: pd.DataFrame, forecast: pd.DataFrame,
        metrics: Dict, forecast_periods: int
    ) -> List[str]:
        """Generate business-friendly insights"""
        insights = []
        
        # Summary
        insights.append(
            f"📊 **Chronos Forecasting Complete**: Generated {forecast_periods}-period forecast "
            f"for '{target_col}' using Prophet time series model."
        )
        
        # Historical data
        last_actual = historical['y'].iloc[-1]
        last_date = historical['ds'].iloc[-1]
        
        insights.append(
            f"📈 **Historical Data**: {len(historical)} observations from "
            f"{historical['ds'].min().strftime('%Y-%m-%d')} to "
            f"{historical['ds'].max().strftime('%Y-%m-%d')}. "
            f"Last actual value: {last_actual:.2f}"
        )
        
        # Validation metrics
        if metrics['test_size'] > 0:
            mape = metrics['mape']
            if mape < 10:
                accuracy = "Excellent"
                emoji = "✅"
            elif mape < 20:
                accuracy = "Good"
                emoji = "👍"
            elif mape < 30:
                accuracy = "Fair"
                emoji = "⚠️"
            else:
                accuracy = "Poor"
                emoji = "❌"
            
            insights.append(
                f"{emoji} **Forecast Accuracy** ({accuracy}): MAPE={mape:.1f}%, "
                f"MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f} "
                f"(validated on {metrics['test_size']} holdout periods)."
            )
        
        # Future predictions
        future_forecasts = forecast[forecast['ds'] > last_date].head(forecast_periods)
        
        if len(future_forecasts) > 0:
            first_forecast = future_forecasts.iloc[0]
            last_forecast = future_forecasts.iloc[-1]
            
            insights.append(
                f"🔮 **Future Forecast**:"
            )
            insights.append(
                f"   • Next period ({first_forecast['ds'].strftime('%Y-%m-%d')}): "
                f"{first_forecast['yhat']:.2f} "
                f"[{first_forecast['yhat_lower']:.2f}, {first_forecast['yhat_upper']:.2f}]"
            )
            insights.append(
                f"   • Final period ({last_forecast['ds'].strftime('%Y-%m-%d')}): "
                f"{last_forecast['yhat']:.2f} "
                f"[{last_forecast['yhat_lower']:.2f}, {last_forecast['yhat_upper']:.2f}]"
            )
            
            # Trend analysis
            avg_forecast = future_forecasts['yhat'].mean()
            pct_change = ((avg_forecast - last_actual) / last_actual) * 100
            
            if pct_change > 5:
                trend = "📈 Upward"
            elif pct_change < -5:
                trend = "📉 Downward"
            else:
                trend = "➡️ Stable"
            
            insights.append(
                f"\n💡 **Trend Analysis**: {trend} trend detected. "
                f"Average forecast: {avg_forecast:.2f} ({pct_change:+.1f}% vs last actual)."
            )
        
        # Business recommendations
        insights.append(
            "🎯 **Strategic Insight**: Use forecasts for resource planning, inventory management, "
            "and budgeting. Confidence intervals show best/worst case scenarios for risk assessment."
        )
        
        return insights

    # =========================================================================
    # PREMIUM OUTPUT
    # =========================================================================
    
    def run_premium(self, df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> PremiumResult:
        """Run Chronos forecasting and return PremiumResult."""
        import time
        start_time = time.time()
        
        config = config or {}
        
        try:
            raw = self.analyze(df, config)
        except Exception as e:
            # Handle any Prophet or analysis errors gracefully
            raw = {
                'error': 'Analysis failed',
                'message': str(e)
            }
        
        if 'error' in raw:
            return self._error_to_premium_result(raw, df, config, start_time)
        
        # Handle multi-variant mode
        if raw.get('mode') == 'all_variants':
            return self._multi_variant_to_premium_result(raw, df, config, start_time)
        
        periods = raw.get('forecast_periods', 30)
        mae = raw.get('metrics', {}).get('mae', 0)
        rmse = raw.get('metrics', {}).get('rmse', 0)
        mape = raw.get('metrics', {}).get('mape', 0)
        train_size = raw.get('metrics', {}).get('train_size', 0)
        test_size = raw.get('metrics', {}).get('test_size', 0)
        
        # Extract forecast data for frontend
        forecast_df = raw.get('forecast')
        forecast_list = []
        if forecast_df is not None and hasattr(forecast_df, 'to_dict'):
            # Convert DataFrame to list of dicts for JSON
            forecast_list = forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_dict('records')
            # Convert timestamps to strings
            for f in forecast_list:
                if hasattr(f['ds'], 'isoformat'):
                    f['ds'] = f['ds'].isoformat()
        
        variants = [Variant(
            rank=1, gemma_score=max(1, 100 - int(mae)), cv_score=1 / (1 + mae),
            variant_type="forecast", model_name="Prophet",
            features_used=[raw.get('target_column', '')],
            interpretation=f"{periods}-period forecast with MAE={mae:.2f}",
            details={
                'mae': mae, 
                'rmse': rmse,
                'mape': mape,
                'train_size': train_size,
                'test_size': test_size,
                'historical_periods': raw.get('historical_periods', 0),
                'seasonality': raw.get('seasonality', {})
            }
        )]
        
        features = [FeatureImportance(
            name=raw.get('target_column', 'target'),
            stability=80.0, importance=1.0, impact="positive",
            explanation=f"Forecasted {periods} periods ahead"
        )]
        
        # Build premium result with extra chronos-specific fields
        result = PremiumResult(
            engine_name="chronos", engine_display_name="Chronos Forecast", engine_icon="⏰",
            task_type=TaskType.FORECASTING, target_column=raw.get('target_column'),
            columns_analyzed=[raw.get('time_column', ''), raw.get('target_column', '')], 
            row_count=len(df),
            variants=variants, best_variant=variants[0], feature_importance=features,
            summary=PlainEnglishSummary(
                f"Generated {periods}-period forecast",
                f"Prophet model forecasts {raw.get('target_column')} with MAE of {mae:.2f}. Seasonality patterns detected and included.",
                "Use forecast for planning and risk assessment.",
                Confidence.HIGH if mae < 10 else Confidence.MEDIUM
            ),
            explanation=TechnicalExplanation(
                "Facebook Prophet Time Series Forecasting",
                "https://facebook.github.io/prophet/",
                [ExplanationStep(1, "Data Prep", "Formatted time series data"),
                 ExplanationStep(2, "Trend Detection", "Identified long-term trends"),
                 ExplanationStep(3, "Seasonality", "Detected weekly/yearly patterns"),
                 ExplanationStep(4, "Forecasting", f"Generated {periods} future predictions")],
                ["Assumes patterns continue", "Sensitive to outliers"]
            ),
            holdout=None, config_used=config,
            config_schema=[ConfigParameter(k, v['type'], v['default'], 
                          v.get('options') or ([v.get('min'), v.get('max')] if 'min' in v else None), 
                          v.get('description', '')) for k, v in CHRONOS_CONFIG_SCHEMA.items()],
            execution_time_seconds=time.time() - start_time, warnings=[]
        )
        
        # Convert to dict and add Chronos-specific extra fields
        result_dict = result.to_dict()
        result_dict['forecast'] = forecast_list
        result_dict['insights'] = raw.get('insights', [])
        result_dict['metrics'] = {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'train_size': train_size,
            'test_size': test_size
        }
        result_dict['historical_periods'] = raw.get('historical_periods', 0)
        result_dict['forecast_periods'] = periods
        result_dict['time_column'] = raw.get('time_column', '')
        
        return result_dict
    
    def _error_to_premium_result(self, raw, df, config, start):
        import time
        result = PremiumResult(
            engine_name="chronos", engine_display_name="Chronos Forecast", engine_icon="⏰",
            task_type=TaskType.FORECASTING, target_column=None, columns_analyzed=list(df.columns),
            row_count=len(df), variants=[],
            best_variant=Variant(1, 0, 0.0, "error", "None", [], raw.get('message', 'Error'), {}),
            feature_importance=[],
            summary=PlainEnglishSummary("Forecasting failed", raw.get('message', 'Error'), 
                                       "Provide time series data.", Confidence.LOW),
            explanation=TechnicalExplanation("Prophet", None, [], ["Forecasting failed"]),
            holdout=None, config_used=config, config_schema=[],
            execution_time_seconds=time.time() - start, warnings=[raw.get('message', '')]
        )
        result_dict = result.to_dict()
        result_dict['forecast'] = []
        result_dict['insights'] = []
        result_dict['metrics'] = {}
        result_dict['historical_periods'] = 0
        result_dict['forecast_periods'] = 0
        result_dict['time_column'] = ''
        return result_dict

    def _multi_variant_to_premium_result(self, raw, df, config, start_time):
        """Convert multi-variant analysis to PremiumResult with high-precision metrics."""
        import time
        
        best = raw.get('best_variant')
        all_variants = raw.get('variants', [])
        
        # Build variant objects for PremiumResult
        premium_variants = []
        for i, v in enumerate(all_variants):
            if v['success']:
                premium_variants.append(Variant(
                    rank=v.get('rank', i + 1),
                    gemma_score=max(1, int(v['metrics']['accuracy'])),
                    cv_score=1 / (1 + v['metrics']['mae']) if v['metrics']['mae'] > 0 else 1.0,
                    variant_type="forecast",
                    model_name=v['name'],
                    features_used=[raw.get('target_column', '')],
                    interpretation=f"{v['description']} - Accuracy: {v['metrics']['accuracy']:.10f}%",
                    details={
                        'short_name': v['short_name'],
                        'mae': v['metrics']['mae'],
                        'rmse': v['metrics']['rmse'],
                        'mape': v['metrics']['mape'],
                        'accuracy': v['metrics']['accuracy'],
                        'execution_time': v['execution_time_seconds'],
                        'config': v['config']
                    }
                ))
        
        # Use best variant metrics for summary
        best_mae = best['metrics']['mae'] if best else 0
        best_mape = best['metrics']['mape'] if best else 0
        best_accuracy = best['metrics']['accuracy'] if best else 0
        best_name = best['name'] if best else 'None'
        
        periods = raw.get('forecast_periods', 30)
        
        result = PremiumResult(
            engine_name="chronos", 
            engine_display_name="Chronos Multi-Variant Forecast", 
            engine_icon="⏰",
            task_type=TaskType.FORECASTING, 
            target_column=raw.get('target_column'),
            columns_analyzed=[raw.get('time_column', ''), raw.get('target_column', '')], 
            row_count=len(df),
            variants=premium_variants, 
            best_variant=premium_variants[0] if premium_variants else None, 
            feature_importance=[FeatureImportance(
                name=raw.get('target_column', 'target'),
                stability=80.0, importance=1.0, impact="positive",
                explanation=f"Compared {len(all_variants)} Prophet configurations"
            )],
            summary=PlainEnglishSummary(
                f"Multi-Variant Forecast: {len(all_variants)} models compared",
                f"Best: {best_name} with {best_accuracy:.10f}% accuracy (MAPE: {best_mape:.10f}%)",
                "Use the best variant for predictions, or compare all models.",
                Confidence.HIGH if best_accuracy > 90 else Confidence.MEDIUM
            ),
            explanation=TechnicalExplanation(
                "Facebook Prophet Multi-Configuration Analysis",
                "https://facebook.github.io/prophet/",
                [ExplanationStep(1, "Config Variants", f"Tested {len(all_variants)} configurations"),
                 ExplanationStep(2, "Growth Models", "Linear, Flat with different flexibility"),
                 ExplanationStep(3, "Seasonality", "Additive and Multiplicative patterns"),
                 ExplanationStep(4, "Best Selection", f"Selected {best_name}")],
                ["Each variant optimized differently", "Compare based on your data characteristics"]
            ),
            holdout=None, config_used=config,
            config_schema=[ConfigParameter(k, v['type'], v['default'], 
                          v.get('options') or ([v.get('min'), v.get('max')] if 'min' in v else None), 
                          v.get('description', '')) for k, v in CHRONOS_CONFIG_SCHEMA.items()],
            execution_time_seconds=time.time() - start_time, 
            warnings=[]
        )
        
        result_dict = result.to_dict()
        
        # Add all variant details with high-precision metrics
        result_dict['all_variants'] = []
        for v in all_variants:
            if v['success']:
                result_dict['all_variants'].append({
                    'name': v['name'],
                    'short_name': v['short_name'],
                    'description': v['description'],
                    'rank': v.get('rank', 0),
                    'config': v['config'],
                    'metrics': {
                        'mae': float(v['metrics']['mae']),
                        'rmse': float(v['metrics']['rmse']),
                        'mape': float(v['metrics']['mape']),
                        'accuracy': float(v['metrics']['accuracy']),
                        'train_size': v['metrics']['train_size'],
                        'test_size': v['metrics']['test_size']
                    },
                    'execution_time_seconds': v['execution_time_seconds'],
                    'forecast': v['forecast']
                })
        
        # Best variant forecast for primary display
        result_dict['forecast'] = best['forecast'] if best else []
        result_dict['insights'] = raw.get('insights', [])
        
        # Best variant metrics
        result_dict['metrics'] = {
            'mae': best['metrics']['mae'] if best else 0,
            'rmse': best['metrics']['rmse'] if best else 0,
            'mape': best['metrics']['mape'] if best else 0,
            'accuracy': best['metrics']['accuracy'] if best else 0,
            'train_size': best['metrics']['train_size'] if best else 0,
            'test_size': best['metrics']['test_size'] if best else 0
        }
        
        result_dict['historical_periods'] = raw.get('historical_periods', 0)
        result_dict['forecast_periods'] = periods
        result_dict['time_column'] = raw.get('time_column', '')
        result_dict['mode'] = 'all_variants'
        result_dict['variant_count'] = len(all_variants)
        result_dict['successful_count'] = raw.get('successful_count', 0)
        result_dict['best_variant_name'] = best_name
        
        return result_dict


# CLI entry point
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python chronos_engine.py <csv_file> [--time COL] [--target COL] [--periods N]")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    
    time_col = None
    target_col = None
    forecast_periods = 30
    
    if '--time' in sys.argv:
        time_idx = sys.argv.index('--time')
        if len(sys.argv) > time_idx + 1:
            time_col = sys.argv[time_idx + 1]
    
    if '--target' in sys.argv:
        target_idx = sys.argv.index('--target')
        if len(sys.argv) > target_idx + 1:
            target_col = sys.argv[target_idx + 1]
    
    if '--periods' in sys.argv:
        periods_idx = sys.argv.index('--periods')
        if len(sys.argv) > periods_idx + 1:
            forecast_periods = int(sys.argv[periods_idx + 1])
    
    df = pd.read_csv(csv_file)
    
    engine = ChronosEngine()
    config = {
        'time_column': time_col,
        'target_column': target_col,
        'forecast_periods': forecast_periods
    }
    
    print(f"\n{'='*60}")
    print(f"CHRONOS ENGINE: Generating {forecast_periods}-period forecast...")
    print(f"{'='*60}\n")
    
    result = engine.analyze(df, config)
    
    print(f"\n{'='*60}")
    print(f"CHRONOS ENGINE RESULTS: {csv_file}")
    print(f"{'='*60}\n")
    
    if 'error' in result:
        print(f"❌ Error: {result['message']}")
        for insight in result.get('insights', []):
            print(f"   {insight}")
    else:
        for insight in result['insights']:
            print(insight)
        
        print(f"\n{'='*60}")
        print("FORECAST PREVIEW (Next 10 periods):")
        print(f"{'='*60}\n")
        
        forecast_df = result['forecast']
        last_historical_date = df[result['time_column']].max()
        future_df = forecast_df[forecast_df['ds'] > pd.to_datetime(last_historical_date)].head(10)
        
        print(future_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_string(index=False))
        
        # Save forecast
        output_file = csv_file.replace('.csv', '_forecast.csv')
        forecast_df.to_csv(output_file, index=False)
        print(f"\n✅ Full forecast saved to: {output_file}")
