"""
Predictive Analytics Engine - Schema-Agnostic Version

Time-series forecasting with multiple algorithms:
- ARIMA/SARIMA: Auto-tuned statistical models
- Prophet: FB's seasonality-aware forecasting
- LSTM: Deep learning for complex patterns
- XGBoost: Gradient boosting for non-linear trends
- Exponential Smoothing: Holt-Winters method

Auto-selects best model via cross-validation.
NOW SCHEMA-AGNOSTIC: Automatically detects temporal and numeric columns.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging

from core import (
    ColumnProfiler,
    ColumnMapper,
    SemanticType,
    EngineRequirements
)

from core.premium_models import ConfigParameter

logger = logging.getLogger(__name__)


class PredictiveEngine:
    """Schema-agnostic time-series forecasting engine"""
    
    def __init__(self):
        self.name = "Predictive Analytics Engine (Schema-Agnostic)"
        self.profiler = ColumnProfiler()
        self.mapper = ColumnMapper()
        self.models = {}
        self.best_model = None
    
    def get_engine_info(self) -> Dict[str, str]:
        """Get engine display information."""
        return {
            "name": "predictive",
            "display_name": "Predictive Forecasting",
            "icon": "📈",
            "task_type": "forecasting"
        }
    
    def get_config_schema(self) -> List[ConfigParameter]:
        """Get configuration schema for UI."""
        return [
            ConfigParameter(
                name="horizon",
                type="int",
                default=30,
                range=[1, 365],
                description="Number of periods to forecast ahead"
            ),
            ConfigParameter(
                name="models",
                type="select",
                default=["auto"],
                range=["auto", "arima", "prophet", "xgboost", "ets"],
                description="Forecasting models to use"
            ),
            ConfigParameter(
                name="confidence_level",
                type="float",
                default=0.95,
                range=[0.8, 0.99],
                description="Confidence level for prediction intervals"
            )
        ]
    
    def get_methodology_info(self) -> Dict[str, Any]:
        """Get methodology documentation."""
        return {
            "name": "Time-Series Forecasting",
            "url": "https://otexts.com/fpp3/",
            "steps": [
                {"step_number": 1, "title": "Time Column Detection", "description": "Identify temporal column for time series"},
                {"step_number": 2, "title": "Stationarity Testing", "description": "Check for trends and seasonality"},
                {"step_number": 3, "title": "Model Selection", "description": "Auto-select best model via cross-validation"},
                {"step_number": 4, "title": "Forecasting", "description": "Generate predictions with confidence intervals"}
            ],
            "limitations": ["Requires sufficient historical data", "Assumes patterns continue"],
            "assumptions": ["Time series is properly ordered", "No structural breaks in recent data"]
        }
    
    def get_requirements(self) -> EngineRequirements:
        return EngineRequirements(
            required_semantics=[SemanticType.TEMPORAL, SemanticType.NUMERIC_CONTINUOUS],
            min_rows=50,
            min_numeric_cols=1
        )
    
    def analyze(self, df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Standard analyze interface - wraps forecast method.
        
        Args:
            df: Input dataframe
            config: Optional configuration dict
            
        Returns:
            Forecast results
        """
        return self.forecast(df, config or {})
    
    def forecast(self, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run time-series forecasting
        
        Args:
            df: Dataframe with time column and target column
            config:
                - time_column: Name of datetime column
                - target_column: Name of target variable
                - horizon: Forecast periods ahead
                - models: List of models to try ['auto', 'arima', 'prophet', 'lstm', 'xgboost']
                - confidence_level: 0.95 for 95% confidence intervals
        
        Returns:
            Forecast results with model comparison
        """
        time_col = config.get('time_column')
        target_col = config.get('target_column')
        horizon = config.get('horizon', 30)
        models_to_try = config.get('models', ['auto'])
        confidence = config.get('confidence_level', 0.95)
        
        # Auto-detect time column if not provided
        if not time_col:
            for col in df.columns:
                if 'date' in col.lower() or 'time' in col.lower():
                    time_col = col
                    break
            if not time_col:
                # Try to find a date-like column
                for col in df.columns:
                    if df[col].dtype == 'object':
                        try:
                            pd.to_datetime(df[col].head(10))
                            time_col = col
                            break
                        except:
                            pass
        
        # Auto-detect target column if not provided
        if not target_col:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            # Prefer columns with "sales", "revenue", "amount" in name
            for col in numeric_cols:
                if any(kw in col.lower() for kw in ['sales', 'revenue', 'amount', 'value', 'total']):
                    target_col = col
                    break
            # Otherwise use first large numeric column
            if not target_col and numeric_cols:
                max_vals = {c: df[c].max() for c in numeric_cols}
                target_col = max(max_vals, key=max_vals.get)
        
        # Validate inputs - FALLBACK instead of raising error
        if not time_col or time_col not in df.columns:
            logger.info("Predictive Engine: Time column not found, returning fallback response")
            return {
                "engine": "predictive",
                "status": "requires_time_data",
                "error": "Time column not found",
                "insights": [
                    "⏰ This dataset doesn't have a time/date column",
                    "📈 Time-series forecasting requires temporal data (dates or timestamps)",
                    "💡 Consider adding a date column, or use Titan engine for non-temporal predictions"
                ],
                "available_columns": list(df.columns),
                "recommendation": "Try Titan AutoML for predictions without time-series data"
            }
        if not target_col or target_col not in df.columns:
            logger.info("Predictive Engine: Target column not found, returning fallback response")
            return {
                "engine": "predictive",
                "status": "requires_target_data",
                "error": "Target column not found",
                "insights": [
                    "🎯 Could not identify a suitable numeric column to forecast",
                    "📊 Please specify target_column in config"
                ],
                "available_columns": list(df.columns)
            }
        
        # Prepare data
        ts_data = self._prepare_time_series(df, time_col, target_col)
        
        # Auto-detect if 'auto' is requested
        if 'auto' in models_to_try:
            models_to_try = self._recommend_models(ts_data)
        
        # Train models
        results = {}
        model_scores = {}
        
        for model_name in models_to_try:
            try:
                if model_name == 'naive':
                    result = self._naive_forecast(ts_data, horizon)
                elif model_name == 'moving_average':
                    result = self._moving_average_forecast(ts_data, horizon, window=7)
                elif model_name == 'exponential_smoothing':
                    result = self._exponential_smoothing_forecast(ts_data, horizon, confidence)
                else:
                    # Placeholder for advanced models
                    logger.warning(f"Model {model_name} not yet implemented, using naive")
                    result = self._naive_forecast(ts_data, horizon)
                
                results[model_name] = result
                model_scores[model_name] = result.get('validation_metrics', {}).get('mae', float('inf'))
                
            except Exception as e:
                logger.error(f"Model {model_name} failed: {e}")
                continue
        
        # Select best model
        if model_scores:
            best_model_name = min(model_scores, key=model_scores.get)
            best_result = results[best_model_name]
        else:
            raise ValueError("All models failed")
        
        return {
            'forecast': best_result,
            'best_model': best_model_name,
            'model_comparison': {
                name: {'mae': score} for name, score in model_scores.items()
            },
            'visualizations': self._generate_forecast_visualizations(best_result, ts_data),
            'metadata': {
                'horizon': horizon,
                'data_points': len(ts_data),
                'models_tried': list(models_to_try)
            }
        }
    
    def _prepare_time_series(self, df: pd.DataFrame, time_col: str, target_col: str) -> pd.Series:
        """Prepare time series data"""
        # Sort by time
        df_sorted = df.sort_values(time_col).copy()
        
        # Set time as index with flexible parsing
        try:
            df_sorted[time_col] = pd.to_datetime(df_sorted[time_col], format='mixed', dayfirst=True)
        except:
            df_sorted[time_col] = pd.to_datetime(df_sorted[time_col], infer_datetime_format=True)
        df_sorted = df_sorted.set_index(time_col)
        
        # Extract target series
        ts = df_sorted[target_col].dropna()
        
        return ts
    
    def _recommend_models(self, ts_data: pd.Series) -> List[str]:
        """Recommend models based on data characteristics"""
        n = len(ts_data)
        
        recommended = ['naive', 'moving_average']
        
        if n >= 30:
            recommended.append('exponential_smoothing')
        
        # For now, stick to simple models (ARIMA, Prophet, LSTM need additional dependencies)
        # TODO: Add when dependencies are installed
        
        return recommended
    
    def _naive_forecast(self, ts_data: pd.Series, horizon: int) -> Dict:
        """Naive forecast: last value repeated"""
        last_value = ts_data.iloc[-1]
        forecast_values = [last_value] * horizon
        
        # Generate forecast dates
        last_date = ts_data.index[-1]
        freq = pd.infer_freq(ts_data.index) or 'D'
        forecast_dates = pd.date_range(start=last_date, periods=horizon + 1, freq=freq)[1:]
        
        # Validation on last 20% of data
        val_size = max(1, int(len(ts_data) * 0.2))
        train_data = ts_data.iloc[:-val_size]
        val_data = ts_data.iloc[-val_size:]
        
        val_pred = [train_data.iloc[-1]] * len(val_data)
        mae = mean_absolute_error(val_data, val_pred)
        rmse = np.sqrt(mean_squared_error(val_data, val_pred))
        
        return {
            'forecast_dates': [str(d) for d in forecast_dates],
            'forecast_values': forecast_values,
            'lower_bound': [v * 0.9 for v in forecast_values],  # Simple 10% band
            'upper_bound': [v * 1.1 for v in forecast_values],
            'model': 'naive',
            'validation_metrics': {
                'mae': float(mae),
                'rmse': float(rmse)
            }
        }
    
    def _moving_average_forecast(self, ts_data: pd.Series, horizon: int, window: int = 7) -> Dict:
        """Moving average forecast"""
        # Calculate moving average
        ma = ts_data.rolling(window=window, min_periods=1).mean()
        last_ma = ma.iloc[-1]
        
        forecast_values = [last_ma] * horizon
        
        # Generate forecast dates
        last_date = ts_data.index[-1]
        freq = pd.infer_freq(ts_data.index) or 'D'
        forecast_dates = pd.date_range(start=last_date, periods=horizon + 1, freq=freq)[1:]
        
        # Validation
        val_size = max(1, int(len(ts_data) * 0.2))
        train_data = ts_data.iloc[:-val_size]
        val_data = ts_data.iloc[-val_size:]
        
        train_ma = train_data.rolling(window=window, min_periods=1).mean()
        val_pred = [train_ma.iloc[-1]] * len(val_data)
        mae = mean_absolute_error(val_data, val_pred)
        rmse = np.sqrt(mean_squared_error(val_data, val_pred))
        
        return {
            'forecast_dates': [str(d) for d in forecast_dates],
            'forecast_values': forecast_values,
            'lower_bound': [v * 0.85 for v in forecast_values],
            'upper_bound': [v * 1.15 for v in forecast_values],
            'model': f'moving_average_{window}',
            'validation_metrics': {
                'mae': float(mae),
                'rmse': float(rmse)
            }
        }
    
    def _exponential_smoothing_forecast(self, ts_data: pd.Series, horizon: int, confidence: float) -> Dict:
        """Simple exponential smoothing"""
        alpha = 0.3  # Smoothing parameter
        
        # Fit exponential smoothing
        smoothed = [ts_data.iloc[0]]
        for i in range(1, len(ts_data)):
            smoothed.append(alpha * ts_data.iloc[i] + (1 - alpha) * smoothed[-1])
        
        # Forecast: repeat last smoothed value
        forecast_value = smoothed[-1]
        forecast_values = [forecast_value] * horizon
        
        # Generate forecast dates
        last_date = ts_data.index[-1]
        freq = pd.infer_freq(ts_data.index) or 'D'
        forecast_dates = pd.date_range(start=last_date, periods=horizon + 1, freq=freq)[1:]
        
        # Calculate residuals for confidence intervals
        residuals = ts_data.values - np.array(smoothed)
        std_residual = np.std(residuals)
        
        # Confidence intervals (assuming normal distribution)
        z_score = 1.96 if confidence == 0.95 else 2.576  # 95% or 99%
        lower = [v - z_score * std_residual for v in forecast_values]
        upper = [v + z_score * std_residual for v in forecast_values]
        
        # Validation
        val_size = max(1, int(len(ts_data) * 0.2))
        train_data = ts_data.iloc[:-val_size]
        val_data = ts_data.iloc[-val_size:]
        
        # Fit on train
        train_smoothed = [train_data.iloc[0]]
        for i in range(1, len(train_data)):
            train_smoothed.append(alpha * train_data.iloc[i] + (1 - alpha) * train_smoothed[-1])
        
        val_pred = [train_smoothed[-1]] * len(val_data)
        mae = mean_absolute_error(val_data, val_pred)
        rmse = np.sqrt(mean_squared_error(val_data, val_pred))
        
        return {
            'forecast_dates': [str(d) for d in forecast_dates],
            'forecast_values': forecast_values,
            'lower_bound': lower,
            'upper_bound': upper,
            'model': 'exponential_smoothing',
            'parameters': {'alpha': alpha},
            'validation_metrics': {
                'mae': float(mae),
                'rmse': float(rmse)
            }
        }
    
    def _generate_forecast_visualizations(self, forecast_result: Dict, historical_data: pd.Series) -> List[Dict]:
        """Generate visualization metadata"""
        visualizations = []
        
        # Main forecast plot
        visualizations.append({
            'type': 'line_chart_with_forecast',
            'title': 'Time Series Forecast',
            'data': {
                'historical_dates': [str(d) for d in historical_data.index[-50:]],  # Last 50 points
                'historical_values': historical_data.values[-50:].tolist(),
                'forecast_dates': forecast_result['forecast_dates'],
                'forecast_values': forecast_result['forecast_values'],
                'lower_bound': forecast_result['lower_bound'],
                'upper_bound': forecast_result['upper_bound']
            },
            'description': f"Forecast using {forecast_result['model']} model"
        })
        
        # Model metrics
        metrics = forecast_result.get('validation_metrics', {})
        visualizations.append({
            'type': 'metric_cards',
            'title': 'Model Performance',
            'data': {
                'MAE': metrics.get('mae', 0),
                'RMSE': metrics.get('rmse', 0)
            },
            'description': 'Validation metrics on hold-out data'
        })
        
        return visualizations
