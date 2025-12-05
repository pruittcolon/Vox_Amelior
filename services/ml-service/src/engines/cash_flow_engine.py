"""
Cash Flow Prediction Engine - Schema-Agnostic Version

Analyzes cash inflows and outflows, predicts future cash positions.

NOW SCHEMA-AGNOSTIC: Works with any dataset structure via semantic column mapping.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging

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
from core.gemma_summarizer import GemmaSummarizer

from core.premium_models import ConfigParameter

logger = logging.getLogger(__name__)


class CashFlowEngine:
    """
    Schema-agnostic cash flow prediction engine.
    
    Automatically detects inflow, outflow, and temporal columns.
    """
    
    def __init__(self):
        self.name = "Cash Flow Prediction Engine (Schema-Agnostic)"
        self.profiler = ColumnProfiler()
        self.mapper = ColumnMapper()
    
    def get_engine_info(self) -> Dict[str, str]:
        return {
            "name": "cash_flow",
            "display_name": "Cash Flow Analysis",
            "icon": "💸",
            "task_type": "forecasting"
        }
    
    def get_config_schema(self) -> List[ConfigParameter]:
        return [
            ConfigParameter(name="inflow_column", type="select", default=None, range=[], description="Cash inflow column"),
            ConfigParameter(name="outflow_column", type="select", default=None, range=[], description="Cash outflow column"),
            ConfigParameter(name="date_column", type="select", default=None, range=[], description="Date column")
        ]
    
    def get_methodology_info(self) -> Dict[str, Any]:
        return {
            "name": "Cash Flow Analysis",
            "url": "https://en.wikipedia.org/wiki/Cash_flow",
            "steps": [
                {"step_number": 1, "title": "Flow Detection", "description": "Identify inflow and outflow columns"},
                {"step_number": 2, "title": "Net Position", "description": "Calculate net cash position over time"},
                {"step_number": 3, "title": "Forecasting", "description": "Predict future cash positions"}
            ],
            "limitations": ["Requires complete transaction data"],
            "assumptions": ["Historical patterns inform forecasts"]
        }
    
    def get_requirements(self) -> EngineRequirements:
        return EngineRequirements(
            required_semantics=[SemanticType.NUMERIC_CONTINUOUS, SemanticType.TEMPORAL],
            optional_semantics={'inflow': [SemanticType.NUMERIC_CONTINUOUS], 'outflow': [SemanticType.NUMERIC_CONTINUOUS]},
            required_entities=[],
            preferred_domains=[],
            applicable_tasks=[],
            min_rows=10,
            min_numeric_cols=1
        )
        
    def analyze(self, df: pd.DataFrame, config: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Run schema-agnostic cash flow analysis.
        
        Args:
            df: DataFrame with transaction data
            config: Optional configuration
        
        Returns:
            Dict with cash flow analysis
        """
        config = config or {}
        
        results = {
            'summary': {},
            'forecast': [],
            'graphs': [],
            'insights': [],
            'column_mappings': {},
            'profiling_used': not config.get('skip_profiling', False)
        }
        
        try:
            # SCHEMA INTELLIGENCE
            if not config.get('skip_profiling', False):
                logger.info("Profiling dataset for cash flow analysis")
                profiles = self.profiler.profile_dataset(df)
                
                classifier = DatasetClassifier()
                classification = classifier.classify(profiles, len(df))
                results['summary']['detected_domain'] = classification.domain.value
                
                date_col = self._smart_detect(
                    df, profiles,
                    keywords=['date', 'time', 'day', 'timestamp'],
                    hint=config.get('date_column'),
                    semantic_type=SemanticType.TEMPORAL
                )
                
                amt_col = self._smart_detect(
                    df, profiles,
                    keywords=['amount', 'value', 'transaction', 'cash', 'flow'],
                    hint=config.get('amount_column')
                )
                
                type_col = self._smart_detect(
                    df, profiles,
                    keywords=['type', 'category', 'direction'],
                    hint=config.get('type_column'),
                    semantic_type=SemanticType.CATEGORICAL
                )
                
                if date_col:
                    results['column_mappings']['temporal'] = {'column': date_col, 'confidence': 0.95}
                if amt_col:
                    results['column_mappings']['amount'] = {'column': amt_col, 'confidence': 0.9}
                if type_col:
                    results['column_mappings']['type'] = {'column': type_col, 'confidence': 0.85}
            else:
                date_col = config.get('date_column') or self._detect_column(df, ['date', 'time', 'day'], None)
                amt_col = config.get('amount_column') or self._detect_column(df, ['amount', 'value', 'transaction'], None)
                type_col = config.get('type_column') or self._detect_column(df, ['type', 'category', 'direction'], None)
                results['profiling_used'] = False
            
            if not (date_col and amt_col):
                # Use Gemma fallback when required columns are missing
                missing = []
                if not date_col:
                    missing.append('date/time')
                if not amt_col:
                    missing.append('amount/value')
                return GemmaSummarizer.generate_fallback_summary(
                    df,
                    engine_name="cash_flow",
                    error_reason=f"Could not detect {' and '.join(missing)} columns. Cash flow analysis requires date and amount columns.",
                    config=config
                )
            
            results['summary']['date_column'] = date_col
            results['summary']['amount_column'] = amt_col
            
            # ANALYSIS PIPELINE
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.sort_values(date_col)
            
            # Normalize amounts
            if type_col:
                mask = df[type_col].astype(str).str.lower().str.contains('out|expense|debit')
                df.loc[mask, amt_col] = -df.loc[mask, amt_col].abs()
                df.loc[~mask, amt_col] = df.loc[~mask, amt_col].abs()
                
                # === ADD: inflows/outflows for Sankey visualization ===
                try:
                    inflow_mask = ~mask
                    outflow_mask = mask
                    
                    inflow_df = df[inflow_mask].copy()
                    outflow_df = df[outflow_mask].copy()
                    
                    if len(inflow_df) > 0:
                        inflows = inflow_df.groupby(type_col)[amt_col].sum().abs()
                        results['inflows'] = {str(k): float(v) for k, v in inflows.items()}
                    else:
                        results['inflows'] = {'Revenue': float(df[df[amt_col] > 0][amt_col].sum())}
                    
                    if len(outflow_df) > 0:
                        outflows = outflow_df.groupby(type_col)[amt_col].sum().abs()
                        results['outflows'] = {str(k): float(v) for k, v in outflows.items()}
                    else:
                        results['outflows'] = {'Expenses': float(df[df[amt_col] < 0][amt_col].abs().sum())}
                except Exception as e:
                    logger.warning(f"Could not build inflows/outflows: {e}")
                    results['inflows'] = {'Revenue': float(df[df[amt_col] > 0][amt_col].sum())}
                    results['outflows'] = {'Expenses': float(df[df[amt_col] < 0][amt_col].abs().sum())}
            else:
                # No type column - split by positive/negative
                results['inflows'] = {'Revenue': float(df[df[amt_col] > 0][amt_col].sum())}
                results['outflows'] = {'Expenses': float(df[df[amt_col] < 0][amt_col].abs().sum())}
            
            # Calculate Cumulative Cash Flow
            df['cumulative_cash'] = df[amt_col].cumsum()
            
            # Monthly Aggregation
            df.set_index(date_col, inplace=True)
            monthly = df[amt_col].resample('ME').sum()
            monthly_cumulative = df['cumulative_cash'].resample('ME').last()
            
            # Burn Rate
            negative_months = monthly[monthly < 0]
            burn_rate = abs(negative_months.mean()) if len(negative_months) > 0 else 0
            
            # Runway
            current_cash = df['cumulative_cash'].iloc[-1]
            runway = (current_cash / burn_rate) if burn_rate > 0 else float('inf')
            
            results['summary'].update({
                'current_balance': float(current_cash),
                'net_cash_flow': float(df[amt_col].sum()),
                'burn_rate': float(burn_rate),
                'runway_months': float(runway) if runway != float('inf') else 'Infinite'
            })
            
            # Visualizations
            cf_chart = {
                'type': 'bar_line_combo',
                'title': 'Cash Flow & Balance',
                'x_data': [d.strftime('%Y-%m') for d in monthly.index],
                'bar_data': monthly.values.tolist(),
                'line_data': monthly_cumulative.values.tolist(),
                'bar_label': 'Net Flow',
                'line_label': 'Cash Balance'
            }
            results['graphs'].append(cf_chart)
            
            # Generate insights
            results['insights'] = self._generate_insights(results)
            
        except Exception as e:
            logger.error(f"Cash flow analysis failed: {e}", exc_info=True)
            
        return results
    
    def _smart_detect(self, df: pd.DataFrame, profiles: Dict,
                     keywords: List[str], hint: Optional[str] = None,
                     semantic_type: Optional[SemanticType] = None) -> Optional[str]:
        """Smart column detection."""
        if hint and hint in df.columns:
            return hint
        
        for col in df.columns:
            if any(kw in col.lower() for kw in keywords):
                prof = profiles.get(col)
                if prof:
                    if semantic_type:
                        if prof.semantic_type == semantic_type:
                            return col
                    elif prof.semantic_type in [SemanticType.NUMERIC_CONTINUOUS, SemanticType.NUMERIC_DISCRETE]:
                        return col
        
        if semantic_type:
            for col, prof in profiles.items():
                if prof.semantic_type == semantic_type:
                    return col
        
        return None
    
    def _detect_column(self, df: pd.DataFrame, keywords: List[str], hint: Optional[str] = None) -> Optional[str]:
        if hint and hint in df.columns:
            return hint
        for col in df.columns:
            if any(kw in col.lower() for kw in keywords):
                return col
        return None
    
    
    def _generate_insights(self, results: Dict, date_col: str = None, amt_col: str = None) -> List[str]:
        """Generate insights mentioning detected columns."""
        insights = []
        
        if date_col and amt_col:
            insights.append(f"💰 Analyzing cash flow from '{amt_col}' over time ('{date_col}')")
        
        summary = results['summary']
        balance = summary.get('current_balance', 0)
        
        if balance < 0:
            insights.append(f"⚠️ Negative Cash Balance: ${balance:,.2f}")
        else:
            insights.append(f"💰 Current Cash Balance: ${balance:,.2f}")
            
        if summary['burn_rate'] > 0:
            runway = summary['runway_months']
            runway_str = f"{runway:.1f}" if isinstance(runway, float) else str(runway)
            insights.append(
                f"🔥 Monthly Burn Rate: ${summary['burn_rate']:,.2f} "
                f"(Runway: {runway_str} months)"
            )
        else:
            insights.append("✅ Positive Cash Flow: No burn rate detected")
            
        return insights


__all__ = ['CashFlowEngine']
