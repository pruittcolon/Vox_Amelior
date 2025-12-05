"""
Customer Lifetime Value (LTV) Engine

Calculates CLV, churn probability, and segments customers by value.
Identifies high-value customers and at-risk accounts.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime, timedelta

from core import (
    ColumnProfiler,
    ColumnMapper,
    SemanticType,
    EngineRequirements,
    DatasetClassifier
)
from core.gemma_summarizer import GemmaSummarizer

from core.premium_models import ConfigParameter

logger = logging.getLogger(__name__)


class CustomerLTVEngine:
    """
    Schema-agnostic engine for Customer Lifetime Value (LTV) analysis.
    Uses RFM analysis and cohort-based predictions.
    """
    
    def __init__(self):
        self.name = "Customer LTV Engine (Schema-Agnostic)"
        self.profiler = ColumnProfiler()
        self.mapper = ColumnMapper()
    
    def get_engine_info(self) -> Dict[str, str]:
        return {
            "name": "customer_ltv",
            "display_name": "Customer LTV",
            "icon": "👤",
            "task_type": "prediction"
        }
    
    def get_config_schema(self) -> List[ConfigParameter]:
        return [
            ConfigParameter(name="customer_column", type="select", default=None, range=[], description="Customer identifier column"),
            ConfigParameter(name="amount_column", type="select", default=None, range=[], description="Transaction amount column"),
            ConfigParameter(name="date_column", type="select", default=None, range=[], description="Transaction date column")
        ]
    
    def get_methodology_info(self) -> Dict[str, Any]:
        return {
            "name": "Customer Lifetime Value Analysis",
            "url": "https://en.wikipedia.org/wiki/Customer_lifetime_value",
            "steps": [
                {"step_number": 1, "title": "RFM Analysis", "description": "Calculate Recency, Frequency, Monetary values"},
                {"step_number": 2, "title": "CLV Calculation", "description": "Compute customer lifetime value"},
                {"step_number": 3, "title": "Segmentation", "description": "Segment customers by value tier"}
            ],
            "limitations": ["Requires transaction history"],
            "assumptions": ["Customer behavior is relatively stable"]
        }
    
    def get_requirements(self) -> EngineRequirements:
        return EngineRequirements(
            required_semantics=[SemanticType.NUMERIC_CONTINUOUS, SemanticType.TEMPORAL],
            optional_semantics={'customer': [SemanticType.CATEGORICAL], 'revenue': [SemanticType.NUMERIC_CONTINUOUS]},
            required_entities=[],
            preferred_domains=[],
            applicable_tasks=[],
            min_rows=10,
            min_numeric_cols=1
        )
        
    def analyze(self, df: pd.DataFrame, config: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Run schema-agnostic CLV analysis.
        
        Args:
            df: DataFrame with transaction data
            config: Optional configuration
        
        Returns:
            Dict with CLV analysis and visualizations
        """
        config = config or {}
        margin = config.get('margin', 0.2)
        discount_rate = config.get('discount_rate', 0.1)
        
        results = {
            'summary': {},
            'segments': {},
            'graphs': [],
            'insights': [],
            'column_mappings': {},
            'profiling_used': not config.get('skip_profiling', False)
        }
        
        try:
            # SCHEMA INTELLIGENCE
            if not config.get('skip_profiling', False):
                profiles = self.profiler.profile_dataset(df)
                
                cust_col = self._smart_detect(df, profiles, 
                    keywords=['customer', 'user', 'client', 'account'],
                    hint=config.get('customer_column'),
                    semantic_type=SemanticType.CATEGORICAL)
                
                date_col = self._smart_detect(df, profiles,
                    keywords=['date', 'time', 'created', 'timestamp'],
                    hint=config.get('date_column'),
                    semantic_type=SemanticType.TEMPORAL)
                
                amt_col = self._smart_detect(df, profiles,
                    keywords=['amount', 'total', 'price', 'revenue', 'value'],
                    hint=config.get('amount_column'))
                
                if cust_col:
                    results['column_mappings']['customer'] = {'column': cust_col, 'confidence': 0.9}
                if date_col:
                    results['column_mappings']['date'] = {'column': date_col, 'confidence': 0.9}
                if amt_col:
                    results['column_mappings']['amount'] = {'column': amt_col, 'confidence': 0.9}
            else:
                cust_col = config.get('customer_column') or self._detect_column(df, ['customer', 'user'], None)
                date_col = config.get('date_column') or self._detect_column(df, ['date', 'time'], None)
                amt_col = config.get('amount_column') or self._detect_column(df, ['amount', 'total'], None)
            
            if not (cust_col and date_col and amt_col):
                # Use Gemma fallback when required columns are missing
                missing = []
                if not cust_col:
                    missing.append('customer')
                if not date_col:
                    missing.append('date')
                if not amt_col:
                    missing.append('amount')
                return GemmaSummarizer.generate_fallback_summary(
                    df,
                    engine_name="customer_ltv",
                    error_reason=f"Could not detect {', '.join(missing)} columns. LTV analysis requires customer ID, transaction date, and amount columns.",
                    config=config
                )
            
            # Preprocessing - runs when all columns detected
            df[date_col] = pd.to_datetime(df[date_col])
            current_date = df[date_col].max()
            
            # RFM Analysis (Recency, Frequency, Monetary)
            rfm = df.groupby(cust_col).agg({
                date_col: lambda x: (current_date - x.max()).days,
                amt_col: ['count', 'sum', 'mean']
            }).reset_index()
            
            rfm.columns = ['customer', 'recency', 'frequency', 'monetary', 'avg_order_value']
            
            # Simple CLV Calculation
            # CLV = ((Average Order Value * Purchase Frequency) / Churn Rate) * Profit Margin
            
            # Estimate churn rate (simplified: customers not seen in last 90 days)
            inactive_threshold = 90
            churned_customers = rfm[rfm['recency'] > inactive_threshold]
            churn_rate = len(churned_customers) / len(rfm) if len(rfm) > 0 else 0.1
            churn_rate = max(churn_rate, 0.01) # Avoid division by zero
            
            # Calculate Annual Value
            rfm['annual_value'] = rfm['monetary'] * (365 / (rfm['recency'] + 1)) # Rough estimate
            
            # Calculate CLV
            rfm['clv'] = (rfm['avg_order_value'] * rfm['frequency']) / churn_rate * margin
            
            # Segmentation
            rfm['segment'] = pd.qcut(rfm['clv'], q=4, labels=['Low Value', 'Medium Value', 'High Value', 'VIP'], duplicates='drop')
            
            # Summary Stats
            results['summary'] = {
                'avg_clv': float(rfm['clv'].mean()),
                'total_clv': float(rfm['clv'].sum()),
                'churn_rate': float(churn_rate * 100),
                'active_customers': len(rfm) - len(churned_customers)
            }
            
            # Visualizations
            
            # 1. CLV Distribution
            clv_dist = self._clv_distribution(rfm)
            results['graphs'].append(clv_dist['graph'])
            
            # 2. Customer Segments
            segments = self._segment_analysis(rfm)
            results['segments'] = segments
            results['graphs'].append(segments['graph'])
            
            # 3. Churn Risk
            churn_risk = self._churn_risk_analysis(rfm)
            results['graphs'].append(churn_risk['graph'])
            
            # Generate insights
            results['insights'] = self._generate_insights(results)
                
        except Exception as e:
            logger.error(f"CLV analysis failed: {e}")
            results['error'] = str(e)
            
        return results
    
    def _smart_detect(self, df: pd.DataFrame, profiles: Dict,
                     keywords: List[str], hint: Optional[str] = None,
                     semantic_type: Optional[SemanticType] = None) -> Optional[str]:
        """Smart column detection with semantic type filtering."""
        if hint and hint in df.columns:
            return hint
        
        for col in df.columns:
            if any(kw in col.lower() for kw in keywords):
                prof = profiles.get(col)
                if prof:
                    if semantic_type:
                        if prof.semantic_type == semantic_type:
                            return col
                    elif prof.semantic_type in [SemanticType.NUMERIC_CONTINUOUS, SemanticType.NUMERIC_DISCRETE, SemanticType.TEMPORAL, SemanticType.CATEGORICAL]:
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
    
    def _clv_distribution(self, rfm: pd.DataFrame) -> Dict:
        """Analyze CLV distribution"""
        # Filter outliers for better visualization
        q95 = rfm['clv'].quantile(0.95)
        filtered = rfm[rfm['clv'] <= q95]
        
        graph = {
            'type': 'histogram',
            'title': 'Customer Lifetime Value Distribution',
            'x_data': filtered['clv'].tolist(),
            'x_label': 'CLV ($)',
            'bins': 30
        }
        
        return {'graph': graph}
    
    def _segment_analysis(self, rfm: pd.DataFrame) -> Dict:
        """Analyze customer segments"""
        segment_stats = rfm.groupby('segment')['clv'].agg(['count', 'sum', 'mean']).reset_index()
        
        graph = {
            'type': 'pie_chart',
            'title': 'Customer Segments by Value',
            'labels': segment_stats['segment'].astype(str).tolist(),
            'values': segment_stats['count'].tolist(),
            'percentages': (segment_stats['count'] / segment_stats['count'].sum() * 100).tolist()
        }
        
        return {
            'stats': segment_stats.to_dict('records'),
            'graph': graph
        }
    
    def _churn_risk_analysis(self, rfm: pd.DataFrame) -> Dict:
        """Analyze churn risk based on recency"""
        rfm['risk_level'] = pd.cut(rfm['recency'], 
                                  bins=[-1, 30, 60, 90, 1000], 
                                  labels=['Active', 'At Risk', 'High Risk', 'Lost'])
        
        risk_counts = rfm['risk_level'].value_counts().sort_index()
        
        graph = {
            'type': 'bar_chart',
            'title': 'Customer Churn Risk (Recency)',
            'x_data': risk_counts.index.astype(str).tolist(),
            'y_data': risk_counts.values.tolist(),
            'x_label': 'Risk Level',
            'y_label': 'Number of Customers',
            'colors': ['#10b981', '#f59e0b', '#ef4444', '#64748b']
        }
        
        return {'graph': graph}
    
    def _generate_insights(self, results: Dict) -> List[str]:
        """Generate insights"""
        insights = []
        
        summary = results['summary']
        insights.append(
            f"👥 Average Customer Lifetime Value: ${summary['avg_clv']:,.2f}"
        )
        
        insights.append(
            f"📉 Estimated Churn Rate: {summary['churn_rate']:.1f}%"
        )
        
        if 'segments' in results:
            vip_stats = [s for s in results['segments']['stats'] if s['segment'] == 'VIP'][0]
            insights.append(
                f"💎 VIP Segment: {vip_stats['count']} customers contribute "
                f"${vip_stats['sum']:,.2f} total value"
            )
            
        return insights


__all__ = ['CustomerLTVEngine']
