"""
Spend Pattern Analysis Engine - Schema-Agnostic Version

Analyzes spending patterns, detects anomalies, identifies maverick spend,
and recognizes seasonal trends in expenditures.

NOW SCHEMA-AGNOSTIC: Works with any dataset structure via semantic column mapping.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
from sklearn.ensemble import IsolationForest

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
from core.gemma_summarizer import GemmaSummarizer

from core.premium_models import ConfigParameter

logger = logging.getLogger(__name__)


class SpendPatternEngine:
    """
    Schema-agnostic spend pattern analysis engine.
    
    Automatically detects spend, temporal, category, and supplier columns.
    """
    
    def __init__(self):
        self.name = "Spend Pattern Analysis Engine (Schema-Agnostic)"
        self.profiler = ColumnProfiler()
        self.mapper = ColumnMapper()
    
    def get_engine_info(self) -> Dict[str, str]:
        """Get engine display information."""
        return {
            "name": "spend_patterns",
            "display_name": "Spend Pattern Analysis",
            "icon": "💳",
            "task_type": "detection"
        }
    
    def get_config_schema(self) -> List[ConfigParameter]:
        """Get configuration schema for UI."""
        return [
            ConfigParameter(
                name="spend_column",
                type="select",
                default=None,
                range=[],
                description="Column containing spend values"
            ),
            ConfigParameter(
                name="date_column",
                type="select",
                default=None,
                range=[],
                description="Column containing dates"
            ),
            ConfigParameter(
                name="category_column",
                type="select",
                default=None,
                range=[],
                description="Column for spend categories"
            )
        ]
    
    def get_methodology_info(self) -> Dict[str, Any]:
        """Get methodology documentation."""
        return {
            "name": "Spend Pattern Analysis",
            "url": None,
            "steps": [
                {"step_number": 1, "title": "Spend Detection", "description": "Identify spending columns via semantic intelligence"},
                {"step_number": 2, "title": "Temporal Analysis", "description": "Analyze spending patterns over time"},
                {"step_number": 3, "title": "Anomaly Detection", "description": "Identify unusual spending using Isolation Forest"},
                {"step_number": 4, "title": "Category Analysis", "description": "Break down spending by category"}
            ],
            "limitations": ["Requires temporal data for trend analysis", "Anomaly detection needs sufficient data"],
            "assumptions": ["Spend data is accurate", "Time periods are consistent"]
        }
    
    def get_requirements(self) -> EngineRequirements:
        """
        Define semantic requirements for spend pattern analysis.
        
        Returns:
            EngineRequirements specifying needed columns
        """
        return EngineRequirements(
            required_semantics=[SemanticType.NUMERIC_CONTINUOUS],  # Spend amount required
            optional_semantics={
                'temporal': [SemanticType.TEMPORAL],  # For time-series analysis
                'category': [SemanticType.CATEGORICAL],  # For grouping
                'supplier': [SemanticType.CATEGORICAL]  # For supplier analysis
            },
            required_entities=[],
            preferred_domains=[],
            applicable_tasks=[],
            min_rows=10,  # Need sufficient transactions
            min_numeric_cols=1
        )
        
    def analyze(self, df: pd.DataFrame, config: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Run schema-agnostic spend pattern analysis.
        
        Args:
            df: DataFrame with any structure containing spending data
            config: Optional configuration with:
                - spend_column: Hint for spend column
                - date_column: Hint for date column
                - category_column: Hint for category column
                - supplier_column: Hint for supplier column
                - contract_column: Hint for contract status column
                - skip_profiling: Skip schema intelligence (default: False)
        
        Returns:
            Dict with pattern analysis and visualizations
        """
        config = config or {}
        
        results = {
            'summary': {},
            'patterns': {},
            'anomalies': [],
            'graphs': [],
            'insights': [],
            'column_mappings': {},
            'profiling_used': not config.get('skip_profiling', False)
        }
        
        try:
            # SCHEMA INTELLIGENCE: Profile and detect columns
            if not config.get('skip_profiling', False):
                logger.info("Profiling dataset for spend pattern analysis")
                profiles = self.profiler.profile_dataset(df)
                
                # Classify dataset
                classifier = DatasetClassifier()
                classification = classifier.classify(profiles, len(df))
                results['summary']['detected_domain'] = classification.domain.value
                
                # SMART COLUMN DETECTION
                spend_col = self._smart_detect(
                    df, profiles,
                    keywords=['spend', 'cost', 'amount', 'payment', 'expense'],
                    hint=config.get('spend_column')
                )
                
                date_col = self._smart_detect(
                    df, profiles,
                    keywords=['date', 'time', 'timestamp', 'period'],
                    hint=config.get('date_column'),
                    semantic_type=SemanticType.TEMPORAL
                )
                
                category_col = self._smart_detect(
                    df, profiles,
                    keywords=['category', 'type', 'department', 'class'],
                    hint=config.get('category_column'),
                    semantic_type=SemanticType.CATEGORICAL
                )
                
                supplier_col = self._smart_detect(
                    df, profiles,
                    keywords=['supplier', 'vendor', 'merchant', 'provider'],
                    hint=config.get('supplier_column'),
                    semantic_type=SemanticType.CATEGORICAL
                )
                
                # Store mappings
                if spend_col:
                    results['column_mappings']['spend'] = {'column': spend_col, 'confidence': 0.9}
                if date_col:
                    results['column_mappings']['temporal'] = {'column': date_col, 'confidence': 0.95}
                if category_col:
                    results['column_mappings']['category'] = {'column': category_col, 'confidence': 0.85}
                if supplier_col:
                    results['column_mappings']['supplier'] = {'column': supplier_col, 'confidence': 0.85}
            else:
                # Fallback
                spend_col = config.get('spend_column') or self._detect_column(
                    df, ['spend', 'cost', 'amount', 'payment'], None
                )
                date_col = config.get('date_column') or self._detect_column(
                    df, ['date', 'time', 'timestamp'], None
                )
                category_col = config.get('category_column') or self._detect_column(
                    df, ['category', 'type', 'department'], None
                )
                supplier_col = config.get('supplier_column') or self._detect_column(
                    df, ['supplier', 'vendor', 'merchant'], None
                )
                results['profiling_used'] = False
            
            if not spend_col:
                # Use Gemma fallback when required columns are missing
                return GemmaSummarizer.generate_fallback_summary(
                    df,
                    engine_name="spend_patterns",
                    error_reason="No spend/cost/amount column detected. Expected columns with 'spend', 'cost', 'amount', or 'expense' in name.",
                    config=config
                )
            
            # ANALYSIS PIPELINE (unchanged logic, schema-agnostic)
            total_spend = df[spend_col].sum()
            avg_spend = df[spend_col].mean()
            
            results['summary'] = {
                'total_spend': float(total_spend),
                'average_spend': float(avg_spend),
                'transaction_count': len(df),
                'unique_categories': df[category_col].nunique() if category_col else 0,
                'spend_column': spend_col
            }
            
            # Categorical spend breakdown
            if category_col:
                category_breakdown = self._category_breakdown(df, spend_col, category_col)
                results['patterns']['category_breakdown'] = category_breakdown
                results['graphs'].append(category_breakdown['graph'])
                
                # === ADD: category_totals for Sankey visualization ===
                try:
                    cat_totals = df.groupby(category_col)[spend_col].sum()
                    results['category_totals'] = {
                        str(cat): float(val) for cat, val in cat_totals.items()
                    }
                except Exception as e:
                    logger.warning(f"Could not build category_totals: {e}")
                    results['category_totals'] = {}
            
            # Temporal patterns
            if date_col:
                temporal = self._temporal_analysis(df, spend_col, date_col)
                results['patterns']['temporal'] = temporal
                results['graphs'].append(temporal['graph'])
                
                # Seasonality detection
                seasonality = self._detect_seasonality(df, spend_col, date_col)
                results['patterns']['seasonality'] = seasonality
                if seasonality['seasonal']:
                    results['graphs'].append(seasonality['graph'])
            
            # Anomaly detection
            anomalies = self._detect_spending_anomalies(df, spend_col, category_col)
            results['anomalies'] = anomalies['anomalies']
            results['graphs'].append(anomalies['graph'])
            
            # Maverick spend detection
            if supplier_col:
                maverick = self._detect_maverick_spend(df, spend_col, supplier_col, config.get('contract_column'))
                results['maverick_spend'] = maverick
                results['graphs'].append(maverick['graph'])
            
            # Supplier concentration
            if supplier_col:
                concentration = self._supplier_concentration(df, spend_col, supplier_col)
                results['supplier_concentration'] = concentration
                results['graphs'].append(concentration['graph'])
            
            # Generate insights
            results['insights'] = self._generate_insights(
                results, spend_col, date_col, category_col, supplier_col
            )
                
        except Exception as e:
            logger.error(f"Spend pattern analysis failed: {e}", exc_info=True)
            results['error'] = str(e)
            
        return results
    
    def _smart_detect(self, df: pd.DataFrame, profiles: Dict,
                     keywords: List[str], hint: Optional[str] = None,
                     semantic_type: Optional[SemanticType] = None) -> Optional[str]:
        """Smart column detection using profiles + keywords."""
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
        
        # Priority 2: Semantic type match
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
            col_lower = col.lower()
            if any(kw in col_lower for kw in keywords):
                return col
        return None
    
    def _category_breakdown(self, df: pd.DataFrame, spend_col: str, category_col: str) -> Dict:
        """Breakdown spend by category"""
        category_spend = df.groupby(category_col)[spend_col].agg(['sum', 'count', 'mean']).reset_index()
        category_spend.columns = ['category', 'total_spend', 'transaction_count', 'avg_transaction']
        category_spend = category_spend.sort_values('total_spend', ascending=False)
        
        total = category_spend['total_spend'].sum()
        category_spend['percentage'] = category_spend['total_spend'] / total * 100
        
        graph = {
            'type': 'pie_chart',
            'title': 'Spend by Category',
            'labels': category_spend['category'].head(10).tolist(),
            'values': category_spend['total_spend'].head(10).tolist(),
            'percentages': category_spend['percentage'].head(10).tolist()
        }
        
        return {
            'top_categories': category_spend.head(10).to_dict('records'),
            'graph': graph
        }
    
    def _temporal_analysis(self, df: pd.DataFrame, spend_col: str, date_col: str) -> Dict:
        """Analyze spending over time"""
        df_temp = df.copy()
        
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(df_temp[date_col]):
            df_temp[date_col] = pd.to_datetime(df_temp[date_col], errors='coerce')
        
        df_temp = df_temp.dropna(subset=[date_col])
        df_temp = df_temp.sort_values(date_col)
        
        # Aggregate by period (daily, weekly, or monthly depending on data span)
        date_range = (df_temp[date_col].max() - df_temp[date_col].min()).days
        
        if date_range > 365:
            freq = 'M'
            label = 'Monthly'
        elif date_range > 60:
            freq = 'W'
            label = 'Weekly'
        else:
            freq = 'D'
            label = 'Daily'
        
        df_temp['period'] = df_temp[date_col].dt.to_period(freq)
        period_spend = df_temp.groupby('period')[spend_col].sum()
        
        graph = {
            'type': 'time_series',
            'title': f'{label} Spending Trend',
            'x_data': period_spend.index.astype(str).tolist(),
            'y_data': period_spend.values.tolist(),
            'x_label': 'Period',
            'y_label': 'Total Spend ($)'
        }
        
        # Calculate trend
        if len(period_spend) > 2:
            from scipy.stats import linregress
            x = np.arange(len(period_spend))
            slope, intercept, r_value, p_value, std_err = linregress(x, period_spend.values)
            
            trend = 'increasing' if slope > 0 else 'decreasing'
            trend_strength = abs(r_value)
        else:
            trend = 'insufficient_data'
            trend_strength = 0
        
        return {
            'trend': trend,
            'trend_strength': float(trend_strength),
            'average_period_spend': float(period_spend.mean()),
            'graph': graph
        }
    
    def _detect_seasonality(self, df: pd.DataFrame, spend_col: str, date_col: str) -> Dict:
        """Detect seasonal patterns"""
        df_temp = df.copy()
        
        if not pd.api.types.is_datetime64_any_dtype(df_temp[date_col]):
            df_temp[date_col] = pd.to_datetime(df_temp[date_col], errors='coerce')
        
        df_temp = df_temp.dropna(subset=[date_col])
        
        # Monthly aggregation
        df_temp['month'] = df_temp[date_col].dt.month
        monthly_spend = df_temp.groupby('month')[spend_col].mean()
        
        # Simple seasonality test: coefficient of variation
        cv = monthly_spend.std() / monthly_spend.mean() if monthly_spend.mean() > 0 else 0
        seasonal = cv > 0.3  # Threshold for seasonality
        
        if seasonal:
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            graph = {
                'type': 'bar_chart',
                'title': 'Seasonal Spend Pattern',
                'x_data': [month_names[m-1] for m in monthly_spend.index],
                'y_data': monthly_spend.values.tolist(),
                'x_label': 'Month',
                'y_label': 'Average Spend ($)'
            }
        else:
            graph = None
        
        return {
            'seasonal': seasonal,
            'coefficient_of_variation': float(cv),
            'peak_month': int(monthly_spend.idxmax()) if len(monthly_spend) > 0 else None,
            'low_month': int(monthly_spend.idxmin()) if len(monthly_spend) > 0 else None,
            'graph': graph
        }
    
    def _detect_spending_anomalies(self, df: pd.DataFrame, spend_col: str, category_col: Optional[str]) -> Dict:
        """Detect anomalous spending transactions"""
        # Use Isolation Forest for anomaly detection
        X = df[[spend_col]].values
        
        iso_forest = IsolationForest(contamination=0.05, random_state=42)
        predictions = iso_forest.fit_predict(X)
        
        # -1 indicates anomaly
        anomaly_indices = np.where(predictions == -1)[0]
        anomalies = df.iloc[anomaly_indices].copy()
        
        anomaly_list = []
        for idx, row in anomalies.head(20).iterrows():
            anomaly_list.append({
                'amount': float(row[spend_col]),
                'category': row[category_col] if category_col and category_col in row else 'Unknown',
                'deviation': float((row[spend_col] - df[spend_col].mean()) / df[spend_col].std())
            })
        
        # Visualization
        graph = {
            'type': 'scatter',
            'title': 'Spending Anomaly Detection',
            'x_data': list(range(len(df))),
            'y_data': df[spend_col].values.tolist(),
            'anomalies': anomaly_indices.tolist(),
            'x_label': 'Transaction Index',
            'y_label': 'Spend Amount ($)'
        }
        
        return {
            'anomaly_count': len(anomalies),
            'anomaly_percentage': float(len(anomalies) / len(df) * 100),
            'total_anomaly_spend': float(anomalies[spend_col].sum()),
            'anomalies': anomaly_list,
            'graph': graph
        }
    
    def _detect_maverick_spend(self, df: pd.DataFrame, spend_col: str, supplier_col: str, contract_col: Optional[str]) -> Dict:
        """Detect off-contract (maverick) spending"""
        # If contract column exists, use it
        if contract_col and contract_col in df.columns:
            maverick = df[df[contract_col].isin([False, 0, 'No', 'no', 'N', 'n'])]
        else:
            # Heuristic: Identify suppliers with low transaction frequency (likely non-contracted)
            supplier_counts = df[supplier_col].value_counts()
            low_freq_suppliers = supplier_counts[supplier_counts <= 2].index
            maverick = df[df[supplier_col].isin(low_freq_suppliers)]
        
        maverick_spend = maverick[spend_col].sum()
        total_spend = df[spend_col].sum()
        maverick_percentage = maverick_spend / total_spend * 100 if total_spend > 0 else 0
        
        # Top maverick suppliers
        top_maverick = maverick.groupby(supplier_col)[spend_col].sum().sort_values(ascending=False).head(10)
        
        graph = {
            'type': 'bar_chart',
            'title': 'Top Maverick Spend (Off-Contract)',
            'x_data': top_maverick.index.tolist(),
            'y_data': top_maverick.values.tolist(),
            'x_label': 'Supplier',
            'y_label': 'Spend ($)'
        }
       
        return {
            'maverick_spend': float(maverick_spend),
            'maverick_percentage': float(maverick_percentage),
            'maverick_transaction_count': len(maverick),
            'top_maverick_suppliers': top_maverick.to_dict(),
            'graph': graph
        }
    
    def _supplier_concentration(self, df: pd.DataFrame, spend_col: str, supplier_col: str) -> Dict:
        """Analyze supplier concentration risk"""
        supplier_spend = df.groupby(supplier_col)[spend_col].sum().sort_values(ascending=False)
        total_spend = supplier_spend.sum()
        
        # Calculate concentration metrics
        top_5_spend = supplier_spend.head(5).sum()
        top_5_percentage = top_5_spend / total_spend * 100 if total_spend > 0 else 0
        
        # HHI (Herfindahl-Hirschman Index) for concentration
        market_shares = (supplier_spend / total_spend * 100) ** 2
        hhi = market_shares.sum()
        
        # Risk classification
        if hhi > 2500:
            risk_level = 'High'
        elif hhi > 1500:
            risk_level = 'Medium'
        else:
            risk_level = 'Low'
        
        graph = {
            'type': 'bar_chart',
            'title': 'Supplier Concentration (Top 10)',
            'x_data': supplier_spend.head(10).index.tolist(),
            'y_data': supplier_spend.head(10).values.tolist(),
            'x_label': 'Supplier',
            'y_label': 'Total Spend ($)'
        }
        
        return {
            'total_suppliers': len(supplier_spend),
            'top_5_percentage': float(top_5_percentage),
            'hhi': float(hhi),
            'concentration_risk': risk_level,
            'graph': graph
        }
    
    def _generate_insights(self, results: Dict, spend_col: str = None, date_col: str = None, category_col: str = None, supplier_col: str = None) -> List[str]:
        """Generate insights"""
        insights = []
        
        if 'summary' in results:
            summary = results['summary']
            insights.append(
                f"💵 Total Spend: ${summary['total_spend']:,.2f} across "
                f"{summary['transaction_count']} transactions"
            )
        
        if 'patterns' in results and 'temporal' in results['patterns']:
            temporal = results['patterns']['temporal']
            trend_emoji = '📈' if temporal['trend'] == 'increasing' else '📉'
            insights.append(
                f"{trend_emoji} Spending trend: {temporal['trend']} "
                f"(strength: {temporal['trend_strength']:.2f})"
            )
        
        if 'anomalies' in results and results['anomalies']:
            insights.append(
                f"⚠️ Detected {len(results['anomalies'])} anomalous transactions - "
                f"recommend review for errors or fraud"
            )
        
        if 'maverick_spend' in results:
            mav = results['maverick_spend']
            if mav['maverick_percentage'] > 10:
                insights.append(
                    f"🚨 Maverick spend at {mav['maverick_percentage']:.1f}% "
                    f"(${mav['maverick_spend']:,.2f}) - enforce contract compliance"
                )
        
        if 'supplier_concentration' in results:
            conc = results['supplier_concentration']
            insights.append(
                f"🎯 Supplier concentration risk: {conc['concentration_risk']} "
                f"(Top 5 suppliers: {conc['top_5_percentage']:.1f}% of spend)"
            )
        
        return insights


__all__ = ['SpendPatternEngine']
