"""
Pricing Strategy Engine - Schema-Agnostic Version

Optimizes pricing based on cost, demand, and competitive analysis.

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
from scipy import stats

from core.premium_models import ConfigParameter

logger = logging.getLogger(__name__)


class PricingStrategyEngine:
    """
    Schema-agnostic pricing optimization engine.
    
    Automatically detects cost, price, and demand columns.
    """
    
    def __init__(self):
        self.name = "Pricing Strategy Engine (Schema-Agnostic)"
        self.profiler = ColumnProfiler()
        self.mapper = ColumnMapper()
    
    def get_engine_info(self) -> Dict[str, str]:
        return {
            "name": "pricing_strategy",
            "display_name": "Pricing Strategy",
            "icon": "🎯",
            "task_type": "prediction"
        }
    
    def get_config_schema(self) -> List[ConfigParameter]:
        return [
            ConfigParameter(name="price_column", type="select", default=None, range=[], description="Column containing prices"),
            ConfigParameter(name="cost_column", type="select", default=None, range=[], description="Column containing costs"),
            ConfigParameter(name="demand_column", type="select", default=None, range=[], description="Column containing demand")
        ]
    
    def get_methodology_info(self) -> Dict[str, Any]:
        return {
            "name": "Pricing Strategy Analysis",
            "url": None,
            "steps": [
                {"step_number": 1, "title": "Elasticity Analysis", "description": "Estimate price elasticity of demand"},
                {"step_number": 2, "title": "Optimization", "description": "Find revenue-maximizing price points"},
                {"step_number": 3, "title": "Scenario Testing", "description": "Test pricing scenarios"}
            ],
            "limitations": ["Assumes rational consumer behavior"],
            "assumptions": ["Price-demand relationship is stable"]
        }
    
    def get_requirements(self) -> EngineRequirements:
        return EngineRequirements(
            required_semantics=[SemanticType.NUMERIC_CONTINUOUS],
            optional_semantics={'cost': [SemanticType.NUMERIC_CONTINUOUS], 'price': [SemanticType.NUMERIC_CONTINUOUS], 'demand': [SemanticType.NUMERIC_CONTINUOUS]},
            required_entities=[],
            preferred_domains=[],
            applicable_tasks=[],
            min_rows=10,
            min_numeric_cols=2
        )
        
    def analyze(self, df: pd.DataFrame, config: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Run schema-agnostic pricing analysis.
        
        Args:
            df: DataFrame with sales history
            config: Optional configuration
        
        Returns:
            Dict with pricing recommendations
        """
        config = config or {}
        
        results = {
            'summary': {},
            'elasticity': [],
            'recommendations': [],
            'graphs': [],
            'insights': [],
            'column_mappings': {},
            'profiling_used': not config.get('skip_profiling', False)
        }
        
        try:
            # SCHEMA INTELLIGENCE
            if not config.get('skip_profiling', False):
                logger.info("Profiling dataset for pricing analysis")
                profiles = self.profiler.profile_dataset(df)
                
                classifier = DatasetClassifier()
                classification = classifier.classify(profiles, len(df))
                results['summary']['detected_domain'] = classification.domain.value
                
                price_col = self._smart_detect(
                    df, profiles,
                    keywords=['price', 'unit_price', 'msrp', 'cost'],
                    hint=config.get('price_column')
                )
                
                qty_col = self._smart_detect(
                    df, profiles,
                    keywords=['quantity', 'sales', 'volume', 'units', 'demand'],
                    hint=config.get('quantity_column')
                )
                
                prod_col = self._smart_detect(
                    df, profiles,
                    keywords=['product', 'sku', 'item'],
                    hint=config.get('product_column'),
                    semantic_type=SemanticType.CATEGORICAL
                )
                
                comp_col = self._smart_detect(
                    df, profiles,
                    keywords=['competitor', 'market_price'],
                    hint=config.get('competitor_price_column')
                )
                
                if price_col:
                    results['column_mappings']['price'] = {'column': price_col, 'confidence': 0.9}
                if qty_col:
                    results['column_mappings']['quantity'] = {'column': qty_col, 'confidence': 0.9}
                if prod_col:
                    results['column_mappings']['product'] = {'column': prod_col, 'confidence': 0.85}
                if comp_col:
                    results['column_mappings']['competitor'] = {'column': comp_col, 'confidence': 0.8}
            else:
                price_col = config.get('price_column') or self._detect_column(df, ['price', 'unit_price'], None)
                qty_col = config.get('quantity_column') or self._detect_column(df, ['quantity', 'sales'], None)
                prod_col = config.get('product_column') or self._detect_column(df, ['product', 'sku'], None)
                comp_col = config.get('competitor_price_column') or self._detect_column(df, ['competitor'], None)
                results['profiling_used'] = False
            
            if not (price_col and qty_col):
                # Use Gemma fallback when required columns are missing
                missing = []
                if not price_col:
                    missing.append('price')
                if not qty_col:
                    missing.append('quantity/demand')
                return GemmaSummarizer.generate_fallback_summary(
                    df,
                    engine_name="pricing_strategy",
                    error_reason=f"Could not detect {' and '.join(missing)} columns. Pricing analysis requires price and quantity/sales columns.",
                    config=config
                )
            
            results['summary']['price_column'] = price_col
            results['summary']['quantity_column'] = qty_col
            
            # ANALYSIS PIPELINE
            elasticities = []
            
            if prod_col:
                products = df[prod_col].unique()
                top_products = df.groupby(prod_col)[qty_col].sum().nlargest(10).index
                
                for prod in top_products:
                    prod_df = df[df[prod_col] == prod].copy()
                    if len(prod_df) > 5:
                        e = self._calculate_elasticity(prod_df, price_col, qty_col)
                        if e:
                            e['product'] = prod
                            elasticities.append(e)
            else:
                e = self._calculate_elasticity(df, price_col, qty_col)
                if e:
                    e['product'] = 'All Products'
                    elasticities.append(e)
            
            results['elasticity'] = elasticities
            
            if comp_col:
                comp_analysis = self._competitor_analysis(df, price_col, comp_col)
                results['competitor'] = comp_analysis
                results['graphs'].append(comp_analysis['graph'])
            
            for e in elasticities:
                rec = self._generate_recommendation(e)
                results['recommendations'].append(rec)
            
            if elasticities:
                elast_chart = self._elasticity_chart(elasticities)
                results['graphs'].append(elast_chart['graph'])
            
            results['insights'] = self._generate_insights(results, price_col, qty_col)
            
        except Exception as e:
            logger.error(f"Pricing strategy analysis failed: {e}", exc_info=True)
            results['error'] = str(e)
                
        except Exception as e:
            logger.error(f"Pricing strategy analysis failed: {e}")
            results['error'] = str(e)
            
        return results
    
    def _smart_detect(self, df: pd.DataFrame, profiles: Dict,
                     keywords: List[str], hint: Optional[str] = None,
                     semantic_type: Optional[SemanticType] = None) -> Optional[str]:
        """Smart column detection."""
        if hint and hint in df.columns:
            return hint
        
        # First pass: look for keyword matches with semantic type constraint
        for col in df.columns:
            if any(kw in col.lower() for kw in keywords):
                prof = profiles.get(col)
                if prof:
                    if semantic_type:
                        if prof.semantic_type == semantic_type:
                            return col
                    elif prof.semantic_type in [SemanticType.NUMERIC_CONTINUOUS, SemanticType.NUMERIC_DISCRETE]:
                        return col
                    elif prof.semantic_type == SemanticType.IDENTIFIER:
                        # Check if it's actually a numeric column
                        if pd.api.types.is_numeric_dtype(df[col]):
                            return col
        
        # Second pass: just keyword match without semantic type
        for col in df.columns:
            if any(kw in col.lower() for kw in keywords):
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
    
    def _calculate_elasticity(self, df: pd.DataFrame, price_col: str, qty_col: str) -> Optional[Dict]:
        """Calculate price elasticity using log-log regression"""
        try:
            # Filter positive values for log
            valid_data = df[(df[price_col] > 0) & (df[qty_col] > 0)]
            
            if len(valid_data) < 5:
                return None
                
            x = np.log(valid_data[price_col])
            y = np.log(valid_data[qty_col])
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            return {
                'elasticity': float(slope),
                'r_squared': float(r_value**2),
                'avg_price': float(valid_data[price_col].mean()),
                'avg_quantity': float(valid_data[qty_col].mean())
            }
        except:
            return None
            
    def _competitor_analysis(self, df: pd.DataFrame, price_col: str, comp_col: str) -> Dict:
        """Compare against competitor pricing"""
        df['price_diff'] = df[price_col] - df[comp_col]
        df['price_diff_pct'] = (df['price_diff'] / df[comp_col] * 100)
        
        avg_diff = df['price_diff_pct'].mean()
        
        graph = {
            'type': 'histogram',
            'title': 'Price vs Competitor Distribution',
            'x_data': df['price_diff_pct'].tolist(),
            'x_label': 'Price Difference (%)',
            'bins': 20
        }
        
        return {
            'avg_price_difference_pct': float(avg_diff),
            'cheaper_than_competitor': float((df['price_diff'] < 0).mean() * 100),
            'graph': graph
        }
    
    def _generate_recommendation(self, elasticity_data: Dict) -> Dict:
        """Generate pricing recommendation based on elasticity"""
        e = elasticity_data['elasticity']
        prod = elasticity_data['product']
        
        if e < -1:
            # Elastic: Price decrease increases revenue
            action = "Lower Price"
            reason = f"Demand is elastic ({e:.2f}). Lowering price may increase total revenue."
        elif e > -1 and e < 0:
            # Inelastic: Price increase increases revenue
            action = "Raise Price"
            reason = f"Demand is inelastic ({e:.2f}). Raising price may increase total revenue."
        else:
            # Positive elasticity (Veblen good?) or noise
            action = "Maintain Price"
            reason = f"Unusual elasticity ({e:.2f}). Further investigation needed."
            
        return {
            'product': prod,
            'elasticity': e,
            'action': action,
            'reason': reason
        }
    
    def _elasticity_chart(self, elasticities: List[Dict]) -> Dict:
        """Chart elasticities by product"""
        graph = {
            'type': 'bar_chart',
            'title': 'Price Elasticity by Product',
            'x_data': [e['product'] for e in elasticities],
            'y_data': [e['elasticity'] for e in elasticities],
            'x_label': 'Product',
            'y_label': 'Elasticity',
            'threshold': -1 # Unit elasticity line
        }
        
        return {'graph': graph}
    
    def _generate_insights(self, results: Dict, price_col: str = None, qty_col: str = None) -> List[str]:
        """Generate insights mentioning detected columns."""
        insights = []
        
        if price_col and qty_col:
            insights.append(f"📊 Pricing analysis using '{price_col}' and '{qty_col}'")
        
        if results['recommendations']:
            rec = results['recommendations'][0]
            insights.append(
                f"🏷️ Recommendation for {rec['product']}: {rec['action']} "
                f"(Elasticity: {rec['elasticity']:.2f})"
            )
            
        if 'competitor' in results:
            comp = results['competitor']
            status = "premium" if comp['avg_price_difference_pct'] > 0 else "discount"
            insights.append(
                f"🏢 Market Position: {abs(comp['avg_price_difference_pct']):.1f}% {status} "
                f"vs competitors"
            )
            
        return insights


__all__ = ['PricingStrategyEngine']
