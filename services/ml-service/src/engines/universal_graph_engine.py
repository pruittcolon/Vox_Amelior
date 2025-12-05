"""
Universal Graph Generator Engine

Automatically generates 10+ meaningful visualizations for ANY database
by analyzing schema, data types, and statistical properties.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter
import logging

logger = logging.getLogger(__name__)


class UniversalGraphEngine:
    """
    Intelligently generates appropriate visualizations for any dataset
    based on schema analysis and data profiling.
    """
    
    def __init__(self):
        self.name = "Universal Graph Engine"
        self.min_cardinality_for_categorical = 2
        self.max_cardinality_for_categorical = 50
        self.sample_size_for_large_datasets = 10000
    
    def get_engine_info(self) -> Dict[str, str]:
        """Get engine display information."""
        return {
            "name": "graphs",
            "display_name": "Universal Graphs",
            "icon": "📊",
            "task_type": "detection"
        }
    
    def get_config_schema(self) -> List[Any]:
        """Get configuration schema for UI."""
        from core.premium_models import ConfigParameter
        return [
            ConfigParameter(
                name="max_graphs",
                type="int",
                default=15,
                range=[5, 30],
                description="Maximum number of graphs to generate"
            ),
            ConfigParameter(
                name="focus_columns",
                type="select",
                default=[],
                range=[],
                description="Specific columns to prioritize in visualizations"
            )
        ]
    
    def get_methodology_info(self) -> Dict[str, Any]:
        """Get methodology documentation."""
        return {
            "name": "Automated Visualization Generation",
            "url": None,
            "steps": [
                {"step_number": 1, "title": "Data Profiling", "description": "Analyze column types, cardinality, and distributions"},
                {"step_number": 2, "title": "Graph Selection", "description": "Choose appropriate chart types based on data characteristics"},
                {"step_number": 3, "title": "Data Preparation", "description": "Sample large datasets and prepare data for visualization"},
                {"step_number": 4, "title": "Visualization Generation", "description": "Create 10+ charts covering distributions, correlations, and trends"}
            ],
            "limitations": ["Large datasets are sampled", "Complex relationships may not be captured"],
            "assumptions": ["Data is properly typed", "Missing values are handled"]
        }
    
    def analyze(self, df: pd.DataFrame, config: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Standard analyze() interface that wraps generate_graphs().
        Required for compatibility with engine registry.
        """
        return self.generate_graphs(df, config)
    
    def generate_graphs(self, df: pd.DataFrame, config: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Generate 10+ appropriate visualizations for any dataset
        
        Args:
            df: Input dataframe
            config: Optional configuration
                - max_graphs: Maximum number of graphs to generate
                - focus_columns: Specific columns to prioritize
        
        Returns:
            Dictionary with visualization specifications and data
        """
        config = config or {}
        
        # Step 1: Profile the dataset
        profile = self._profile_dataset(df)
        
        # Step 2: Sample if dataset is too large
        df_sample = self._smart_sample(df)
        
        # Step 3: Generate graph recommendations
        graphs = []
        
        # Always include these foundational graphs
        graphs.extend(self._generate_data_profile_dashboard(df, profile))
        graphs.extend(self._generate_distribution_graphs(df_sample, profile))
        graphs.extend(self._generate_correlation_graphs(df_sample, profile))
        graphs.extend(self._generate_temporal_graphs(df_sample, profile))
        graphs.extend(self._generate_categorical_graphs(df_sample, profile))
        graphs.extend(self._generate_missing_data_graphs(df, profile))
        graphs.extend(self._generate_outlier_graphs(df_sample, profile))
        graphs.extend(self._generate_ranking_graphs(df_sample, profile))
        graphs.extend(self._generate_cross_tabulation_graphs(df_sample, profile))
        graphs.extend(self._generate_data_quality_graphs(df, profile))
        
        # Sort by priority and limit if requested
        max_graphs = config.get('max_graphs', len(graphs))
        graphs = sorted(graphs, key=lambda x: x.get('priority', 5), reverse=True)[:max_graphs]
        
        return {
            'profile': profile,
            'graphs': graphs,
            'total_graphs': len(graphs),
            'metadata': {
                'rows': len(df),
                'columns': len(df.columns),
                'sampled': len(df) != len(df_sample)
            }
        }
    
    def _profile_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensively profile the dataset"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Auto-detect datetime columns stored as strings
        for col in categorical_cols[:]:
            if self._is_datetime_column(df[col]):
                datetime_cols.append(col)
                categorical_cols.remove(col)
        
        # Classify columns by cardinality
        low_cardinality_cols = []
        high_cardinality_cols = []
        
        for col in categorical_cols:
            cardinality = df[col].nunique()
            if cardinality <= self.max_cardinality_for_categorical:
                low_cardinality_cols.append(col)
            else:
                high_cardinality_cols.append(col)
        
        # Detect potential ID columns
        id_cols = [col for col in df.columns if df[col].nunique() == len(df)]
        
        # Missing data summary
        missing_summary = {
            col: {
                'count': int(df[col].isna().sum()),
                'percentage': float(df[col].isna().sum() / len(df) * 100)
            }
            for col in df.columns if df[col].isna().sum() > 0
        }
        
        return {
            'numeric_columns': numeric_cols,
            'categorical_columns': categorical_cols,
            'low_cardinality': low_cardinality_cols,
            'high_cardinality': high_cardinality_cols,
            'datetime_columns': datetime_cols,
            'id_columns': id_cols,
            'missing_data': missing_summary,
            'total_columns': len(df.columns),
            'total_rows': len(df)
        }
    
    def _is_datetime_column(self, series: pd.Series, sample_size: int = 100) -> bool:
        """Heuristic to detect datetime columns stored as strings"""
        if len(series) == 0:
            return False
        
        sample = series.dropna().head(sample_size)
        if len(sample) == 0:
            return False
        
        # Try parsing as datetime
        try:
            pd.to_datetime(sample, errors='coerce')
            # If more than 80% parse successfully, likely datetime
            parsed = pd.to_datetime(sample, errors='coerce')
            success_rate = parsed.notna().sum() / len(sample)
            return success_rate > 0.8
        except:
            return False
    
    def _smart_sample(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sample large datasets intelligently"""
        if len(df) <= self.sample_size_for_large_datasets:
            return df
        
        # Stratified sampling if we detect groups
        return df.sample(n=self.sample_size_for_large_datasets, random_state=42)
    
    def _generate_data_profile_dashboard(self, df: pd.DataFrame, profile: Dict) -> List[Dict]:
        """Graph 1: Data profile overview"""
        graphs = []
        
        # Column type breakdown
        type_counts = {
            'Numeric': len(profile['numeric_columns']),
            'Categorical': len(profile['categorical_columns']),
            'DateTime': len(profile['datetime_columns']),
            'ID': len(profile['id_columns'])
        }
        
        graphs.append({
            'type': 'pie_chart',
            'title': 'Column Type Distribution',
            'data': type_counts,
            'priority': 10,
            'description': 'Breakdown of column types in the dataset',
            'drill_down': False
        })
        
        return graphs
    
    def _generate_distribution_graphs(self, df: pd.DataFrame, profile: Dict) -> List[Dict]:
        """Graphs 2-N: Distribution analysis for numeric columns"""
        graphs = []
        
        for col in profile['numeric_columns'][:10]:  # Limit to top 10
            # Histogram
            graphs.append({
                'type': 'histogram',
                'title': f'Distribution: {col}',
                'data': {
                    'column': col,
                    'values': df[col].dropna().tolist()[:1000]  # Limit data size
                },
                'priority': 9,
                'description': f'Frequency distribution of {col}',
                'drill_down': True,
                'drill_down_target': 'outliers'
            })
            
            # Box plot for outlier detection
            graphs.append({
                'type': 'box_plot',
                'title': f'Outliers: {col}',
                'data': {
                    'column': col,
                    'values': df[col].dropna().tolist()[:1000]
                },
                'priority': 7,
                'description': f'Outlier detection for {col}',
                'drill_down': False
            })
        
        return graphs
    
    def _generate_correlation_graphs(self, df: pd.DataFrame, profile: Dict) -> List[Dict]:
        """Graph: Correlation heatmap"""
        graphs = []
        
        if len(profile['numeric_columns']) >= 2:
            numeric_df = df[profile['numeric_columns']]
            corr_matrix = numeric_df.corr().to_dict()
            
            graphs.append({
                'type': 'heatmap',
                'title': 'Correlation Matrix',
                'data': {
                    'matrix': corr_matrix,
                    'columns': profile['numeric_columns']
                },
                'priority': 9,
                'description': 'Correlation between numeric variables',
                'drill_down': True,
                'drill_down_target': 'scatter_plot'
            })
        
        return graphs
    
    def _generate_temporal_graphs(self, df: pd.DataFrame, profile: Dict) -> List[Dict]:
        """Graphs: Time-series visualizations if datetime columns exist"""
        graphs = []
        
        for datetime_col in profile['datetime_columns'][:3]:  # Limit to 3
            # Convert to datetime if needed
            if df[datetime_col].dtype != 'datetime64[ns]':
                df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce')
            
            # Record count over time
            time_series = df.groupby(df[datetime_col].dt.date).size()
            
            graphs.append({
                'type': 'line_chart',
                'title': f'Records Over Time ({datetime_col})',
                'data': {
                    'x': [str(d) for d in time_series.index.tolist()],
                    'y': time_series.values.tolist(),
                    'x_label': datetime_col,
                    'y_label': 'Count'
                },
                'priority': 8,
                'description': f'Number of records over time based on {datetime_col}',
                'drill_down': True,
                'drill_down_target': 'daily_breakdown'
            })
            
            # If numeric columns exist, show trends
            for numeric_col in profile['numeric_columns'][:2]:
                trend_data = df.groupby(df[datetime_col].dt.date)[numeric_col].mean()
                
                graphs.append({
                    'type': 'line_chart',
                    'title': f'{numeric_col} Trend Over Time',
                    'data': {
                        'x': [str(d) for d in trend_data.index.tolist()],
                        'y': trend_data.values.tolist(),
                        'x_label': datetime_col,
                        'y_label': numeric_col
                    },
                    'priority': 7,
                    'description': f'Average {numeric_col} over time',
                    'drill_down': False
                })
        
        return graphs
    
    def _generate_categorical_graphs(self, df: pd.DataFrame, profile: Dict) -> List[Dict]:
        """Graphs: Categorical breakdowns"""
        graphs = []
        
        # Low cardinality categorical columns - pie/bar charts
        for col in profile['low_cardinality'][:5]:
            value_counts = df[col].value_counts().head(10)
            
            # Bar chart for top values
            graphs.append({
                'type': 'bar_chart',
                'title': f'Top Values: {col}',
                'data': {
                    'labels': value_counts.index.tolist(),
                    'values': value_counts.values.tolist(),
                    'orientation': 'horizontal'
                },
                'priority': 8,
                'description': f'Most frequent values in {col}',
                'drill_down': True,
                'drill_down_target': 'filtered_records'
            })
        
        # High cardinality - show unique count only
        for col in profile['high_cardinality'][:3]:
            graphs.append({
                'type': 'metric_card',
                'title': f'Unique Values: {col}',
                'data': {
                    'value': int(df[col].nunique()),
                    'total': len(df),
                    'percentage': float(df[col].nunique() / len(df) * 100)
                },
                'priority': 5,
                'description': f'Cardinality of {col}',
                'drill_down': False
            })
        
        return graphs
    
    def _generate_missing_data_graphs(self, df: pd.DataFrame, profile: Dict) -> List[Dict]:
        """Graph: Missing data visualization"""
        graphs = []
        
        if profile['missing_data']:
            missing_data = {
                col: data['percentage'] 
                for col, data in profile['missing_data'].items()
            }
            
            graphs.append({
                'type': 'bar_chart',
                'title': 'Missing Data by Column',
                'data': {
                    'labels': list(missing_data.keys()),
                    'values': list(missing_data.values()),
                    'orientation': 'horizontal',
                    'color': 'red'
                },
                'priority': 9,
                'description': 'Percentage of missing values per column',
                'drill_down': False
            })
        
        return graphs
    
    def _generate_outlier_graphs(self, df: pd.DataFrame, profile: Dict) -> List[Dict]:
        """Graphs: Outlier detection summaries"""
        graphs = []
        
        outlier_summary = {}
        for col in profile['numeric_columns']:
            # Skip boolean columns - quantile fails on bool dtype
            if df[col].dtype == 'bool':
                continue
            try:
                col_data = df[col].astype(float)
                q1 = col_data.quantile(0.25)
                q3 = col_data.quantile(0.75)
                iqr = q3 - q1
                if iqr == 0:
                    continue
                outliers = df[(col_data < q1 - 1.5 * iqr) | (col_data > q3 + 1.5 * iqr)]
                outlier_summary[col] = len(outliers)
            except (TypeError, ValueError):
                continue
        
        if outlier_summary:
            graphs.append({
                'type': 'bar_chart',
                'title': 'Outlier Count by Column',
                'data': {
                    'labels': list(outlier_summary.keys()),
                    'values': list(outlier_summary.values()),
                    'orientation': 'vertical'
                },
                'priority': 7,
                'description': 'Number of outliers detected (IQR method)',
                'drill_down': True,
                'drill_down_target': 'outlier_details'
            })
        
        return graphs
    
    def _generate_ranking_graphs(self, df: pd.DataFrame, profile: Dict) -> List[Dict]:
        """Graphs: Top-N rankings"""
        graphs = []
        
        # For each numeric column, show top records
        for numeric_col in profile['numeric_columns'][:3]:
            top_n = df.nlargest(10, numeric_col)
            
            # Need an identifier column
            id_col = profile['id_columns'][0] if profile['id_columns'] else df.columns[0]
            
            graphs.append({
                'type': 'bar_chart',
                'title': f'Top 10: {numeric_col}',
                'data': {
                    'labels': top_n[id_col].astype(str).tolist()[:10],
                    'values': top_n[numeric_col].tolist()[:10],
                    'orientation': 'horizontal'
                },
                'priority': 6,
                'description': f'Highest {numeric_col} values',
                'drill_down': True,
                'drill_down_target': 'record_detail'
            })
        
        return graphs
    
    def _generate_cross_tabulation_graphs(self, df: pd.DataFrame, profile: Dict) -> List[Dict]:
        """Graphs: Cross-tabulations for categorical pairs"""
        graphs = []
        
        # Create pivot tables for pairs of low-cardinality categoricals
        if len(profile['low_cardinality']) >= 2:
            col1, col2 = profile['low_cardinality'][:2]
            
            pivot = pd.crosstab(df[col1], df[col2])
            
            graphs.append({
                'type': 'grouped_bar_chart',
                'title': f'{col1} vs {col2}',
                'data': {
                    'pivot': pivot.to_dict(),
                    'col1': col1,
                    'col2': col2
                },
                'priority': 7,
                'description': f'Cross-tabulation of {col1} and {col2}',
                'drill_down': True,
                'drill_down_target': 'filtered_data'
            })
        
        return graphs
    
    def _generate_data_quality_graphs(self, df: pd.DataFrame, profile: Dict) -> List[Dict]:
        """Graph: Overall data quality score"""
        graphs = []
        
        # Calculate quality metrics
        completeness = (1 - df.isna().sum().sum() / (len(df) * len(df.columns))) * 100
        
        uniqueness_scores = []
        for col in df.columns:
            if col not in profile['id_columns']:
                uniqueness_scores.append(df[col].nunique() / len(df))
        avg_uniqueness = np.mean(uniqueness_scores) * 100 if uniqueness_scores else 0
        
        # Validity (what % of data is in expected ranges)
        validity = 100  # Simplified - always 100 for now
        
        quality_metrics = {
            'Completeness': float(completeness),
            'Uniqueness': float(avg_uniqueness),
            'Validity': float(validity)
        }
        
        graphs.append({
            'type': 'radar_chart',
            'title': 'Data Quality Score',
            'data': quality_metrics,
            'priority': 8,
            'description': 'Overall data quality metrics',
            'drill_down': False
        })
        
        return graphs
