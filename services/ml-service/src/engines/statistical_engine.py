"""
Statistical Analysis Engine

Provides comprehensive statistical analysis including:
- Descriptive statistics
- Variance analysis (ANOVA)
- Correlation analysis
- Regression analysis
- Hypothesis testing
- Distribution analysis
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from scipy import stats
from scipy.stats import shapiro, anderson, pearsonr, spearmanr, kendalltau
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import logging

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


class StatisticalEngine:
    """Schema-agnostic engine for comprehensive statistical analysis."""
    
    def __init__(self):
        self.name = "Statistical Analysis Engine (Schema-Agnostic)"
        self.profiler = ColumnProfiler()
        self.mapper = ColumnMapper()
        self.results_cache = {}
    
    def get_engine_info(self) -> Dict[str, str]:
        """Get engine display information."""
        return {
            "name": "statistical",
            "display_name": "Statistical Analysis",
            "icon": "📊",
            "task_type": "detection"
        }
    
    def get_config_schema(self) -> List[ConfigParameter]:
        """Get configuration schema for UI."""
        return [
            ConfigParameter(
                name="analysis_types",
                type="select",
                default=["all"],
                range=["all", "descriptive", "correlation", "anova", "regression", "distribution", "hypothesis"],
                description="Types of statistical analysis to run"
            ),
            ConfigParameter(
                name="confidence_level",
                type="float",
                default=0.95,
                range=[0.9, 0.99],
                description="Confidence level for statistical tests"
            )
        ]
    
    def get_methodology_info(self) -> Dict[str, Any]:
        """Get methodology documentation."""
        return {
            "name": "Statistical Analysis",
            "url": "https://en.wikipedia.org/wiki/Statistics",
            "steps": [
                {"step_number": 1, "title": "Data Profiling", "description": "Profile columns to identify numeric and categorical variables"},
                {"step_number": 2, "title": "Descriptive Statistics", "description": "Calculate mean, median, std, skewness, kurtosis"},
                {"step_number": 3, "title": "Correlation Analysis", "description": "Compute Pearson, Spearman, and Kendall correlations"},
                {"step_number": 4, "title": "Hypothesis Testing", "description": "Run normality tests, ANOVA, and regression analysis"}
            ],
            "limitations": ["Assumes data is representative", "Correlation does not imply causation"],
            "assumptions": ["Numeric columns are continuous or discrete measurements"]
        }
    
    def get_requirements(self) -> EngineRequirements:
        return EngineRequirements(
            required_semantics=[SemanticType.NUMERIC_CONTINUOUS],
            optional_semantics={},
            required_entities=[],
            preferred_domains=[],
            applicable_tasks=[],
            min_rows=5,
            min_numeric_cols=1
        )
    
    def analyze(self, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run schema-agnostic comprehensive statistical analysis.
        
        Args:
            df: Input dataframe
            config: Analysis configuration
        
        Returns:
            Dictionary with analysis results and visualizations
        """
        # SCHEMA INTELLIGENCE
        analysis_types = config.get('analysis_types', ['all'])
        
        if not config.get('skip_profiling', False):
            profiles = self.profiler.profile_dataset(df)
            results = {
                'column_mappings': {},
                'profiling_used': True,
                'summary': {},
                'visualizations': [],
                'metadata': {
                    'rows': len(df),
                    'columns': len(df.columns),
                    'analysis_types': analysis_types
                }
            }
        else:
            results = {
                'profiling_used': False,
                'summary': {},
                'visualizations': [],
                'metadata': {
                    'rows': len(df),
                    'columns': len(df.columns),
                    'analysis_types': analysis_types
                }
            }
        
        # Run requested analyses
        if 'all' in analysis_types or 'descriptive' in analysis_types:
            results['descriptive'] = self._descriptive_statistics(df, config)
        
        if 'all' in analysis_types or 'correlation' in analysis_types:
            results['correlation'] = self._correlation_analysis(df, config)
        
        if 'all' in analysis_types or 'anova' in analysis_types:
            group_by = config.get('group_by')
            if group_by:
                results['anova'] = self._anova_analysis(df, config)
        
        if 'all' in analysis_types or 'regression' in analysis_types:
            target = config.get('target')
            features = config.get('features') or []
            if target and features:
                results['regression'] = self._regression_analysis(df, config)
        
        if 'all' in analysis_types or 'distribution' in analysis_types:
            results['distribution'] = self._distribution_analysis(df, config)
        
        if 'all' in analysis_types or 'hypothesis' in analysis_types:
            results['hypothesis'] = self._hypothesis_testing(df, config)
        
        # Generate visualizations metadata
        results['visualizations'] = self._generate_visualizations_metadata(results)
        
        # Generate summary
        results['summary'] = self._generate_summary(results, df)
        
        return results
    
    def _descriptive_statistics(self, df: pd.DataFrame, config: Dict) -> Dict:
        """Calculate comprehensive descriptive statistics"""
        # Get numeric columns but exclude booleans (quantile fails on bool)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if df[c].dtype != 'bool']
        
        stats_dict = {}
        for col in numeric_cols:
            data = df[col].dropna()
            
            # Skip if empty or if still boolean-like
            if len(data) == 0:
                continue
                
            # Convert to float to avoid boolean issues
            data = data.astype(float)
            
            try:
                stats_dict[col] = {
                    'count': int(len(data)),
                    'mean': float(data.mean()),
                    'median': float(data.median()),
                    'std': float(data.std()),
                    'variance': float(data.var()),
                    'min': float(data.min()),
                    'max': float(data.max()),
                    'q25': float(data.quantile(0.25)),
                    'q75': float(data.quantile(0.75)),
                    'iqr': float(data.quantile(0.75) - data.quantile(0.25)),
                    'skewness': float(data.skew()),
                    'kurtosis': float(data.kurtosis()),
                    'coefficient_of_variation': float(data.std() / data.mean() if data.mean() != 0 else 0),
                    'confidence_interval': self._confidence_interval(data, config.get('confidence_level', 0.95))
                }
            except (TypeError, ValueError):
                # Skip columns that cause errors
                continue
        
        # Categorical statistics
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        categorical_stats = {}
        
        for col in categorical_cols:
            value_counts = df[col].value_counts()
            categorical_stats[col] = {
                'unique_count': int(df[col].nunique()),
                'mode': str(df[col].mode()[0]) if len(df[col].mode()) > 0 else None,
                'top_values': [
                    {'value': str(val), 'count': int(count), 'percentage': float(count / len(df) * 100)}
                    for val, count in value_counts.head(10).items()
                ]
            }
        
        return {
            'numeric': stats_dict,
            'categorical': categorical_stats
        }
    
    def _confidence_interval(self, data: pd.Series, confidence: float = 0.95) -> Dict:
        """Calculate confidence interval for a dataset"""
        mean = data.mean()
        sem = stats.sem(data)
        ci = stats.t.interval(confidence, len(data) - 1, loc=mean, scale=sem)
        
        return {
            'confidence_level': confidence,
            'lower': float(ci[0]),
            'upper': float(ci[1]),
            'margin_of_error': float(ci[1] - mean)
        }
    
    def _correlation_analysis(self, df: pd.DataFrame, config: Dict) -> Dict:
        """Calculate correlation matrices (Pearson, Spearman, Kendall)"""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) < 2:
            return {'error': 'Need at least 2 numeric columns for correlation analysis'}
        
        # Pearson correlation
        pearson_corr = numeric_df.corr(method='pearson')
        
        # Spearman correlation (rank-based, robust to outliers)
        spearman_corr = numeric_df.corr(method='spearman')
        
        # Kendall correlation (for small samples)
        kendall_corr = numeric_df.corr(method='kendall')
        
        # Find strongest correlations
        strong_correlations = self._find_strong_correlations(pearson_corr, threshold=0.7)
        
        return {
            'pearson': pearson_corr.to_dict(),
            'spearman': spearman_corr.to_dict(),
            'kendall': kendall_corr.to_dict(),
            'strong_correlations': strong_correlations,
            'columns': numeric_df.columns.tolist()
        }
    
    def _find_strong_correlations(self, corr_matrix: pd.DataFrame, threshold: float = 0.7) -> List[Dict]:
        """Identify pairs of variables with strong correlation"""
        strong_pairs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) >= threshold:
                    strong_pairs.append({
                        'var1': corr_matrix.columns[i],
                        'var2': corr_matrix.columns[j],
                        'correlation': float(corr_value),
                        'strength': 'strong' if abs(corr_value) >= 0.9 else 'moderate'
                    })
        
        return sorted(strong_pairs, key=lambda x: abs(x['correlation']), reverse=True)
    
    def _anova_analysis(self, df: pd.DataFrame, config: Dict) -> Dict:
        """Perform ANOVA (Analysis of Variance) to compare group means"""
        group_col = config.get('group_by')
        if not group_col:
            return {'error': 'No group_by column specified for ANOVA'}
        
        target_cols = config.get('target_columns')
        if not target_cols:
            target_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        results = {}
        
        for target_col in target_cols:
            if target_col == group_col:
                continue
            
            # Get groups
            groups = df.groupby(group_col)[target_col].apply(list)
            
            if len(groups) < 2:
                continue
            
            # One-way ANOVA
            f_stat, p_value = stats.f_oneway(*groups.values)
            
            # Effect size (eta-squared)
            grand_mean = df[target_col].mean()
            ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups.values)
            ss_total = sum((df[target_col] - grand_mean) ** 2)
            eta_squared = ss_between / ss_total if ss_total > 0 else 0
            
            results[target_col] = {
                'f_statistic': float(f_stat),
                'p_value': float(p_value),
                'significant': p_value < 0.05,
                'eta_squared': float(eta_squared),
                'effect_size': self._interpret_effect_size(eta_squared),
                'group_means': {str(k): float(np.mean(v)) for k, v in groups.items()},
                'group_counts': {str(k): len(v) for k, v in groups.items()}
            }
        
        return results
    
    def _interpret_effect_size(self, eta_squared: float) -> str:
        """Interpret eta-squared effect size (Cohen's guidelines)"""
        if eta_squared < 0.01:
            return 'negligible'
        elif eta_squared < 0.06:
            return 'small'
        elif eta_squared < 0.14:
            return 'medium'
        else:
            return 'large'
    
    def _regression_analysis(self, df: pd.DataFrame, config: Dict) -> Dict:
        """Perform linear and polynomial regression analysis"""
        target = config['target']
        features = config['features']
        
        if target not in df.columns:
            return {'error': f'Target column {target} not found'}
        
        # Prepare data
        X = df[features].dropna()
        y = df.loc[X.index, target]
        
        # Linear regression
        linear_model = LinearRegression()
        linear_model.fit(X, y)
        
        linear_predictions = linear_model.predict(X)
        linear_r2 = linear_model.score(X, y)
        
        # Calculate residuals
        residuals = y - linear_predictions
        
        # Polynomial regression (degree 2)
        poly_features = PolynomialFeatures(degree=2)
        X_poly = poly_features.fit_transform(X)
        
        poly_model = LinearRegression()
        poly_model.fit(X_poly, y)
        poly_predictions = poly_model.predict(X_poly)
        poly_r2 = poly_model.score(X_poly, y)
        
        return {
            'linear': {
                'coefficients': {feat: float(coef) for feat, coef in zip(features, linear_model.coef_)},
                'intercept': float(linear_model.intercept_),
                'r_squared': float(linear_r2),
                'adjusted_r_squared': float(1 - (1 - linear_r2) * (len(y) - 1) / (len(y) - len(features) - 1)),
                'residuals_stats': {
                    'mean': float(residuals.mean()),
                    'std': float(residuals.std()),
                    'min': float(residuals.min()),
                    'max': float(residuals.max())
                }
            },
            'polynomial': {
                'r_squared': float(poly_r2),
                'degree': 2
            },
            'target': target,
            'features': features,
            'sample_size': len(X)
        }
    
    def _distribution_analysis(self, df: pd.DataFrame, config: Dict) -> Dict:
        """Analyze distributions and test for normality"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        target_cols = config.get('target_columns') or numeric_cols
        
        results = {}
        
        for col in target_cols:
            if col not in df.columns:
                continue
            
            data = df[col].dropna()
            
            if len(data) < 3:
                continue
            
            # Shapiro-Wilk test for normality
            shapiro_stat, shapiro_p = shapiro(data) if len(data) <= 5000 else (None, None)
            
            # Anderson-Darling test
            anderson_result = anderson(data)
            
            # Detect distribution type
            distribution_type = self._detect_distribution(data)
            
            results[col] = {
                'normality_tests': {
                    'shapiro_wilk': {
                        'statistic': float(shapiro_stat) if shapiro_stat else None,
                        'p_value': float(shapiro_p) if shapiro_p else None,
                        'is_normal': shapiro_p > 0.05 if shapiro_p else None
                    } if shapiro_stat else None,
                    'anderson_darling': {
                        'statistic': float(anderson_result.statistic),
                        'critical_values': anderson_result.critical_values.tolist(),
                        'significance_levels': anderson_result.significance_level.tolist()
                    }
                },
                'distribution_type': distribution_type,
                'outliers': self._detect_outliers(data)
            }
        
        return results
    
    def _detect_distribution(self, data: pd.Series) -> str:
        """Detect the likely distribution type"""
        skewness = data.skew()
        kurtosis = data.kurtosis()
        
        # Simple heuristics
        if abs(skewness) < 0.5 and abs(kurtosis) < 1:
            return 'normal'
        elif skewness > 1:
            return 'right_skewed'
        elif skewness < -1:
            return 'left_skewed'
        elif kurtosis > 3:
            return 'heavy_tailed'
        else:
            return 'unknown'
    
    def _detect_outliers(self, data: pd.Series) -> Dict:
        """Detect outliers using IQR and Z-score methods"""
        # IQR method
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        iqr_outliers = data[(data < lower_bound) | (data > upper_bound)]
        
        # Z-score method
        z_scores = np.abs(stats.zscore(data))
        z_outliers = data[z_scores > 3]
        
        return {
            'iqr_method': {
                'count': int(len(iqr_outliers)),
                'percentage': float(len(iqr_outliers) / len(data) * 100),
                'bounds': {'lower': float(lower_bound), 'upper': float(upper_bound)}
            },
            'z_score_method': {
                'count': int(len(z_outliers)),
                'percentage': float(len(z_outliers) / len(data) * 100)
            }
        }
    
    def _hypothesis_testing(self, df: pd.DataFrame, config: Dict) -> Dict:
        """Perform various hypothesis tests"""
        results = {}
        
        # T-test (if two groups specified)
        if 'group_col' in config and 'target_col' in config:
            group_col = config['group_col']
            target_col = config['target_col']
            
            groups = df[group_col].unique()
            if len(groups) == 2:
                group1 = df[df[group_col] == groups[0]][target_col].dropna()
                group2 = df[df[group_col] == groups[1]][target_col].dropna()
                
                # Independent t-test
                t_stat, p_value = stats.ttest_ind(group1, group2)
                
                # Cohen's d effect size
                pooled_std = np.sqrt(((len(group1) - 1) * group1.std() ** 2 + 
                                     (len(group2) - 1) * group2.std() ** 2) / 
                                    (len(group1) + len(group2) - 2))
                cohens_d = (group1.mean() - group2.mean()) / pooled_std if pooled_std > 0 else 0
                
                results['t_test'] = {
                    't_statistic': float(t_stat),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05,
                    'cohens_d': float(cohens_d),
                    'effect_size': self._interpret_cohens_d(cohens_d),
                    'group1_mean': float(group1.mean()),
                    'group2_mean': float(group2.mean())
                }
        
        return results
    
    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size"""
        abs_d = abs(d)
        if abs_d < 0.2:
            return 'negligible'
        elif abs_d < 0.5:
            return 'small'
        elif abs_d < 0.8:
            return 'medium'
        else:
            return 'large'
    
    def _generate_visualizations_metadata(self, results: Dict) -> List[Dict]:
        """Generate metadata for visualizations to be rendered on frontend"""
        visualizations = []
        
        # Descriptive statistics visualizations
        if 'descriptive' in results and 'numeric' in results['descriptive']:
            # Box plots for each numeric column
            for col in results['descriptive']['numeric'].keys():
                visualizations.append({
                    'type': 'box_plot',
                    'title': f'Distribution: {col}',
                    'data_source': f'descriptive.numeric.{col}',
                    'config': {'column': col}
                })
        
        # Correlation heatmap
        if 'correlation' in results and 'pearson' in results['correlation']:
            visualizations.append({
                'type': 'heatmap',
                'title': 'Correlation Matrix (Pearson)',
                'data_source': 'correlation.pearson',
                'config': {'color_scale': 'RdBu', 'center': 0}
            })
        
        # ANOVA results
        if 'anova' in results:
            for target_col in results['anova'].keys():
                visualizations.append({
                    'type': 'bar_chart',
                    'title': f'Group Means: {target_col}',
                    'data_source': f'anova.{target_col}.group_means',
                    'config': {'orientation': 'vertical'}
                })
        
        # Regression plots
        if 'regression' in results:
            visualizations.append({
                'type': 'scatter_plot',
                'title': 'Regression: Actual vs Predicted',
                'data_source': 'regression',
                'config': {'show_regression_line': True}
            })
        
        return visualizations
    
    def _generate_summary(self, results: Dict, df: pd.DataFrame) -> Dict:
        """Generate executive summary of statistical analysis"""
        summary = {
            'total_variables': len(df.columns),
            'numeric_variables': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_variables': len(df.select_dtypes(include=['object', 'category']).columns),
            'sample_size': len(df)
        }
        
        # Key insights
        insights = []
        
        if 'correlation' in results and 'strong_correlations' in results['correlation']:
            strong_corr = results['correlation']['strong_correlations']
            if strong_corr:
                insights.append(f"Found {len(strong_corr)} strong correlations between variables")
        
        if 'anova' in results:
            if isinstance(results['anova'], dict):
                significant_anova = [
                    k for k, v in results['anova'].items()
                    if isinstance(v, dict) and v.get('significant')
                ]
            else:
                significant_anova = []
            if significant_anova:
                insights.append(f"{len(significant_anova)} variables show significant group differences")
        
        summary['insights'] = insights
        
        return summary
