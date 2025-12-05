"""
Business Insight Generator - Translates ML outputs to actionable business insights
Converts technical ML results into human-readable, executive-friendly analytics.
"""

from typing import Dict, List, Any
import numpy as np


class BusinessInsightGenerator:
    """
    Translates technical ML outputs into business-friendly insights.
    Designed to showcase both ML expertise and business acumen.
    """
    
    @staticmethod
    def generate_clustering_insights(result: Dict[str, Any], df_name: str = "dataset") -> List[str]:
        """Convert clustering results to business insights"""
        insights = []
        
        n_clusters = result.get('n_clusters', 0)
        metrics = result.get('metrics', {})
        cluster_profiles = result.get('cluster_profiles', {})
        features_used = result.get('metadata', {}).get('features_used', [])
        
        # Header insight
        insights.append(f"🎯 **Segmentation Analysis**: Identified {n_clusters} distinct customer/data segments using ML clustering on {len(features_used)} features.")
        
        # Quality metric
        silhouette = metrics.get('silhouette_score', 0)
        if silhouette > 0.5:
            insights.append(f"✅ **High-Quality Segments**: Silhouette score of {silhouette:.2f} indicates well-separated, actionable segments (>0.5 is excellent).")
        elif silhouette > 0.25:
            insights.append(f"⚠️ **Moderate Segments**: Silhouette score of {silhouette:.2f} shows some overlap between segments - consider refining targeting strategy.")
        else:
            insights.append(f"❌ **Weak Segmentation**: Score {silhouette:.2f} suggests segments blend together - data may not have natural groupings.")
        
        # Segment size distribution
        if cluster_profiles:
            # Handle both dict and list formats
            if isinstance(cluster_profiles, dict):
                sizes = [prof.get('size', 0) for prof in cluster_profiles.values()]
            elif isinstance(cluster_profiles, list):
                sizes = [prof.get('size', 0) if isinstance(prof, dict) else 0 for prof in cluster_profiles]
            else:
                sizes = []
            
            if sizes:
                largest_pct = max(sizes) / sum(sizes) * 100 if sum(sizes) > 0 else 0
                smallest_pct = min(sizes) / sum(sizes) * 100 if sum(sizes) > 0 else 0
                
                if largest_pct > 60:
                    insights.append(f"⚖️ **Imbalanced Segments**: Largest segment contains {largest_pct:.0f}% of data - consider if this represents your true market structure.")
                else:
                    insights.append(f"✅ **Balanced Distribution**: Segments range from {smallest_pct:.0f}% to {largest_pct:.0f}% - good balance for targeted strategies.")
        
        # Actionable recommendation
        insights.append(f"💡 **Business Action**: Use these {n_clusters} segments for personalized marketing, pricing strategies, or resource allocation. Each segment likely has different needs and value.")
        
        # Feature importance
        if features_used:
            insights.append(f"📊 **Key Differentiators**: Segmentation based on: {', '.join(features_used[:3])}{'...' if len(features_used) > 3 else ''}")
        
        return insights
    
    @staticmethod
    def generate_anomaly_insights(result: Dict[str, Any], df_name: str = "dataset") -> List[str]:
        """Convert anomaly detection to business insights"""
        insights = []
        
        outlier_count = result.get('outlier_count', 0)
        total_points = result.get('total_points', 0)
        outlier_pct = (outlier_count / total_points * 100) if total_points > 0 else 0
        features = result.get('features_analyzed', [])
        
        # Header
        insights.append(f"🚨 **Anomaly Detection**: Found {outlier_count} outliers ({outlier_pct:.1f}% of {total_points} records) using ML-based outlier detection.")
        
        # Severity assessment
        if outlier_pct > 10:
            insights.append(f"⚠️ **High Anomaly Rate** :{outlier_pct:.1f}% suggests systemic issues - investigate data quality, fraud, or process breakdowns.")
        elif outlier_pct > 2:
            insights.append(f"📊 **Normal Anomaly Rate**: {outlier_pct:.1f}% is expected - likely represents edge cases, VIP customers, or special transactions.")
        else:
            insights.append(f"✅ **Clean Data**: Only {outlier_pct:.1f}% anomalies detected - data appears consistent and reliable.")
        
        # Business context
        if outlier_count > 0:
            if outlier_count < 5:
                insights.append(f"🔍 **High-Priority Review**: With only {outlier_count} outliers, manually inspect each one - could be fraud, errors, or high-value opportunities.")
            elif outlier_count < 50:
                insights.append(f"📋 **Systematic Review**: {outlier_count} outliers warrant investigation - prioritize by monetary value or business impact.")
            else:
                insights.append(f"🤖 **Automated Flagging**: {outlier_count} outliers require automated rules - flag for review rather than manual inspection.")
        
        # Actionable
        insights.append(f"💰 **Potential Value**: Anomalies often represent fraud to prevent, errors to fix, or exceptional opportunities to replicate.")
        
        return insights
    
    @staticmethod
    def generate_prediction_insights(result: Dict[str, Any], df_name: str = "dataset") -> List[str]:
        """Convert prediction results to business insights"""
        insights = []
        
        accuracy = result.get('accuracy', 0)
        model_type = result.get('model_type', 'ML model')
        features_used = result.get('features_used', [])
        predictions = result.get('predictions', [])
        
        # Model performance
        insights.append(f"🎯 **Predictive Model**: Built {model_type} with {accuracy*100:.1f}% accuracy on {len(predictions)} predictions.")
        
        if accuracy > 0.85:
            insights.append(f"✅ **Highly Reliable**: {accuracy*100:.1f}% accuracy means you can confidently act on these predictions for business decisions.")
        elif accuracy > 0.70:
            insights.append(f"⚠️ **Use with Caution**: {accuracy*100:.1f}% accuracy is decent but verify high-stakes predictions manually.")
        else:
            insights.append(f"❌ **Low Confidence**: {accuracy*100:.1f}% accuracy - use as one input among many, not sole decision driver.")
        
        # Feature importance
        if features_used:
            insights.append(f"📊 **Key Drivers**: Predictions based on: {', '.join(features_used[:3])}{'...' if len(features_used) > 3 else ''}")
        
        # Business value
        insights.append(f"💡 **Business Application**: Use predictions for proactive decision-making - prevent churn, optimize inventory, or personalize offers before issues arise.")
        
        return insights
    
    @staticmethod
    def generate_market_basket_insights(result: Dict[str, Any], df_name: str = "dataset") -> List[str]:
        """Convert market basket analysis to business insights"""
        insights = []
        
        rules = result.get('association_rules', [])
        min_support = result.get('min_support', 0)
        min_confidence = result.get('min_confidence', 0)
        
        if not rules:
            return [f"📊 **No Strong Patterns**: No product associations found above {min_confidence*100:.0f}% confidence threshold - items may be purchased independently."]
        
        # Header
        insights.append(f"🛒 **Product Recommendations**: Discovered {len(rules)} purchase patterns for cross-selling and bundling opportunities.")
        
        # Top rule
        if rules:
            top_rule = rules[0]
            antecedent = top_rule.get('antecedent', 'ItemA')
            consequent = top_rule.get('consequent', 'ItemB')
            confidence = top_rule.get('confidence', 0)
            lift = top_rule.get('lift', 1)
            
            insights.append(f"🏆 **Strongest Pattern**: Customers who buy '{antecedent}' have {confidence*100:.0f}% chance of also buying '{consequent}' (lift: {lift:.2f}x)")
            
            if lift > 2:
                insights.append(f"💰 **High-Impact Bundle**: {lift:.1f}x lift means this combination is {lift:.1f} times more likely than random - create bundle or place items near each other!")
            elif lift > 1.5:
                insights.append(f"📈 **Good Opportunity**: {lift:.1f}x lift shows moderate correlation - test cross-promotions or product placement.")
        
        # Business actions
        insights.append(f"💡 **Recommended Actions**:")
        insights.append(f"   • Create product bundles for frequently co-purchased items")
        insights.append(f"   • Place related items adjacent in store/website")
        insights.append(f"   • Trigger 'customers also bought' recommendations")
        insights.append(f"   • Offer discounts on complementary products")
        
        return insights
    
    @staticmethod
    def generate_ltv_insights(result: Dict[str, Any], df_name: str = "dataset") -> List[str]:
        """Convert customer LTV analysis to business insights"""
        insights = []
        
        avg_ltv = result.get('average_ltv', 0)
        ltv_segments = result.get('ltv_segments', {})
        churn_risk = result.get('churn_risk', {})
        
        # Header
        insights.append(f"💰 **Customer Value Analysis**: Average customer lifetime value is ${avg_ltv:,.2f}")
        
        # Segmentation
        if ltv_segments:
            high_value_pct = ltv_segments.get('high_value_percentage', 0)
            high_value_ltv = ltv_segments.get('high_value_avg_ltv', 0)
            
            insights.append(f"🎯 **High-Value Segment**: Top {high_value_pct:.0f}% of customers worth ${high_value_ltv:,.2f} on average - focus retention efforts here!")
            
            low_value_ltv = ltv_segments.get('low_value_avg_ltv', 0)
            if low_value_ltv < avg_ltv * 0.3:
                insights.append(f"⚠️ **Unprofitable Segment**: Low-value customers (${low_value_ltv:,.2f}) may cost more to serve than they generate.")
        
        # Churn implications
        if churn_risk:
            at_risk_count = churn_risk.get('at_risk_customers', 0)
            potential_loss = churn_risk.get('potential_revenue_loss', 0)
            
            if at_risk_count > 0:
                insights.append(f"🚨 **Churn Risk**: {at_risk_count} high-value customers at risk, representing ${potential_loss:,.2f} in potential loss.")
        
        # Actionable recommendations
        insights.append(f"💡 **Strategic Actions**: Invest in retention for high-LTV customers - even 5% improvement = ${avg_ltv * 0.05:,.2f} per customer saved.")
        
        return insights
    
    @staticmethod
    def generate_profit_margin_insights(result: Dict[str, Any], df_name: str = "dataset") -> List[str]:
        """Convert profit margin analysis to business insights"""
        insights = []
        
        avg_margin = result.get('average_margin', 0)
        margin_pct = result.get('average_margin_percentage', 0)
        top_products = result.get('top_margin_products', [])
        bottom_products = result.get('bottom_margin_products', [])
        
        # Header
        insights.append(f"💵 **Profitability Analysis**: Average profit margin is {margin_pct:.1f}% (${avg_margin:,.2f} per item)")
        
        # Health assessment
        if margin_pct > 40:
            insights.append(f"✅ **Healthy Margins**: {margin_pct:.0f}% is excellent - strong pricing power and cost control.")
        elif margin_pct > 20:
            insights.append(f"📊 **Moderate Margins**: {margin_pct:.0f}% is acceptable but watch for cost creep and competitive pressure.")
        else:
            insights.append(f"⚠️ **Thin Margins**: {margin_pct:.0f}% leaves little room for error - urgent cost optimization or price increases needed.")
        
        # Product insights
        if top_products:
            top_name = top_products[0].get('name', 'ProductA')
            top_margin = top_products[0].get('margin_pct', 0)
            insights.append(f"🏆 **Star Product**: '{top_name}' delivers {top_margin:.0f}% margin - promote heavily and protect pricing!")
        
        if bottom_products:
            bottom_name = bottom_products[0].get('name', 'ProductZ')
            bottom_margin = bottom_products[0].get('margin_pct', 0)
            if bottom_margin < 10:
                insights.append(f"❌ **Loss Leader**: '{bottom_name}' at {bottom_margin:.0f}% margin - discontinue unless strategic (traffic driver).")
        
        # Actions
        insights.append(f"💡 **Optimization Playbook**: Focus sales on high-margin products, renegotiate costs on low-margin items, or raise prices where possible.")
        
        return insights
    
    @staticmethod
    def generate_statistical_insights(result: Dict[str, Any], df_name: str = "dataset") -> List[str]:
        """Convert statistical analysis to business insights"""
        insights = []
        
        correlations = result.get('top_correlations', [])
        distributions = result.get('distributions', {})
        summary_stats = result.get('summary_stats', {})
        
        # Correlations
        if correlations:
            top_corr = correlations[0]
            var1 = top_corr.get('variable1', 'X')
            var2 = top_corr.get('variable2', 'Y')
            strength = top_corr.get('correlation', 0)
            
            insights.append(f"🔗 **Key Relationship**: {var1} and {var2} are {'strongly' if abs(strength) > 0.7 else 'moderately'} correlated (r={strength:.2f})")
            
            if strength > 0:
                insights.append(f"📈 **Positive Linkage**: When {var1} increases, {var2} tends to increase too - optimize both together.")
            else:
                insights.append(f"📉 **Trade-off Detected**: {var1} and {var2} move in opposite directions - balance carefully in strategy.")
        
        # Distribution insights
        for col, dist_info in list(distributions.items())[:2]:  # Top 2
            skew = dist_info.get('skewness', 0)
            if abs(skew) > 1:
                if skew > 0:
                    insights.append(f"📊 **'{col}' Distribution**: Right-skewed ({skew:.1f}) - indicates a 'Long Tail' distribution. A small number of high values (whales/power users) drive the metric.")
                else:
                    insights.append(f"📊 **'{col}' Distribution**: Left-skewed ({skew:.1f}) - most values are high, with a few low-performing outliers.")
        
        # Business interpretation
        insights.append(f"💡 **Strategic Insight**: Leverage the identified correlations to build predictive models. The skewed distributions suggest segmented strategies (e.g., VIP tier for the 'long tail') are necessary.")
        
        return insights

    @staticmethod
    def generate_general_insights(result: Dict[str, Any], df_name: str = "dataset") -> List[str]:
        """Robust fallback generator for any engine result"""
        insights = []
        
        # Try to find common patterns in result
        keys = list(result.keys())
        numeric_values = {k: v for k, v in result.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
        
        insights.append(f"📊 **General Analysis**: Successfully analyzed '{df_name}' extracting {len(result)} key metrics.")
        
        if numeric_values:
            # Sort by magnitude to find interesting numbers
            sorted_metrics = sorted(numeric_values.items(), key=lambda x: abs(x[1]), reverse=True)
            top_metrics = sorted_metrics[:3]
            
            insights.append(f"📈 **Key Metrics Detected**:")
            for k, v in top_metrics:
                name = k.replace('_', ' ').title()
                insights.append(f"   • {name}: {v:,.2f}")
                
        # Check for specific keys that might indicate success but weren't caught by specific handlers
        if 'anomalies' in keys or 'outliers' in keys:
            insights.append("🔍 **Data Quality Signal**: Analysis detected potential anomalies or outliers - review detailed data for quality issues.")
            
        if 'correlations' in keys:
            insights.append("🔗 **Relationships Found**: Data contains significant correlations between variables, suggesting predictive power.")
            
        return insights

    @staticmethod
    def generate_insights_from_engine_result(engine_name: str, result: Dict[str, Any], df_name: str = "dataset") -> List[str]:
        """
        Main entry point - routes to appropriate insight generator based on engine type.
        """
        if not result or 'error' in result:
            return [f"❌ Analysis failed: {result.get('error', 'Unknown error')}"]
        
        try:
            # Route to appropriate generator
            if engine_name == "Clustering":
                return BusinessInsightGenerator.generate_clustering_insights(result, df_name)
            elif engine_name == "Anomaly Detection":
                return BusinessInsightGenerator.generate_anomaly_insights(result, df_name)
            elif engine_name in ["Predictive", "Prediction"]:
                return BusinessInsightGenerator.generate_prediction_insights(result, df_name)
            elif engine_name == "Market Basket":
                return BusinessInsightGenerator.generate_market_basket_insights(result, df_name)
            elif engine_name == "Customer LTV":
                return BusinessInsightGenerator.generate_ltv_insights(result, df_name)
            elif engine_name == "Profit Margin":
                return BusinessInsightGenerator.generate_profit_margin_insights(result, df_name)
            elif engine_name == "Statistical":
                return BusinessInsightGenerator.generate_statistical_insights(result, df_name)
            else:
                return BusinessInsightGenerator.generate_general_insights(result, df_name)
        except Exception as e:
            # Ultimate fallback to prevent crashes
            return [f"⚠️ **Analysis Completed**: Generated raw results but insight translation failed ({str(e)}). Raw keys: {list(result.keys())}"]
