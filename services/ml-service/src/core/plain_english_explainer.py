"""
Plain English Explainer - ML Results for Laypeople

Translates ML metrics into understandable explanations.
No jargon, just clear cause-and-effect language.

Author: NeMo Analytics Team
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class MetricQuality(Enum):
    """Quality tiers for metrics."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    FAILED = "failed"


@dataclass
class PlainEnglishExplanation:
    """Complete explanation in plain English."""
    headline: str
    what_this_means: str
    why_it_works: Optional[str]
    what_failed: Optional[str]
    recommendation: str
    quality: MetricQuality
    

class PlainEnglishExplainer:
    """
    Translates ML metrics into layman-friendly explanations.
    
    Example:
        explainer = PlainEnglishExplainer()
        result = explainer.explain(
            metric_type='r2',
            value=0.67,
            target='salary',
            features=['industry', 'experience_years']
        )
        print(result.headline)
        # "Salary is 67% predictable!"
    """
    
    def __init__(self):
        # R² thresholds and templates
        self.r2_thresholds = {
            MetricQuality.EXCELLENT: (0.8, 1.0),
            MetricQuality.GOOD: (0.5, 0.8),
            MetricQuality.FAIR: (0.3, 0.5),
            MetricQuality.POOR: (0.1, 0.3),
            MetricQuality.FAILED: (0.0, 0.1),
        }
        
        # Accuracy thresholds
        self.accuracy_thresholds = {
            MetricQuality.EXCELLENT: (0.9, 1.0),
            MetricQuality.GOOD: (0.8, 0.9),
            MetricQuality.FAIR: (0.7, 0.8),
            MetricQuality.POOR: (0.5, 0.7),
            MetricQuality.FAILED: (0.0, 0.5),
        }
    
    def explain(
        self,
        metric_type: str,
        value: float,
        target: str,
        features: List[str],
        failed_attempts: Optional[List[Dict]] = None,
        dataset_context: Optional[Dict] = None
    ) -> PlainEnglishExplanation:
        """
        Generate comprehensive plain English explanation.
        
        Args:
            metric_type: 'r2', 'accuracy', 'f1', etc.
            value: Metric value (0.0 to 1.0)
            target: What we're trying to predict
            features: Features used in best model
            failed_attempts: List of feature combos that performed poorly
            dataset_context: Optional context about the dataset
        
        Returns:
            PlainEnglishExplanation with all components
        """
        if metric_type.lower() in ['r2', 'r_squared', 'r-squared']:
            return self._explain_r2(value, target, features, failed_attempts)
        elif metric_type.lower() in ['accuracy', 'acc']:
            return self._explain_accuracy(value, target, features, failed_attempts)
        elif metric_type.lower() in ['f1', 'f1_score']:
            return self._explain_f1(value, target, features, failed_attempts)
        else:
            return self._explain_generic(metric_type, value, target, features)
    
    def _get_quality(self, value: float, thresholds: Dict) -> MetricQuality:
        """Determine quality tier based on value."""
        for quality, (low, high) in thresholds.items():
            if low <= value <= high:
                return quality
        return MetricQuality.FAILED
    
    def _format_features(self, features: List[str], max_show: int = 5) -> str:
        """Format feature list for display."""
        if len(features) <= max_show:
            if len(features) == 1:
                return f"**{features[0]}**"
            elif len(features) == 2:
                return f"**{features[0]}** and **{features[1]}**"
            else:
                return ", ".join(f"**{f}**" for f in features[:-1]) + f", and **{features[-1]}**"
        else:
            shown = ", ".join(f"**{f}**" for f in features[:max_show-1])
            return f"{shown}, and {len(features) - max_show + 1} more"
    
    def _explain_r2(
        self, 
        value: float, 
        target: str, 
        features: List[str],
        failed_attempts: Optional[List[Dict]] = None
    ) -> PlainEnglishExplanation:
        """Generate R² explanation."""
        quality = self._get_quality(value, self.r2_thresholds)
        features_str = self._format_features(features)
        percent = value * 100
        remaining = (1 - value) * 100
        
        if quality == MetricQuality.EXCELLENT:
            headline = f"🎯 **{target.title()}** is highly predictable! ({percent:.0f}% accuracy)"
            what_this_means = (
                f"Using {features_str}, we can predict **{target}** with {percent:.0f}% accuracy. "
                f"This is excellent - the model captures almost all the important patterns.\n\n"
                f"**In practical terms:** For every 100 predictions, about {int(percent)} will be "
                f"very close to the actual value."
            )
            why_it_works = (
                f"These features have a strong, consistent relationship with {target}. "
                f"The patterns are clear and reliable."
            )
            recommendation = (
                f"This model is ready for production use. You can confidently predict {target} "
                f"using {features_str}."
            )
            
        elif quality == MetricQuality.GOOD:
            headline = f"✅ **{target.title()}** is {percent:.0f}% predictable"
            what_this_means = (
                f"Using {features_str}, we can explain {percent:.0f}% of why {target} varies.\n\n"
                f"**What this means:** If you know these values, you can make reasonable predictions. "
                f"About {int(percent)} out of 100 predictions will be fairly accurate, "
                f"but {int(remaining)} will have noticeable error."
            )
            why_it_works = (
                f"These features capture the main factors that influence {target}, "
                f"though some variation comes from other factors not in the data."
            )
            recommendation = (
                f"Good for general predictions and trend analysis. "
                f"Consider adding more relevant data columns to improve accuracy."
            )
            
        elif quality == MetricQuality.FAIR:
            headline = f"⚠️ **{target.title()}** is partially predictable ({percent:.0f}%)"
            what_this_means = (
                f"Using {features_str}, we can only explain {percent:.0f}% of {target} variation.\n\n"
                f"**The other {remaining:.0f}%** comes from factors not in your data. "
                f"Predictions will be rough estimates, not precise values."
            )
            why_it_works = (
                f"These features have some relationship with {target}, but it's not strong enough "
                f"for reliable predictions. Other unmeasured factors play a bigger role."
            )
            recommendation = (
                f"Use with caution. Consider what other factors might affect {target} "
                f"(industry, location, timing, etc.) and add those columns if possible."
            )
            
        elif quality == MetricQuality.POOR:
            headline = f"📉 **{target.title()}** has weak predictability ({percent:.1f}%)"
            what_this_means = (
                f"{features_str} explain only {percent:.1f}% of why {target} varies.\n\n"
                f"**This is too weak for predictions.** The relationship exists but is very noisy. "
                f"Using this model would be only slightly better than random guessing."
            )
            why_it_works = None
            recommendation = (
                f"Don't use these features alone to predict {target}. "
                f"You need to find better predictors or accept that {target} "
                f"may be inherently unpredictable from available data."
            )
            
        else:  # FAILED
            headline = f"❌ **{target.title()}** cannot be predicted from these features ({percent:.1f}%)"
            what_this_means = (
                f"{features_str} have almost NO relationship with {target}.\n\n"
                f"**What this means:** Knowing these values tells you essentially nothing about {target}. "
                f"It's like trying to predict the weather from someone's shoe size - the connection doesn't exist.\n\n"
                f"**R² of {percent:.1f}%** means the 'prediction' is no better than just guessing "
                f"the average {target} every time."
            )
            why_it_works = None
            recommendation = (
                f"These features won't help. {target.title()} likely depends on completely "
                f"different factors not captured in this dataset.\n\n"
                f"**Suggestions:**\n"
                f"• Add relevant columns (e.g., for salary: industry, company size, location)\n"
                f"• Try predicting a different target variable\n"
                f"• Accept that {target} may not be predictable from available data"
            )
        
        # Add "what failed" section if we have failed attempts
        what_failed = None
        if failed_attempts and len(failed_attempts) > 0:
            worst = sorted(failed_attempts, key=lambda x: x.get('score', 0))[:3]
            failed_features = [self._format_features(f.get('features', [])) for f in worst]
            what_failed = (
                f"**Other combinations we tried that didn't work:**\n"
                + "\n".join(f"• {feat} → Only {f.get('score', 0)*100:.1f}% accuracy" 
                           for f, feat in zip(worst, failed_features))
            )
        
        return PlainEnglishExplanation(
            headline=headline,
            what_this_means=what_this_means,
            why_it_works=why_it_works,
            what_failed=what_failed,
            recommendation=recommendation,
            quality=quality
        )
    
    def _explain_accuracy(
        self,
        value: float,
        target: str,
        features: List[str],
        failed_attempts: Optional[List[Dict]] = None
    ) -> PlainEnglishExplanation:
        """Generate accuracy explanation for classification."""
        quality = self._get_quality(value, self.accuracy_thresholds)
        features_str = self._format_features(features)
        percent = value * 100
        correct = int(percent)
        wrong = 100 - correct
        
        if quality == MetricQuality.EXCELLENT:
            headline = f"🎯 Correctly predicts **{target}** {percent:.0f}% of the time!"
            what_this_means = (
                f"Using {features_str}, the model correctly classifies {target} "
                f"for {correct} out of every 100 cases.\n\n"
                f"**Only {wrong} mistakes per 100** - this is highly reliable."
            )
            why_it_works = f"Clear patterns in {features_str} strongly indicate {target} outcomes."
            recommendation = f"Ready for production. Trust this model for {target} predictions."
            
        elif quality == MetricQuality.GOOD:
            headline = f"✅ **{target}** prediction is {percent:.0f}% accurate"
            what_this_means = (
                f"The model gets {correct} out of 100 predictions right.\n\n"
                f"**Good for most uses**, but expect {wrong} wrong predictions per 100 cases."
            )
            why_it_works = f"{features_str} are good indicators, but some cases are ambiguous."
            recommendation = f"Useful for screening and prioritization, but verify critical decisions."
            
        elif quality == MetricQuality.FAIR:
            headline = f"⚠️ **{target}** prediction is {percent:.0f}% accurate"
            what_this_means = (
                f"Gets {correct} out of 100 right, misses {wrong}.\n\n"
                f"Better than random, but significant error rate."
            )
            why_it_works = None
            recommendation = "Use as one input among many, not as sole decision-maker."
            
        else:  # POOR or FAILED
            headline = f"❌ Cannot reliably predict **{target}** ({percent:.0f}%)"
            what_this_means = (
                f"Only {correct}% accuracy - {'barely ' if value > 0.5 else ''}"
                f"{'better than' if value > 0.5 else 'worse than'} flipping a coin.\n\n"
                f"{features_str} don't contain the information needed to predict {target}."
            )
            why_it_works = None
            recommendation = f"Don't use for {target} predictions. Find better features."
        
        return PlainEnglishExplanation(
            headline=headline,
            what_this_means=what_this_means,
            why_it_works=why_it_works,
            what_failed=None,
            recommendation=recommendation,
            quality=quality
        )
    
    def _explain_f1(
        self,
        value: float,
        target: str,
        features: List[str],
        failed_attempts: Optional[List[Dict]] = None
    ) -> PlainEnglishExplanation:
        """Generate F1 score explanation."""
        # F1 uses same thresholds as accuracy roughly
        quality = self._get_quality(value, self.accuracy_thresholds)
        percent = value * 100
        
        headline = f"F1 Score: {percent:.0f}% (balanced precision & recall)"
        what_this_means = (
            f"The model balances finding all {target} cases (recall) with avoiding "
            f"false alarms (precision).\n\n"
            f"**{percent:.0f}% F1** means the model is "
            f"{'excellent' if quality == MetricQuality.EXCELLENT else 'good' if quality == MetricQuality.GOOD else 'fair' if quality == MetricQuality.FAIR else 'poor'} "
            f"at both tasks."
        )
        
        return PlainEnglishExplanation(
            headline=headline,
            what_this_means=what_this_means,
            why_it_works=None,
            what_failed=None,
            recommendation="Consider specific precision/recall needs for your use case.",
            quality=quality
        )
    
    def _explain_generic(
        self,
        metric_type: str,
        value: float,
        target: str,
        features: List[str]
    ) -> PlainEnglishExplanation:
        """Generic explanation for unknown metric types."""
        features_str = self._format_features(features)
        
        return PlainEnglishExplanation(
            headline=f"{metric_type}: {value:.3f} for **{target}**",
            what_this_means=(
                f"Using {features_str} to predict {target}, "
                f"the model achieved a {metric_type} of {value:.3f}."
            ),
            why_it_works=None,
            what_failed=None,
            recommendation="Consult documentation for this specific metric's interpretation.",
            quality=MetricQuality.FAIR
        )
    
    def explain_comparison(
        self,
        results: List[Dict],
        target: str
    ) -> str:
        """
        Explain why one result is better than others.
        
        Args:
            results: List of {features, score, score_type} dicts, sorted best-first
        
        Returns:
            Plain English comparison explanation
        """
        if len(results) < 2:
            return ""
        
        best = results[0]
        second = results[1]
        worst = results[-1]
        
        best_features = self._format_features(best['features'])
        second_features = self._format_features(second['features'])
        
        diff = (best['score'] - second['score']) * 100
        
        explanation = (
            f"**Why {best_features} is the best choice:**\n\n"
            f"• Scored {best['score']*100:.1f}% vs second-best {second_features} at {second['score']*100:.1f}%\n"
            f"• That {diff:.1f} percentage point difference means noticeably better predictions\n"
        )
        
        if len(results) > 5:
            explanation += (
                f"• We tested {len(results)} different combinations to find this winner\n"
            )
        
        if worst['score'] < 0.1:
            worst_features = self._format_features(worst['features'])
            explanation += (
                f"\n**What definitely doesn't work:** {worst_features} "
                f"(only {worst['score']*100:.1f}% - essentially useless)"
            )
        
        return explanation
