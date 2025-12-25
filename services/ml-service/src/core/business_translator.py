"""
Business Translator

Converts ML metrics and jargon into plain English that business users understand.
No technical terms - everything is explained in terms of business value.
"""

import re


class BusinessTranslator:
    """
    Translates ML metrics to plain English.

    Usage:
        translator = BusinessTranslator()
        plain = translator.translate_metric("accuracy", 0.86)
        # Returns: "correctly identifies patterns 86% of the time"
    """

    # Metric translations
    METRIC_TEMPLATES = {
        # Classification metrics
        "accuracy": {
            "template": "correctly identifies patterns {pct}% of the time",
            "good_threshold": 0.8,
            "interpretation": {
                "high": "This is excellent - the model rarely makes mistakes.",
                "medium": "This is decent, but there's room for improvement.",
                "low": "This needs work - the model makes frequent errors.",
            },
        },
        "precision": {
            "template": "when it predicts positive, it's right {pct}% of the time",
            "good_threshold": 0.75,
            "interpretation": {
                "high": "Very few false alarms.",
                "medium": "Some false positives occur.",
                "low": "Many false alarms - predictions are unreliable.",
            },
        },
        "recall": {
            "template": "catches {pct}% of all actual positive cases",
            "good_threshold": 0.75,
            "interpretation": {
                "high": "Rarely misses important cases.",
                "medium": "Misses some cases.",
                "low": "Misses many cases - not suitable for critical detection.",
            },
        },
        "f1": {
            "template": "achieves a balanced score of {pct}%",
            "good_threshold": 0.75,
            "interpretation": {
                "high": "Well-balanced performance.",
                "medium": "Acceptable balance between precision and recall.",
                "low": "Struggles to balance accuracy and coverage.",
            },
        },
        "auc": {
            "template": "{quality} at distinguishing between categories",
            "good_threshold": 0.8,
            "interpretation": {
                "high": "Excellent discrimination ability.",
                "medium": "Moderate discrimination ability.",
                "low": "Struggles to tell categories apart.",
            },
        },
        # Regression metrics
        "r2": {
            "template": "explains {pct}% of the variation in your data",
            "good_threshold": 0.7,
            "interpretation": {
                "high": "The model captures most of what's happening.",
                "medium": "The model captures the main patterns.",
                "low": "Many factors are missing from this model.",
            },
        },
        "mse": {
            "template": "average prediction error is {val:.2f} (squared)",
            "good_threshold": None,  # Context-dependent
            "interpretation": {
                "high": "Predictions are very close to actual values.",
                "medium": "Predictions have moderate error.",
                "low": "Predictions are often far from actual values.",
            },
        },
        "rmse": {
            "template": "predictions are typically off by {val:.2f}",
            "good_threshold": None,
            "interpretation": {
                "high": "Very accurate predictions.",
                "medium": "Reasonably accurate predictions.",
                "low": "Predictions vary significantly from reality.",
            },
        },
        "mae": {
            "template": "on average, predictions miss by {val:.2f}",
            "good_threshold": None,
            "interpretation": {
                "high": "Minimal prediction error.",
                "medium": "Acceptable prediction error.",
                "low": "Large prediction errors.",
            },
        },
        # Statistical metrics
        "p_value": {
            "template": "{significance}",
            "good_threshold": 0.05,
            "interpretation": {
                "high": "This relationship is statistically significant.",
                "medium": "Weak evidence of a relationship.",
                "low": "No significant relationship found.",
            },
        },
        "correlation": {
            "template": "{strength} {direction} relationship",
            "good_threshold": 0.5,
            "interpretation": {
                "high": "These variables are strongly connected.",
                "medium": "These variables are moderately connected.",
                "low": "These variables are weakly connected.",
            },
        },
    }

    # Task type descriptions
    TASK_DESCRIPTIONS = {
        "prediction": "predict future outcomes based on patterns in your data",
        "classification": "categorize items into groups",
        "regression": "estimate numerical values",
        "detection": "identify unusual patterns or anomalies",
        "generation": "create new synthetic data that looks like your real data",
        "discovery": "find hidden mathematical relationships",
        "forecasting": "predict future trends over time",
        "explanation": "understand what would need to change to get different outcomes",
        "graph_analysis": "analyze relationships and connections between entities",
    }

    # Feature impact translations
    IMPACT_TEMPLATES = {
        "positive": "Higher values of {feature} lead to higher {target}",
        "negative": "Higher values of {feature} lead to lower {target}",
        "mixed": "{feature} affects {target} in complex ways",
        "categorical": "{feature} = '{best_value}' gives the best results",
    }

    def __init__(self):
        pass

    def translate_metric(self, metric_name: str, value: float) -> str:
        """
        Translate a single metric to plain English.

        Args:
            metric_name: Name of the metric (accuracy, r2, etc.)
            value: The metric value

        Returns:
            Plain English description
        """
        metric_name = metric_name.lower().replace("-", "_").replace(" ", "_")

        if metric_name not in self.METRIC_TEMPLATES:
            return f"{metric_name} = {value:.4f}"

        template_info = self.METRIC_TEMPLATES[metric_name]
        template = template_info["template"]

        # Calculate percentage if needed
        pct = round(value * 100, 1)

        # Determine quality level
        if metric_name == "auc":
            if value >= 0.9:
                quality = "excellent"
            elif value >= 0.8:
                quality = "good"
            elif value >= 0.7:
                quality = "fair"
            else:
                quality = "poor"
            return template.format(quality=quality)

        if metric_name == "p_value":
            if value < 0.01:
                significance = "this relationship is highly statistically significant (p < 0.01)"
            elif value < 0.05:
                significance = "this relationship is statistically significant (p < 0.05)"
            elif value < 0.1:
                significance = "this relationship shows weak significance (p < 0.10)"
            else:
                significance = "no statistically significant relationship found"
            return template.format(significance=significance)

        if metric_name == "correlation":
            abs_val = abs(value)
            if abs_val >= 0.7:
                strength = "strong"
            elif abs_val >= 0.4:
                strength = "moderate"
            else:
                strength = "weak"
            direction = "positive" if value > 0 else "negative"
            return template.format(strength=strength, direction=direction)

        # Default: use percentage
        return template.format(pct=pct, val=value)

    def translate_feature_importance(
        self, feature_name: str, importance: float, stability: float, impact: str, target_name: str
    ) -> str:
        """
        Translate feature importance to plain English.

        Args:
            feature_name: Name of the feature
            importance: Importance score (0-1)
            stability: How often feature appears in bootstrap (0-100)
            impact: "positive", "negative", or "mixed"
            target_name: What we're predicting

        Returns:
            Plain English explanation
        """
        # Clean up feature name
        clean_name = self._clean_column_name(feature_name)
        clean_target = self._clean_column_name(target_name)

        # Determine importance level
        if importance >= 0.3:
            level = "the most important factor"
        elif importance >= 0.15:
            level = "an important factor"
        elif importance >= 0.05:
            level = "a contributing factor"
        else:
            level = "a minor factor"

        # Determine reliability
        if stability >= 90:
            reliability = "This is highly reliable"
        elif stability >= 70:
            reliability = "This is fairly reliable"
        else:
            reliability = "This is somewhat reliable"

        # Build explanation
        if impact == "positive":
            direction = f"Higher {clean_name} tends to increase {clean_target}"
        elif impact == "negative":
            direction = f"Higher {clean_name} tends to decrease {clean_target}"
        else:
            direction = f"{clean_name} affects {clean_target} in complex ways"

        return f"{clean_name} is {level} in predicting {clean_target}. {direction}. {reliability} (appeared in {stability:.0f}% of tests)."

    def translate_task_type(self, task_type: str) -> str:
        """Get plain English description of task type"""
        return self.TASK_DESCRIPTIONS.get(task_type.lower(), task_type)

    def generate_headline(self, task_type: str, target_column: str, best_score: float, score_metric: str) -> str:
        """
        Generate a headline for the analysis results.

        Example: "Revenue can be predicted with 89% accuracy"
        """
        clean_target = self._clean_column_name(target_column)

        if task_type in ["classification", "prediction"]:
            if score_metric in ["accuracy", "f1", "auc"]:
                pct = round(best_score * 100, 1)
                return f"{clean_target} can be predicted with {pct}% accuracy"
            else:
                return f"{clean_target} prediction model is ready"

        elif task_type == "regression":
            pct = round(best_score * 100, 1)
            return f"{clean_target} can be estimated with {pct}% of variance explained"

        elif task_type == "forecasting":
            return f"Future {clean_target} trends have been forecasted"

        elif task_type == "detection":
            return f"Anomaly patterns in {clean_target} have been identified"

        elif task_type == "discovery":
            return f"Mathematical relationships in {clean_target} have been discovered"

        elif task_type == "generation":
            return f"Synthetic data matching {clean_target} patterns is ready"

        elif task_type == "explanation":
            return f"What-if analysis for {clean_target} is available"

        else:
            return f"Analysis of {clean_target} is complete"

    def generate_explanation(
        self, task_type: str, target_column: str, top_features: list[str], best_score: float, score_metric: str
    ) -> str:
        """
        Generate a 2-3 sentence explanation of the results.
        """
        clean_target = self._clean_column_name(target_column)
        clean_features = [self._clean_column_name(f) for f in top_features[:3]]

        features_text = ", ".join(clean_features[:-1])
        if len(clean_features) > 1:
            features_text += f" and {clean_features[-1]}"
        else:
            features_text = clean_features[0] if clean_features else "the available data"

        pct = round(best_score * 100, 1)

        if task_type in ["classification", "prediction"]:
            return (
                f"Based on your data, we can predict {clean_target} using {features_text}. "
                f"The model achieves {pct}% {score_metric}, meaning it correctly identifies patterns "
                f"in most cases. {self._get_score_context(best_score)}"
            )

        elif task_type == "regression":
            return (
                f"We found that {features_text} are the main drivers of {clean_target}. "
                f"Together, they explain {pct}% of the variation in your data. "
                f"{self._get_score_context(best_score)}"
            )

        elif task_type == "forecasting":
            return (
                f"Using historical patterns in {features_text}, we've projected future {clean_target}. "
                f"The forecast accounts for trends and seasonality in your data."
            )

        elif task_type == "detection":
            return (
                f"We scanned your data for unusual patterns in {clean_target}. "
                f"The analysis uses {features_text} to establish what's 'normal' and flag outliers."
            )

        else:
            return f"The analysis of {clean_target} is complete using {features_text}."

    def generate_recommendation(
        self, task_type: str, target_column: str, top_features: list[str], feature_impacts: dict[str, str]
    ) -> str:
        """
        Generate an actionable recommendation.
        """
        clean_target = self._clean_column_name(target_column)

        if not top_features:
            return f"Collect more data to improve {clean_target} predictions."

        top_feature = self._clean_column_name(top_features[0])
        impact = feature_impacts.get(top_features[0], "positive")

        if impact == "positive":
            action = "increase"
        elif impact == "negative":
            action = "optimize (reduce)"
        else:
            action = "analyze"

        return f"Focus on {top_feature} - it has the strongest influence on {clean_target}. Consider ways to {action} this factor to improve outcomes."

    def get_confidence_level(self, score: float, stability: float) -> str:
        """
        Determine overall confidence level based on score and stability.
        """
        if score >= 0.85 and stability >= 85:
            return "high"
        elif score >= 0.7 and stability >= 70:
            return "medium"
        else:
            return "low"

    def translate_feature(self, feature_name: str, stability: float, is_stable: bool, target_name: str) -> str:
        """
        Translate a feature's importance to plain English.

        Args:
            feature_name: Name of the feature
            stability: Stability score (0-1)
            is_stable: Whether feature passed stability threshold
            target_name: What we're predicting

        Returns:
            Plain English explanation
        """
        clean_name = self._clean_column_name(feature_name)
        clean_target = self._clean_column_name(target_name)
        pct = round(stability * 100, 1)

        if is_stable and stability >= 0.9:
            return f"{clean_name} is a highly reliable predictor of {clean_target}, appearing in {pct}% of bootstrap samples."
        elif is_stable and stability >= 0.8:
            return f"{clean_name} is a reliable predictor of {clean_target}, selected in {pct}% of validation tests."
        elif stability >= 0.5:
            return f"{clean_name} shows moderate influence on {clean_target} ({pct}% stability)."
        else:
            return f"{clean_name} has weak predictive value for {clean_target} ({pct}% stability)."

    def generate_summary(
        self, engine_name: str, task_type: str, target: str, score: float, stable_features: int, total_features: int
    ) -> str:
        """
        Generate a summary explanation for analysis results.

        Args:
            engine_name: Name of the engine used
            task_type: Type of task (classification, regression, etc.)
            target: Target column name
            score: Best score achieved
            stable_features: Number of stable features found
            total_features: Total number of features analyzed

        Returns:
            2-3 sentence plain English summary
        """
        clean_target = self._clean_column_name(target)
        engine_info = self.translate_engine_name(engine_name)

        # Build the summary
        pct = round(score * 100 if 0 <= score <= 1 else score, 1)
        ratio = stable_features / max(1, total_features)

        if task_type == "classification":
            performance = f"achieves {pct}% accuracy"
        elif task_type == "regression":
            performance = f"has an average error of {abs(score):.2f}"
        else:
            performance = f"scores {pct}%"

        if ratio >= 0.5:
            feature_quality = f"Found {stable_features} reliable predictors out of {total_features} features analyzed."
        elif ratio >= 0.2:
            feature_quality = (
                f"Found {stable_features} stable features - consider engineering more features for better results."
            )
        else:
            feature_quality = (
                f"Only {stable_features} of {total_features} features are statistically robust - more data may help."
            )

        return (
            f"The {engine_info['display_name']} model {performance} when predicting {clean_target}. {feature_quality}"
        )

    def translate_engine_name(self, engine_name: str) -> dict[str, str]:
        """
        Get display name and description for an engine.
        """
        engines = {
            "titan": {
                "display_name": "Titan AutoML",
                "icon": "🔮",
                "description": "Enterprise-grade prediction with robust validation",
            },
            "chaos": {
                "display_name": "Chaos Detector",
                "icon": "🌀",
                "description": "Non-linear relationship discovery",
            },
            "scout": {
                "display_name": "Scout Monitor",
                "icon": "🔍",
                "description": "Data drift and concept change detection",
            },
            "oracle": {
                "display_name": "Oracle Causality",
                "icon": "🔗",
                "description": "Cause-and-effect relationship analysis",
            },
            "newton": {
                "display_name": "Newton Equations",
                "icon": "📐",
                "description": "Discover mathematical formulas in your data",
            },
            "flash": {
                "display_name": "Flash What-If",
                "icon": "⚡",
                "description": "Counterfactual explanations and scenarios",
            },
            "mirror": {
                "display_name": "Mirror Synthetic",
                "icon": "🪞",
                "description": "Privacy-preserving synthetic data generation",
            },
            "chronos": {
                "display_name": "Chronos Forecast",
                "icon": "⏰",
                "description": "Time series prediction and trend analysis",
            },
            "deep_feature": {
                "display_name": "DeepFeature Engineering",
                "icon": "🧬",
                "description": "Automated feature creation and optimization",
            },
            "galileo": {
                "display_name": "Galileo Graph AI",
                "icon": "🌐",
                "description": "Network and relationship analysis",
            },
        }
        return engines.get(
            engine_name.lower(),
            {"display_name": engine_name.title(), "icon": "📊", "description": "Data analysis engine"},
        )

    def _clean_column_name(self, name: str) -> str:
        """
        Clean up a column name for display.

        'customer_lifetime_value' -> 'customer lifetime value'
        'CLV' -> 'CLV' (keep acronyms)
        'camelCase' -> 'camel case'
        """
        if not name:
            return "the target"

        # Check if it's an acronym (all caps, short)
        if name.isupper() and len(name) <= 5:
            return name

        # Convert camelCase to spaces
        name = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)

        # Convert snake_case to spaces
        name = name.replace("_", " ")

        # Convert kebab-case to spaces
        name = name.replace("-", " ")

        # Lowercase and strip
        return name.lower().strip()

    def _get_score_context(self, score: float) -> str:
        """Get contextual statement based on score level"""
        if score >= 0.9:
            return "This is excellent performance."
        elif score >= 0.8:
            return "This is good performance suitable for most business decisions."
        elif score >= 0.7:
            return "This is acceptable but could be improved with more data."
        elif score >= 0.6:
            return "Use these predictions with caution - accuracy is moderate."
        else:
            return "This model needs improvement before business use."


# Convenience function for quick translations
def translate(metric_name: str, value: float) -> str:
    """Quick translation function"""
    return BusinessTranslator().translate_metric(metric_name, value)
