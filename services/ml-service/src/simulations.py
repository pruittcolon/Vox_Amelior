import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

class WhatIfSimulator:
    """
    Simulates "What-If" scenarios by perturbing features and re-calculating predictions.
    """
    
    def __init__(self, model, X_baseline: pd.DataFrame, target_col: str):
        self.model = model
        self.X_baseline = X_baseline
        self.target_col = target_col
        
    def simulate(self, perturbations: Dict[str, float]) -> Dict[str, Any]:
        """
        Run a simulation.
        
        Args:
            perturbations: Dict mapping feature names to percentage change (e.g., {'price': 1.10} for +10%)
                           or absolute values if we supported that (future).
                           For now, we assume these are multipliers (1.1 = +10%, 0.9 = -10%).
        
        Returns:
            Dict containing impact analysis.
        """
        try:
            # 1. Create perturbed dataset
            X_simulated = self.X_baseline.copy()
            
            applied_changes = []
            
            for feature, multiplier in perturbations.items():
                if feature in X_simulated.columns:
                    # Check if numeric
                    if pd.api.types.is_numeric_dtype(X_simulated[feature]):
                        X_simulated[feature] = X_simulated[feature] * multiplier
                        applied_changes.append(f"Adjusted {feature} by factor {multiplier}")
                    else:
                        logger.warning(f"Skipping perturbation for non-numeric feature: {feature}")
                else:
                    logger.warning(f"Feature not found for perturbation: {feature}")
            
            # 2. Get Baseline Predictions
            # We assume binary classification for churn (prob of class 1)
            if hasattr(self.model, "predict_proba"):
                baseline_probs = self.model.predict_proba(self.X_baseline)[:, 1]
                new_probs = self.model.predict_proba(X_simulated)[:, 1]
            else:
                # Regression or simple predict
                baseline_probs = self.model.predict(self.X_baseline)
                new_probs = self.model.predict(X_simulated)
                
            # 3. Calculate Impact
            avg_baseline = float(np.mean(baseline_probs))
            avg_new = float(np.mean(new_probs))
            delta = avg_new - avg_baseline
            pct_change = (delta / avg_baseline) * 100 if avg_baseline != 0 else 0.0
            
            return {
                "baseline_mean": avg_baseline,
                "simulated_mean": avg_new,
                "absolute_change": delta,
                "percent_change": pct_change,
                "applied_changes": applied_changes,
                "perturbations": perturbations
            }
            
        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            return {"error": str(e)}
