#!/usr/bin/env python3
"""
=============================================================================
Quasi-Experimentation: Difference-in-Differences (DiD)
Used when true randomized A/B testing is impossible (e.g., turning on a new 
marketing campaign for a whole city). Compares the target group against a 
similar control group over time.
=============================================================================
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import Dict, Any

class DifferenceInDifferences:
    @staticmethod
    def evaluate(df: pd.DataFrame, time_col: str, group_col: str, target_col: str, intervention_time: Any) -> Dict[str, Any]:
        """
        Executes a DiD linear regression to find the true causal effect of an intervention.
        df must contain: time_col, group_col (0 for control, 1 for treatment), and target_col.
        """
        # Create Dummy Variables
        df['post_intervention'] = (df[time_col] >= intervention_time).astype(int)
        df['treatment_group'] = df[group_col].astype(int)
        
        # Interaction Term (This captures the actual Difference-in-Difference effect)
        df['did_interaction'] = df['post_intervention'] * df['treatment_group']
        
        # Define independent variables (X) and dependent variable (y)
        X = df[['treatment_group', 'post_intervention', 'did_interaction']]
        X = sm.add_constant(X)
        y = df[target_col]
        
        # Fit OLS Regression
        model = sm.OLS(y, X).fit()
        
        # Extract the interaction coefficient and p-value
        did_effect = model.params['did_interaction']
        p_value = model.pvalues['did_interaction']
        
        return {
            "causal_uplift": float(did_effect),
            "p_value": float(p_value),
            "statistically_significant": bool(p_value < 0.05),
            "r_squared": float(model.rsquared)
        }

if __name__ == "__main__":
    print("[*] Simulating Difference-in-Differences (DiD) Quasi-Experiment...")
    
    # Generate Mock Data
    np.random.seed(42)
    dates = pd.date_range("2026-05-01", periods=30).tolist() * 2
    groups = [0] * 30 + [1] * 30  # 0: Control City, 1: Treatment City
    
    df = pd.DataFrame({"date": dates, "group": groups})
    
    # Baseline metric logic
    base_metric = 100
    df["metric"] = base_metric + df["group"] * 10 + np.random.normal(0, 2, 60)
    
    # Inject Intervention at day 15 for the Treatment Group only
    intervention_date = pd.to_datetime("2026-05-15")
    treatment_mask = (df["group"] == 1) & (df["date"] >= intervention_date)
    
    # Inject a true +15.0 uplift
    df.loc[treatment_mask, "metric"] += 15.0 
    
    # Evaluate
    result = DifferenceInDifferences.evaluate(
        df, time_col="date", group_col="group", target_col="metric", intervention_time=intervention_date
    )
    
    print(f"[+] Detected Causal Uplift (True injected was +15.0): +{result['causal_uplift']:.2f}")
    print(f"[+] P-Value: {result['p_value']:.5f}")
    print(f"[+] Significant: {result['statistically_significant']}")