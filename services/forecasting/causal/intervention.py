#!/usr/bin/env python3
"""
=============================================================================
Bayesian Structural Time Series (BSTS) concept for Causal Intervention.
Replicates Google's CausalImpact framework. Compares post-intervention actuals 
against a synthetic control counterfactual constructed from pre-intervention covariates.
=============================================================================
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any

class BayesianCausalImpact:
    @staticmethod
    def estimate_causal_impact(
        pre_intervention_data: pd.DataFrame,
        post_intervention_data: pd.DataFrame,
        target_col: str,
        covariate_cols: list
    ) -> Dict[str, Any]:
        """
        Calculates post-intervention target uplift, relative effect sizes, and p-values
        by constructing a synthetic counterfactual baseline.
        """
        # 1. Fit linear synthetic control model over pre-intervention baseline covariates
        X_pre = pre_intervention_data[covariate_cols].values
        y_pre = pre_intervention_data[target_col].values
        
        # Append intercept
        X_pre_bias = np.hstack([np.ones((X_pre.shape[0], 1)), X_pre])
        
        # Closed-form Least Squares regression weights: Beta = (X^T * X)^-1 * X^T * y
        try:
            beta = np.linalg.pinv(X_pre_bias.T @ X_pre_bias) @ X_pre_bias.T @ y_pre
        except np.linalg.LinAlgError:
            beta = np.zeros(X_pre_bias.shape[1])
            beta[0] = np.mean(y_pre)

        # 2. Predict synthetic counterfactual over the post-intervention window
        X_post = post_intervention_data[covariate_cols].values
        y_post_actual = post_intervention_data[target_col].values
        
        X_post_bias = np.hstack([np.ones((X_post.shape[0], 1)), X_post])
        y_post_counterfactual = X_post_bias @ beta
        
        # 3. Calculate absolute and cumulative effects
        absolute_effect = y_post_actual - y_post_counterfactual
        cumulative_actual = np.sum(y_post_actual)
        cumulative_counterfactual = np.sum(y_post_counterfactual)
        cumulative_absolute_effect = np.sum(absolute_effect)
        
        relative_effect = (cumulative_actual - cumulative_counterfactual) / cumulative_counterfactual if cumulative_counterfactual != 0 else 0
        
        # 4. Perform statistical significance test (Welch's t-test over effects)
        t_stat, p_value = stats.ttest_ind(y_post_actual, y_post_counterfactual, equal_var=False)
        
        return {
            "summary": {
                "actual_cumulative": float(cumulative_actual),
                "counterfactual_cumulative": float(cumulative_counterfactual),
                "absolute_cumulative_uplift": float(cumulative_absolute_effect),
                "relative_lift_pct": float(relative_effect * 100.0)
            },
            "statistical_significance": {
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "is_statistically_significant": bool(p_value < 0.05)
            },
            "trajectories": {
                "actual": y_post_actual.tolist(),
                "counterfactual": y_post_counterfactual.tolist(),
                "pointwise_lift": absolute_effect.tolist()
            }
        }

if __name__ == "__main__":
    print("[*] Testing Causal Intervention Analysis Model...")
    np.random.seed(42)
    pre_dates = pd.date_range("2026-05-01", periods=100)
    post_dates = pd.date_range("2026-05-15", periods=30)
    
    covariates_pre = np.random.normal(50, 5, (100, 2))
    target_pre = covariates_pre[:, 0] * 1.5 + covariates_pre[:, 1] * 0.5 + np.random.normal(0, 2, 100)
    
    covariates_post = np.random.normal(50, 5, (30, 2))
    target_post = (covariates_post[:, 0] * 1.5 + covariates_post[:, 1] * 0.5) * 1.15 + np.random.normal(0, 2, 30)
    
    df_pre = pd.DataFrame(covariates_pre, columns=["marketing_spend", "search_traffic"], index=pre_dates)
    df_pre["orders"] = target_pre
    
    df_post = pd.DataFrame(covariates_post, columns=["marketing_spend", "search_traffic"], index=post_dates)
    df_post["orders"] = target_post
    
    impact = BayesianCausalImpact.estimate_causal_impact(
        df_pre, df_post, target_col="orders", covariate_cols=["marketing_spend", "search_traffic"]
    )
    
    print(f"[+] Estimated Relative Uplift: {impact['summary']['relative_lift_pct']:.2f}%")
    print(f"[+] Is Statistically Significant: {impact['statistical_significance']['is_statistically_significant']}")