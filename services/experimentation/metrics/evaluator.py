#!/usr/bin/env python3
"""
=============================================================================
High-End A/B Experimentation Engine (Inspired by LinkedIn DARWIN)
Implements standard Welch's t-test and CUPAC (Controlled-covariate Using 
Pre-Experiment Data) variance reduction to accelerate statistical significance.
=============================================================================
"""

import numpy as np
from scipy import stats
from typing import Dict, Any

class ExperimentEvaluator:
    @staticmethod
    def evaluate_standard_ab(control_metrics: np.ndarray, treatment_metrics: np.ndarray) -> Dict[str, Any]:
        """Executes a standard two-sample independent Welch's t-test."""
        mean_c, mean_t = np.mean(control_metrics), np.mean(treatment_metrics)
        t_stat, p_val = stats.ttest_ind(control_metrics, treatment_metrics, equal_var=False)
        
        return {
            "control_mean": float(mean_c),
            "treatment_mean": float(mean_t),
            "relative_lift_pct": float((mean_t - mean_c) / mean_c * 100 if mean_c > 0 else 0.0),
            "t_statistic": float(t_stat),
            "p_value": float(p_val),
            "statistically_significant": bool(p_val < 0.05)
        }

    @staticmethod
    def evaluate_cupac(
        y_control: np.ndarray, 
        y_treatment: np.ndarray, 
        x_control_pre: np.ndarray, 
        x_treatment_pre: np.ndarray
    ) -> Dict[str, Any]:
        """
        Applies CUPAC variance reduction.
        Subtracts predictable variance based on pre-experiment covariates (x).
        """
        # Combine arrays to estimate general covariance scaling parameter (theta)
        y_all = np.concatenate([y_control, y_treatment])
        x_all = np.concatenate([x_control_pre, x_treatment_pre])
        
        cov_matrix = np.cov(y_all, x_all)
        var_x = np.var(x_all)
        
        # Calculate theta (correlation adjustment multiplier)
        theta = cov_matrix[0, 1] / var_x if var_x > 0 else 0.0
        
        # Calculate reduced variance vectors
        x_mean = np.mean(x_all)
        y_control_cupac = y_control - theta * (x_control_pre - x_mean)
        y_treatment_cupac = y_treatment - theta * (x_treatment_pre - x_mean)
        
        # Perform comparative standard analysis over CUPAC-adjusted variance arrays
        results = ExperimentEvaluator.evaluate_standard_ab(y_control_cupac, y_treatment_cupac)
        
        # Calculate how much "noise" we successfully removed
        original_var = np.var(y_all)
        cupac_var = np.var(np.concatenate([y_control_cupac, y_treatment_cupac]))
        variance_reduction = ((original_var - cupac_var) / original_var) * 100.0 if original_var > 0 else 0.0
        
        results["variance_reduction_pct"] = float(variance_reduction)
        return results

if __name__ == "__main__":
    print("[*] Simulating Standard vs. CUPAC A/B Test...")
    np.random.seed(42)
    
    # 1. Historical Data (Pre-Experiment Covariates)
    pre_control = np.random.normal(50, 10, 1000)
    pre_treatment = np.random.normal(50, 10, 1000)
    
    # 2. Experiment Data (Treatment gets a tiny +1.0 lift, hidden by noise)
    y_control = pre_control * 1.05 + np.random.normal(0, 5, 1000)
    y_treatment = (pre_treatment * 1.05) + 1.0 + np.random.normal(0, 5, 1000)
    
    print("\n--- Standard A/B Test Results ---")
    std_res = ExperimentEvaluator.evaluate_standard_ab(y_control, y_treatment)
    print(f"Lift: {std_res['relative_lift_pct']:.2f}% | P-Value: {std_res['p_value']:.4f} | Significant: {std_res['statistically_significant']}")
    
    print("\n--- CUPAC Adjusted Results (Variance Reduced) ---")
    cupac_res = ExperimentEvaluator.evaluate_cupac(y_control, y_treatment, pre_control, pre_treatment)
    print(f"Lift: {cupac_res['relative_lift_pct']:.2f}% | P-Value: {cupac_res['p_value']:.4f} | Significant: {cupac_res['statistically_significant']}")
    print(f"Variance Reduced By: {cupac_res['variance_reduction_pct']:.2f}%")