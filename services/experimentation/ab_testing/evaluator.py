import numpy as np
from scipy import stats
from typing import Dict, Any

class ExperimentEvaluator:
    """
    High-end A/B Experimentation Engine.
    Implements standard Student's t-test and CUPAC (Controlled-covariate Using Pre-Experiment Data)
    variance reduction algorithms to accelerate sample size convergence.
    """
    @staticmethod
    def evaluate_standard_ab(control_metrics: np.ndarray, treatment_metrics: np.ndarray) -> Dict[str, Any]:
        """Executes standard two-sample independent Welch's t-test."""
        mean_c, mean_t = np.mean(control_metrics), np.mean(treatment_metrics)
        t_stat, p_val = stats.ttest_ind(control_metrics, treatment_metrics, equal_var=False)
        
        return {
            "control_mean": float(mean_c),
            "treatment_mean": float(mean_t),
            "relative_lift": float((mean_t - mean_c) / mean_c if mean_c > 0 else 0.0),
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
        Utilizes pre-experiment historical covariates (x) to subtract predictable variance 
        from the active evaluation metrics (y): y_adjusted = y - theta * x
        """
        # Combine to estimate general covariance scaling parameter (theta)
        y_all = np.concatenate([y_control, y_treatment])
        x_all = np.concatenate([x_control_pre, x_treatment_pre])
        
        cov_matrix = np.cov(y_all, x_all)
        var_x = np.var(x_all)
        
        theta = cov_matrix[0, 1] / var_x if var_x > 0 else 0.0
        
        # Calculate reduced variance vectors
        y_control_cupac = y_control - theta * (x_control_pre - np.mean(x_all))
        y_treatment_cupac = y_treatment - theta * (x_treatment_pre - np.mean(x_all))
        
        # Perform comparative standard analysis over Cupac-adjusted variance maps
        results = ExperimentEvaluator.evaluate_standard_ab(y_control_cupac, y_treatment_cupac)
        results["variance_reduction_percentage"] = float((np.var(y_all) - np.var(np.concatenate([y_control_cupac, y_treatment_cupac]))) / np.var(y_all) * 100)
        
        return results
