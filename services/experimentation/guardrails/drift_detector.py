#!/usr/bin/env python3
"""
=============================================================================
Model Drift Detector (Population Stability Index)
Calculates the PSI between the training data distribution and the live 
production data distribution. If PSI > 0.2, it mathematically triggers an 
automated retraining pipeline (simulating Dagster/Airflow).
=============================================================================
"""

import numpy as np

class DriftDetector:
    @staticmethod
    def calculate_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
        """
        Calculates the Population Stability Index (PSI).
        Rule of thumb:
        PSI < 0.1: No significant population change
        PSI < 0.2: Moderate population change
        PSI >= 0.2: Significant population change (TRIGGER RETRAIN)
        """
        # Define bin edges based on the expected (training) distribution percentiles
        breakpoints = np.linspace(0, 100, buckets + 1)
        bins = np.percentile(expected, breakpoints)
        
        # Add slight adjustments to bin edges to include all data
        bins[0] -= 0.0001
        bins[-1] += 0.0001
        
        # Count occurrences in each bin
        expected_counts, _ = np.histogram(expected, bins=bins)
        actual_counts, _ = np.histogram(actual, bins=bins)
        
        # Convert to percentages and avoid divide-by-zero errors
        expected_pct = np.clip(expected_counts / len(expected), 0.0001, 1.0)
        actual_pct = np.clip(actual_counts / len(actual), 0.0001, 1.0)
        
        # PSI Formula: Sum((Actual% - Expected%) * ln(Actual% / Expected%))
        psi_values = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
        psi = np.sum(psi_values)
        
        return float(psi)

if __name__ == "__main__":
    print("[*] Running Model Drift Detection (Population Stability Index)...")
    
    # 1. Base training data (Expected Distribution)
    np.random.seed(42)
    training_data_feature = np.random.normal(loc=100.0, scale=15.0, size=5000)
    
    # 2. Live production data (Actual Distribution)
    # Scenario A: No Drift (Distribution is the same)
    live_data_stable = np.random.normal(loc=101.0, scale=15.5, size=2000)
    
    # Scenario B: Severe Drift (User behavior changed dramatically!)
    live_data_drifted = np.random.normal(loc=115.0, scale=20.0, size=2000)
    
    psi_stable = DriftDetector.calculate_psi(training_data_feature, live_data_stable)
    psi_drifted = DriftDetector.calculate_psi(training_data_feature, live_data_drifted)
    
    print("\n--- Scenario A: Stable Production Data ---")
    print(f"Calculated PSI: {psi_stable:.4f}")
    if psi_stable >= 0.2:
        print("🚨 DRIFT DETECTED: Triggering Dagster Retraining Pipeline!")
    else:
        print("✅ Status: Stable. No retraining required.")
        
    print("\n--- Scenario B: Shifted Production Data (User Behavior Changed) ---")
    print(f"Calculated PSI: {psi_drifted:.4f}")
    if psi_drifted >= 0.2:
        print("🚨 DRIFT DETECTED: Triggering automated Ray/Dagster Retraining Pipeline!")
    else:
        print("✅ Status: Stable. No retraining required.")