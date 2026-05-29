#!/usr/bin/env python3
"""
=============================================================================
Sequential Testing Guardrails
Monitors running A/B tests continuously. Implements early-stopping boundaries
to kill experiments that are severely degrading business metrics (e.g., crashes, 
revenue drop) before reaching full sample size.
=============================================================================
"""

import numpy as np

class SequentialGuardrail:
    """
    Implements a basic sequential probability ratio test (SPRT) wrapper.
    Evaluates cumulative data dynamically to halt bad experiments early.
    """
    def __init__(self, critical_z_score: float = -2.58): # approx p < 0.01 threshold
        self.critical_z_score = critical_z_score

    def evaluate_live_stream(self, control_stream: np.ndarray, treatment_stream: np.ndarray) -> str:
        """
        Evaluates streams to see if treatment is performing disastrously.
        Returns "CONTINUE", "STOP_DEGRADATION", or "STOP_SUCCESS".
        """
        if len(control_stream) < 50 or len(treatment_stream) < 50:
            return "CONTINUE (Gathering Data)"
            
        mean_c = np.mean(control_stream)
        mean_t = np.mean(treatment_stream)
        
        # Calculate standard Z-score
        pooled_se = np.sqrt(np.var(control_stream)/len(control_stream) + np.var(treatment_stream)/len(treatment_stream))
        if pooled_se == 0:
            return "CONTINUE"
            
        z_score = (mean_t - mean_c) / pooled_se
        
        if z_score <= self.critical_z_score:
            return f"🚨 STOP_DEGRADATION (Z-Score: {z_score:.2f} | Metric is tanking!)"
        elif z_score >= abs(self.critical_z_score):
            return f"🎉 STOP_SUCCESS (Z-Score: {z_score:.2f} | Clear Winner!)"
            
        return f"✅ CONTINUE (Z-Score: {z_score:.2f} | Within normal variance)"

if __name__ == "__main__":
    guardrail = SequentialGuardrail()
    
    # Simulate an experiment where the new model is breaking the UI and tanking clicks
    np.random.seed(42)
    control = np.random.normal(5.0, 1.0, 200)
    treatment_disaster = np.random.normal(4.2, 1.0, 200) # Significant drop!
    
    print("[*] Running Live Guardrail Monitor on Degrading Treatment Stream...")
    for step in [50, 100, 150, 200]:
        decision = guardrail.evaluate_live_stream(control[:step], treatment_disaster[:step])
        print(f"    Batch Size {step}: {decision}")