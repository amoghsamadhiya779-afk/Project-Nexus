#!/usr/bin/env python3
"""
=============================================================================
Nexus MLflow Model Registry Automator
Scans the MLflow tracking server for the best-performing models (highest AUC),
registers them as versioned artifacts, and officially transitions their 
stage to 'Production' for the FastAPI serving layer to ingest.
=============================================================================
"""

import mlflow
from mlflow.tracking import MlflowClient

# Initialize connection to the local Dockerized MLflow server
MLFLOW_URI = "http://localhost:5000"
mlflow.set_tracking_uri(MLFLOW_URI)
client = MlflowClient(tracking_uri=MLFLOW_URI)

def promote_best_model(experiment_name: str, metric_name: str, model_name: str):
    """Finds the best run in an experiment and promotes it to Production."""
    print(f"[*] Scanning MLflow Experiment: '{experiment_name}' for best '{metric_name}'...")
    
    try:
        experiment = client.get_experiment_by_name(experiment_name)
        if not experiment:
            print(f"[❌] Experiment '{experiment_name}' not found.")
            return

        # Query all runs in the experiment, order by the target metric descending
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric_name} DESC"],
            max_results=1
        )

        if not runs:
            print(f"[⚠️] No runs found in experiment '{experiment_name}'.")
            return

        best_run = runs[0]
        best_metric_value = best_run.data.metrics.get(metric_name)
        run_id = best_run.info.run_id
        
        print(f"[+] Found Best Run ID: {run_id} | {metric_name}: {best_metric_value}")

        # In a real environment, you would log the PyTorch model artifact directly.
        # Since we use localized files, we will create a registry entry to track the metadata.
        print(f"[*] Registering run as Model: '{model_name}'...")
        
        # Check if registered model exists, if not create it
        try:
            client.create_registered_model(model_name)
        except Exception:
            pass # Model already exists
            
        # Create a new version
        model_version = client.create_model_version(
            name=model_name,
            source=f"runs:/{run_id}/model",
            run_id=run_id
        )
        
        version_num = model_version.version
        print(f"[+] Created Version {version_num} of '{model_name}'.")

        # Transition to Production
        print(f"[*] Transitioning Version {version_num} to 'Production'...")
        client.transition_model_version_stage(
            name=model_name,
            version=version_num,
            stage="Production",
            archive_existing_versions=True
        )
        
        print(f"🚀 SUCCESS: '{model_name}' v{version_num} is now active in Production!")
        
    except Exception as e:
        print(f"[❌] MLflow Connection Failed. Is the Docker container running? Error: {e}")

if __name__ == "__main__":
    print("=========================================================")
    print("       NEXUS MLOPS: MODEL PROMOTION AUTOMATION           ")
    print("=========================================================\n")
    
    promote_best_model(
        experiment_name="recommender_mmoe_ranker",
        metric_name="ctr_auc",
        model_name="Nexus_MMoE_Ranker"
    )