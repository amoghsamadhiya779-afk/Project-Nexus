#!/usr/bin/env python3
"""
=============================================================================
Nexus MLOps - Dagster Orchestration Pipeline
Defines the Directed Acyclic Graph (DAG) for automated model retraining.
Pipeline: Extract Features -> Train Two-Tower -> Train Ranker -> Evaluate.
=============================================================================
"""

import os
import time
import mlflow
from dagster import asset, define_asset_job, Definitions, AssetExecutionContext

# Set MLflow tracking URI for the pipeline
os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"

@asset(group_name="data_engineering")
def feature_store_snapshot(context: AssetExecutionContext) -> str:
    """Extracts a point-in-time correct snapshot from the Postgres/Parquet offline store."""
    context.log.info("Connecting to Nexus Feature Store...")
    # Simulating data extraction
    time.sleep(2) 
    snapshot_path = "C:/data/features/offline/historical_interactions.parquet"
    context.log.info(f"✅ Extracted 1,000,000 feature rows to {snapshot_path}")
    return snapshot_path

@asset(group_name="model_training", deps=[feature_store_snapshot])
def candidate_generation_model(context: AssetExecutionContext) -> dict:
    """Trains the PyTorch Two-Tower Model using Ray (Simulated)."""
    context.log.info("Provisioning Ray cluster for distributed training...")
    time.sleep(3) # Simulating GPU training time
    
    # Log to MLflow
    mlflow.set_experiment("recommender_two_tower")
    with mlflow.start_run(run_name="dagster_automated_run"):
        mlflow.log_param("architecture", "two_tower")
        mlflow.log_metric("info_nce_loss", 0.0124)
        
    context.log.info("✅ Two-Tower Model training complete.")
    return {"status": "success", "loss": 0.0124, "path": "models_export/two_tower.pt"}

@asset(group_name="model_training", deps=[feature_store_snapshot, candidate_generation_model])
def multi_task_ranker_model(context: AssetExecutionContext) -> dict:
    """Trains the PyTorch MMoE DCN-v2 Ranker."""
    context.log.info("Initializing Multi-gate Mixture-of-Experts training...")
    time.sleep(3)
    
    mlflow.set_experiment("recommender_mmoe_ranker")
    with mlflow.start_run(run_name="dagster_automated_run"):
        mlflow.log_param("architecture", "mmoe_dcn_v2")
        mlflow.log_metric("ctr_auc", 0.891)
        mlflow.log_metric("cvr_auc", 0.842)
        
    context.log.info("✅ MMoE Ranker training complete.")
    return {"status": "success", "ctr_auc": 0.891, "path": "models_export/mmoe_ranker.pt"}

@asset(group_name="model_evaluation", deps=[multi_task_ranker_model])
def model_registry_promotion(context: AssetExecutionContext):
    """Evaluates the new model against the current production model."""
    context.log.info("Running offline evaluation comparison...")
    # Simulate an evaluation pass
    new_model_auc = 0.891
    current_prod_auc = 0.875
    
    if new_model_auc > current_prod_auc:
        context.log.info(f"🚀 New model outperformed production ({new_model_auc} > {current_prod_auc}).")
        context.log.info("Tagging model for Shadow Deployment in MLflow Registry.")
    else:
        context.log.warning("⚠️ New model underperformed. Halting deployment pipeline.")

# Define the overarching job that runs these assets sequentially
recsys_training_job = define_asset_job(name="nightly_recsys_retrain_job", selection="*")

# Expose definitions to the Dagster UI
defs = Definitions(
    assets=[
        feature_store_snapshot,
        candidate_generation_model,
        multi_task_ranker_model,
        model_registry_promotion
    ],
    jobs=[recsys_training_job],
)