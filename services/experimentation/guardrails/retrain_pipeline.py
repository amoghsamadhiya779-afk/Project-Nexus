#!/usr/bin/env python3
"""
=============================================================================
Nexus MLOps - Dagster Orchestration Pipeline
Automates the retraining of the PyTorch Two-Tower and MMoE models when 
the Population Stability Index (PSI) drift detector fires an alert.
=============================================================================
"""

from dagster import asset, define_asset_job, Definitions, AssetExecutionContext
import os
import time

@asset(group_name="mlops_monitoring")
def drift_detection_alert(context: AssetExecutionContext) -> bool:
    """Simulates checking the PSI drift detector metrics from Prometheus/Redis."""
    context.log.info("Evaluating live production data vs training distribution...")
    # Simulate a drift trigger
    drift_detected = True 
    if drift_detected:
        context.log.warning("🚨 SEVERE DRIFT DETECTED (PSI > 0.20). Triggering retrain.")
    return drift_detected

@asset(group_name="model_training", deps=[drift_detection_alert])
def fetch_latest_feature_store_data(context: AssetExecutionContext) -> str:
    """Pulls the latest point-in-time correct data from the offline Parquet store."""
    context.log.info("Connecting to Offline Store (PostgreSQL/Parquet)...")
    time.sleep(1) # Simulating I/O
    dataset_path = "C:/data/features/offline/historical_interactions.parquet"
    context.log.info(f"Successfully extracted 1M+ rows to {dataset_path}")
    return dataset_path

@asset(group_name="model_training")
def train_two_tower_model(context: AssetExecutionContext, fetch_latest_feature_store_data: str) -> str:
    """Executes the Ray/PyTorch distributed training job for Candidate Generation."""
    context.log.info(f"Loading data from {fetch_latest_feature_store_data} into PyTorch DataLoader...")
    time.sleep(2) # Simulating GPU training epochs
    model_path = "models_export/two_tower_v2.pt"
    context.log.info(f"✅ Two-Tower Retraining Complete. Model exported to {model_path}")
    return model_path

@asset(group_name="model_training")
def train_mmoe_ranker_model(context: AssetExecutionContext, fetch_latest_feature_store_data: str) -> str:
    """Executes the PyTorch training job for the MMoE Multi-Task Ranker."""
    context.log.info(f"Loading data from {fetch_latest_feature_store_data} into PyTorch DataLoader...")
    time.sleep(2) # Simulating GPU training epochs
    model_path = "models_export/mmoe_ranker_v2.pt"
    context.log.info(f"✅ MMoE Ranker Retraining Complete. Model exported to {model_path}")
    return model_path

@asset(group_name="model_deployment", deps=[train_two_tower_model, train_mmoe_ranker_model])
def deploy_to_shadow_mode(context: AssetExecutionContext):
    """Pushes the newly trained models to the Triton/FastAPI serving gateway in Shadow Mode."""
    context.log.info("Registering new model weights in MLflow Model Registry...")
    context.log.info("Routing 5% of live traffic to new models via Interleaving Engine for validation.")
    context.log.info("🚀 CI/CD ML Deployment Successful.")

# Define the DAG Job that connects all these assets
retrain_job = define_asset_job(name="automated_retraining_pipeline", selection="*")

# Expose to Dagster framework
defs = Definitions(
    assets=[
        drift_detection_alert, 
        fetch_latest_feature_store_data, 
        train_two_tower_model, 
        train_mmoe_ranker_model, 
        deploy_to_shadow_mode
    ],
    jobs=[retrain_job],
)