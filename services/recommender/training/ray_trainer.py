#!/usr/bin/env python3
"""
=============================================================================
Nexus MLOps: Ray Distributed Training
Scales the PyTorch Two-Tower candidate generation model across a multi-node 
GPU cluster using Ray Train and PyTorch DDP (Distributed Data Parallel).
=============================================================================
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import ray
from ray import train
from ray.train import Checkpoint
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig

from services.recommender.candidate_gen.two_tower import TwoTowerModel

def train_func(config):
    """The training loop executed on each distributed Ray worker."""
    batch_size = config.get("batch_size", 256)
    epochs = config.get("epochs", 5)
    
    # 1. Initialize Ray's distributed PyTorch wrapper
    train.torch.prepare_model
    train.torch.prepare_data_loader
    
    # 2. Setup Model
    user_feats = {"user_id": 500, "user_city": 20, "user_segment": 5}
    item_feats = {"item_id": 1000, "item_category": 15, "merchant_id": 100}
    
    model = TwoTowerModel(
        user_embeddings_map=user_feats, item_embeddings_map=item_feats,
        embedding_dim=16, user_dense_dim=4, item_dense_dim=4, projection_dim=16
    )
    
    # Prepare model for Distributed Data Parallel (DDP)
    model = train.torch.prepare_model(model)
    optimizer = optim.Adam(model.parameters(), lr=config.get("lr", 0.001))
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Simulated DDP Sharded Batch
        device = train.torch.get_device()
        sparse_users = {k: torch.randint(0, v, (batch_size,), device=device) for k, v in user_feats.items()}
        dense_users = torch.randn(batch_size, 4, device=device)
        sparse_items = {k: torch.randint(0, v, (batch_size,), device=device) for k, v in item_feats.items()}
        dense_items = torch.randn(batch_size, 4, device=device)
        
        user_emb, item_emb = model(sparse_users, dense_users, sparse_items, dense_items)
        scores = torch.matmul(user_emb, item_emb.T) / 0.07
        labels = torch.arange(batch_size, device=device)
        
        loss = criterion(scores, labels)
        loss.backward()
        optimizer.step()
        
        # Report metrics back to the Ray Head Node
        train.report({"loss": loss.item(), "epoch": epoch})

if __name__ == "__main__":
    print("[*] Initializing Ray Distributed Training Cluster...")
    ray.init(ignore_reinit_error=True)
    
    scaling_config = ScalingConfig(
        num_workers=2,          # Number of distributed workers
        use_gpu=False           # Set to True if running on AWS g4dn instances
    )
    
    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config={"lr": 0.001, "batch_size": 128, "epochs": 5},
        scaling_config=scaling_config,
    )
    
    print("[*] Executing PyTorch DDP Training via Ray...")
    result = trainer.fit()
    print(f"[+] Ray Training Complete! Final Loss: {result.metrics['loss']:.4f}")