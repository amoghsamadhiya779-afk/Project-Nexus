import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
from services.recommender.candidate_gen.two_tower import TwoTowerModel
from services.recommender.ranking.mmoe_dcn import MMoEDCNRanker

def train_two_tower(args):
    print("[*] Starting training loop for Recommender Candidate Retrieval Two-Tower neural network...")
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment("recommender_two_tower")
    
    with mlflow.start_run():
        # Instantiate model under tracked training sequence
        user_dim, item_dim = 16, 16
        model = TwoTowerModel(user_dim, item_dim, embedding_dim=32)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        
        mlflow.log_param("learning_rate", args.lr)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("epochs", args.epochs)

        # Iterate training epochs
        for epoch in range(args.epochs):
            # Generate mock structured representations matching active database dimensions
            mock_users = torch.randn(args.batch_size, user_dim)
            mock_items = torch.randn(args.batch_size, item_dim)
            
            model.train()
            optimizer.zero_grad()
            user_emb, item_emb = model(mock_users, mock_items)
            
            # InfoNCE contrastive evaluation matrix dot multiplication
            scores = torch.matmul(user_emb, item_emb.T) / model.temperature
            labels = torch.arange(args.batch_size, device=scores.device)
            loss = nn.CrossEntropyLoss()(scores, labels)
            
            loss.backward()
            optimizer.step()
            
            mlflow.log_metric("info_nce_loss", float(loss.item()), step=epoch)
            if (epoch + 1) % 2 == 0:
                print(f"    Epoch {epoch+1}/{args.epochs} - InfoNCE Loss: {loss.item():.4f}")

        # Serialize trained state tensors
        os.makedirs(args.output_dir, exist_ok=True)
        torch.save(model.user_tower.state_dict(), os.path.join(args.output_dir, "user_tower.pt"))
        torch.save(model.item_tower.state_dict(), os.path.join(args.output_dir, "item_tower.pt"))
        print(f"[+] Output state weights written to: {args.output_dir}")

def train_ranker(args):
    print("[*] Starting training loop for MMoE DCN-V2 Ranking models...")
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment("recommender_mmoe_ranker")
    
    with mlflow.start_run():
        input_dim = 24
        model = MMoEDCNRanker(input_dim=input_dim, num_experts=4, num_tasks=2)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        
        mlflow.log_param("learning_rate", args.lr)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("epochs", args.epochs)

        bce_loss = nn.BCELoss()
        
        for epoch in range(args.epochs):
            # Generate mock representations of joined entity logs
            x = torch.randn(args.batch_size, input_dim)
            ctr_labels = torch.randint(0, 2, (args.batch_size, 1)).float()
            cvr_labels = torch.randint(0, 2, (args.batch_size, 1)).float()
            
            model.train()
            optimizer.zero_grad()
            ctr_pred, cvr_pred = model(x)
            
            loss_ctr = bce_loss(ctr_pred, ctr_labels)
            loss_cvr = bce_loss(cvr_pred, cvr_labels)
            total_loss = loss_ctr + 0.5 * loss_cvr # Multi-objective relative weighted loss
            
            total_loss.backward()
            optimizer.step()
            
            mlflow.log_metric("ctr_bce_loss", float(loss_ctr.item()), step=epoch)
            mlflow.log_metric("cvr_bce_loss", float(loss_cvr.item()), step=epoch)
            mlflow.log_metric("joint_multi_task_loss", float(total_loss.item()), step=epoch)
            
            if (epoch + 1) % 2 == 0:
                print(f"    Epoch {epoch+1}/{args.epochs} - Joint multi-task loss: {total_loss.item():.4f}")

        os.makedirs(args.output_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(args.output_dir, "mmoe_dcn_model.pt"))
        print(f"[+] Multi-gate model binaries serialized to: {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", choices=["two_tower", "ranker"], required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--mlflow-uri", type=str, default="http://localhost:5000")
    parser.add_argument("--output-dir", type=str, required=True)
    
    args = parser.parse_args()
    if args.model_type == "two_tower":
        train_two_tower(args)
    else:
        train_ranker(args)
