"""
services/recommender/training/train.py
========================================
Training entrypoint for two-tower + multi-task ranker.

PLACE AT: nexus/services/recommender/training/train.py

Run:
    python -m services.recommender.training.train
    OR
    make train-recommender
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import mlflow
import numpy as np
import torch
import torch.nn as nn
import typer
from loguru import logger
from torch.utils.data import DataLoader, random_split

from services.recommender.candidate_gen.two_tower import (
    TwoTowerModel, FAISSItemIndex
)
from services.recommender.ranking.ranker import MultiTaskRanker, RankerConfig
from services.recommender.training.dataset import TwoTowerDataset, RankerDataset
from shared.utils.config import settings

app = typer.Typer()


# ─── Two-Tower Training ───────────────────────────────────────────────────────

def train_two_tower(
    offline_path: str  = "/data/features/offline",
    output_path:  str  = "/data/models/two_tower",
    epochs:       int  = 10,
    batch_size:   int  = 1024,
    lr:           float= 1e-3,
    embedding_dim:int  = 64,
    output_dim:   int  = 128,
    max_users:    int  = 100_001,
    max_items:    int  = 500_001,
) -> str:
    """Train two-tower model and build FAISS index. Returns model path."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training two-tower on {device}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset = TwoTowerDataset(
        events_path=f"{offline_path}/events",
        users_path =f"{offline_path}/users/users.parquet",
        items_path =f"{offline_path}/items/items.parquet",
        event_types=["purchase", "click", "add_to_cart"],
        max_user_id=max_users,
        max_item_id=max_items,
    )

    n_val   = max(1, int(len(dataset) * 0.1))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = TwoTowerModel(
        num_users=max_users,
        num_items=max_items,
        user_dense_dim=dataset.user_dense_dim,
        item_dense_dim=dataset.item_dense_dim,
        embedding_dim=embedding_dim,
        output_dim=output_dim,
        temperature=0.07,
    ).to(device)

    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=epochs, eta_min=lr * 0.1
    )

    # ── MLflow run ────────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("nexus-two-tower")

    with mlflow.start_run(run_name=f"two_tower_e{epochs}_b{batch_size}"):
        mlflow.log_params({
            "epochs":       epochs,
            "batch_size":   batch_size,
            "lr":           lr,
            "embedding_dim":embedding_dim,
            "output_dim":   output_dim,
            "n_train":      n_train,
            "n_val":        n_val,
            "device":       str(device),
        })

        best_val_loss = float("inf")
        best_model_path = None

        for epoch in range(1, epochs + 1):
            # ── Train ──────────────────────────────────────────────────────────
            model.train()
            train_losses = []
            t0 = time.time()

            for batch in train_loader:
                user_ids    = batch["user_id"].to(device)
                item_ids    = batch["item_id"].to(device)
                user_dense  = batch["user_dense"].to(device)
                item_dense  = batch["item_dense"].to(device)
                history_ids = batch["history_ids"].to(device)

                user_emb, item_emb = model(
                    user_ids, user_dense,
                    item_ids, item_dense,
                    history_ids=history_ids,
                )
                loss = model.compute_loss(user_emb, item_emb)

                optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimiser.step()

                train_losses.append(loss.item())

            scheduler.step()

            # ── Validate ───────────────────────────────────────────────────────
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    user_emb, item_emb = model(
                        batch["user_id"].to(device),
                        batch["user_dense"].to(device),
                        batch["item_id"].to(device),
                        batch["item_dense"].to(device),
                        history_ids=batch["history_ids"].to(device),
                    )
                    val_losses.append(model.compute_loss(user_emb, item_emb).item())

            train_loss = np.mean(train_losses)
            val_loss   = np.mean(val_losses)
            elapsed    = time.time() - t0

            logger.info(
                f"Epoch {epoch:2d}/{epochs} | "
                f"train={train_loss:.4f} | val={val_loss:.4f} | "
                f"lr={scheduler.get_last_lr()[0]:.2e} | {elapsed:.1f}s"
            )
            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss":   val_loss,
                "lr":         scheduler.get_last_lr()[0],
            }, step=epoch)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                Path(output_path).mkdir(parents=True, exist_ok=True)
                best_model_path = f"{output_path}/best_model.pt"
                torch.save({
                    "epoch":      epoch,
                    "state_dict": model.state_dict(),
                    "val_loss":   val_loss,
                    "config": {
                        "num_users": max_users, "num_items": max_items,
                        "user_dense_dim": dataset.user_dense_dim,
                        "item_dense_dim": dataset.item_dense_dim,
                        "embedding_dim": embedding_dim, "output_dim": output_dim,
                    }
                }, best_model_path)

        mlflow.log_metric("best_val_loss", best_val_loss)
        mlflow.log_artifact(best_model_path)

        # ── Build FAISS Item Index ─────────────────────────────────────────────
        logger.info("Building FAISS item index from trained embeddings...")
        _build_faiss_index(model, dataset, device, output_path, max_items)

        run_id = mlflow.active_run().info.run_id
        logger.info(f"Training complete. MLflow run: {run_id}")
        return best_model_path


def _build_faiss_index(
    model:       TwoTowerModel,
    dataset:     TwoTowerDataset,
    device:      torch.device,
    output_path: str,
    max_items:   int,
    batch_size:  int = 4096,
) -> None:
    """
    Generate embeddings for all items and build FAISS HNSW index.
    This index is used at serving time for ANN candidate retrieval.
    """
    model.eval()
    items_path = f"{dataset._offline_path if hasattr(dataset, '_offline_path') else '/data/features/offline'}/items/items.parquet"

    try:
        import pandas as pd
        items_df = pd.read_parquet(items_path)
    except Exception:
        logger.warning("Could not load items for FAISS index building")
        return

    item_ids  = items_df["item_id"].values.tolist()
    n_items   = len(item_ids)
    all_embeds = []

    item_dense_cols = dataset.item_dense_cols
    for col in item_dense_cols:
        col_max = items_df[col].max()
        if col_max > 0:
            items_df[col] = items_df[col] / col_max

    with torch.no_grad():
        for start in range(0, n_items, batch_size):
            batch_items  = items_df.iloc[start:start + batch_size]
            ids_tensor   = torch.tensor(
                [i % max_items for i in batch_items["item_id"].values],
                dtype=torch.long
            ).to(device)
            dense_tensor = torch.tensor(
                batch_items[item_dense_cols].fillna(0).values.astype(np.float32)
            ).to(device)

            emb = model.item_tower(ids_tensor, dense_tensor).cpu().numpy()
            all_embeds.append(emb)

    embeddings = np.vstack(all_embeds).astype(np.float32)

    index = FAISSItemIndex(dim=model.item_tower.output_dim)
    index.build(item_ids, embeddings)
    index.save(output_path)
    logger.info(f"FAISS index saved to {output_path}")


# ─── Ranker Training ──────────────────────────────────────────────────────────

def train_ranker(
    offline_path: str  = "/data/features/offline",
    output_path:  str  = "/data/models/ranker",
    epochs:       int  = 5,
    batch_size:   int  = 2048,
    lr:           float= 3e-4,
) -> str:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training multi-task ranker on {device}")

    dataset = RankerDataset(
        events_path=f"{offline_path}/events",
        users_path =f"{offline_path}/users/users.parquet",
        items_path =f"{offline_path}/items/items.parquet",
    )

    n_val = max(1, int(len(dataset) * 0.1))
    train_ds, val_ds = random_split(dataset, [len(dataset) - n_val, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size * 2, shuffle=False)

    config = RankerConfig(input_dim=128)
    model  = MultiTaskRanker(config).to(device)
    optim  = torch.optim.AdamW(model.parameters(), lr=lr)

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("nexus-ranker")

    with mlflow.start_run(run_name=f"ranker_e{epochs}"):
        mlflow.log_params({"epochs": epochs, "batch_size": batch_size, "lr": lr})
        best_loss = float("inf")

        for epoch in range(1, epochs + 1):
            model.train()
            losses = []
            for batch in train_loader:
                features  = batch["features"].to(device)
                positions = batch["position"].to(device)
                labels    = {
                    "ctr":           batch["label_ctr"].to(device),
                    "cvr":           batch["label_cvr"].to(device),
                    "response_rate": batch["label_response"].to(device),
                    "price_score":   batch["label_price"].to(device),
                }
                task_out = model(features, positions)
                loss     = model.compute_loss(task_out, labels)
                optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                losses.append(loss.item())

            train_loss = np.mean(losses)
            mlflow.log_metric("ranker_train_loss", train_loss, step=epoch)
            logger.info(f"Ranker epoch {epoch}/{epochs} | loss={train_loss:.4f}")

            if train_loss < best_loss:
                best_loss = train_loss
                Path(output_path).mkdir(parents=True, exist_ok=True)
                out = f"{output_path}/ranker.pt"
                torch.save(model.state_dict(), out)

        mlflow.log_metric("ranker_best_loss", best_loss)
        logger.info(f"Ranker training complete. Best loss: {best_loss:.4f}")
        return f"{output_path}/ranker.pt"


# ─── CLI Entrypoint ───────────────────────────────────────────────────────────

@app.command()
def main(
    offline_path:  str = typer.Option("/data/features/offline", "--offline-path"),
    output_path:   str = typer.Option("/data/models",           "--output-path"),
    epochs:        int = typer.Option(10,                       "--epochs"),
    batch_size:    int = typer.Option(1024,                     "--batch-size"),
    lr:           float= typer.Option(1e-3,                     "--lr"),
    skip_ranker:  bool = typer.Option(False,                    "--skip-ranker"),
):
    """Train two-tower retrieval + multi-task ranker."""

    logger.info("=== Phase 2: Model Training ===")

    logger.info("--- Two-Tower Training ---")
    model_path = train_two_tower(
        offline_path=offline_path,
        output_path=f"{output_path}/two_tower",
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
    )
    logger.info(f"Two-tower saved: {model_path}")

    if not skip_ranker:
        logger.info("--- Ranker Training ---")
        ranker_path = train_ranker(
            offline_path=offline_path,
            output_path=f"{output_path}/ranker",
            epochs=max(3, epochs // 2),
            batch_size=batch_size * 2,
        )
        logger.info(f"Ranker saved: {ranker_path}")

    logger.info("=== Phase 2 Complete ===")


if __name__ == "__main__":
    app()