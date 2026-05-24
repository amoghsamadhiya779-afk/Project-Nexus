"""
services/recommender/training/dataset.py
==========================================
PyTorch Dataset classes for two-tower and ranker training.

PLACE AT: nexus/services/recommender/training/dataset.py
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from loguru import logger


class TwoTowerDataset(Dataset):
    """
    Dataset for two-tower model training.

    Each sample: (user_id, user_dense, item_id, item_dense, history_ids)
    Label is implicit — the positive item is the one the user interacted with.
    Negatives are generated in-batch during training (no explicit negatives needed).

    Reads from Parquet offline store produced by Phase 1.
    """

    def __init__(
        self,
        events_path:    str,
        users_path:     str,
        items_path:     str,
        event_types:    List[str] = None,  # filter to these event types
        history_len:    int = 50,
        user_dense_cols: List[str] = None,
        item_dense_cols: List[str] = None,
        max_user_id:    int = 100_001,
        max_item_id:    int = 500_001,
    ):
        self.history_len  = history_len
        self.max_user_id  = max_user_id
        self.max_item_id  = max_item_id

        self.user_dense_cols = user_dense_cols or [
            "lifetime_value", "price_sensitivity", "activity_level"
        ]
        self.item_dense_cols = item_dense_cols or [
            "price", "popularity", "avg_rating", "review_count"
        ]

        # Load and merge
        logger.info("Loading datasets...")
        events = self._load_parquet(events_path)
        users  = self._load_parquet(users_path)
        items  = self._load_parquet(items_path)

        if event_types:
            events = events[events["event_type"].isin(event_types)]

        # Keep only purchase + click events for positive signal
        positive_events = events[
            events["event_type"].isin(["purchase", "click", "add_to_cart"])
        ].copy()

        # Merge user features
        positive_events = positive_events.merge(
            users[["user_id"] + self.user_dense_cols],
            on="user_id", how="left"
        )
        # Merge item features
        positive_events = positive_events.merge(
            items[["item_id"] + self.item_dense_cols],
            on="item_id", how="left"
        )

        # Fill NaN with 0
        positive_events[self.user_dense_cols] = (
            positive_events[self.user_dense_cols].fillna(0).clip(lower=0)
        )
        positive_events[self.item_dense_cols] = (
            positive_events[self.item_dense_cols].fillna(0).clip(lower=0)
        )

        # Normalise dense features
        for col in self.user_dense_cols + self.item_dense_cols:
            col_max = positive_events[col].max()
            if col_max > 0:
                positive_events[col] = positive_events[col] / col_max

        # Build user interaction history (for attention in user tower)
        logger.info("Building user interaction histories...")
        user_history = (
            events.sort_values("event_timestamp")
            .groupby("user_id")["item_id"]
            .apply(list)
            .to_dict()
        )

        self._data = positive_events.reset_index(drop=True)
        self._user_history = user_history

        logger.info(f"Dataset ready: {len(self._data):,} positive interactions")

    def _load_parquet(self, path: str) -> pd.DataFrame:
        p = Path(path)
        if p.is_dir():
            files = list(p.rglob("*.parquet"))
            if not files:
                logger.warning(f"No parquet files in {path}")
                return pd.DataFrame()
            dfs = [pd.read_parquet(f) for f in files[:50]]  # cap for dev
            return pd.concat(dfs, ignore_index=True)
        return pd.read_parquet(path)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self._data.iloc[idx]

        user_id = int(row["user_id"]) % self.max_user_id
        item_id = int(row["item_id"]) % self.max_item_id

        user_dense = torch.tensor(
            row[self.user_dense_cols].values.astype(np.float32),
            dtype=torch.float32
        )
        item_dense = torch.tensor(
            row[self.item_dense_cols].values.astype(np.float32),
            dtype=torch.float32
        )

        # User interaction history
        history = self._user_history.get(int(row["user_id"]), [])
        if len(history) > self.history_len:
            history = history[-self.history_len:]
        history_padded = (history + [0] * self.history_len)[:self.history_len]
        history_ids = torch.tensor(
            [h % self.max_item_id for h in history_padded],
            dtype=torch.long
        )

        return {
            "user_id":     torch.tensor(user_id,  dtype=torch.long),
            "item_id":     torch.tensor(item_id,  dtype=torch.long),
            "user_dense":  user_dense,
            "item_dense":  item_dense,
            "history_ids": history_ids,
        }

    @property
    def user_dense_dim(self) -> int:
        return len(self.user_dense_cols)

    @property
    def item_dense_dim(self) -> int:
        return len(self.item_dense_cols)


class RankerDataset(Dataset):
    """
    Dataset for the multi-task ranker.
    Labels: click (binary), purchase (binary), response_rate (binary), price_ok (binary).

    Each sample is a (user, item, context) triple with per-task labels.
    Includes negatives: randomly sampled items the user did NOT interact with.
    """

    def __init__(
        self,
        events_path: str,
        users_path:  str,
        items_path:  str,
        feature_dim: int = 128,
        neg_ratio:   int = 4,    # 4 negatives per positive
        max_item_id: int = 500_001,
    ):
        self.feature_dim = feature_dim
        self.neg_ratio   = neg_ratio
        self.max_item_id = max_item_id

        logger.info("Building ranker dataset...")

        events = self._load_parquet(events_path)
        users  = self._load_parquet(users_path)
        items  = self._load_parquet(items_path)

        self._item_ids = items["item_id"].values.tolist()

        # Build positive samples
        positives = events[events["event_type"].isin(["purchase", "click"])].copy()
        positives["label_ctr"]      = (positives["event_type"] == "click").astype(float)
        positives["label_cvr"]      = (positives["event_type"] == "purchase").astype(float)
        positives["label_response"] = np.random.beta(8, 2, len(positives))  # simulated
        positives["label_price"]    = np.random.beta(5, 2, len(positives))  # simulated
        positives["is_positive"]    = 1

        # Build negative samples (items user did NOT buy)
        neg_rows = []
        user_positives = positives.groupby("user_id")["item_id"].apply(set).to_dict()

        for user_id, pos_items in list(user_positives.items())[:10_000]:  # cap
            neg_pool = [i for i in self._item_ids if i not in pos_items]
            n_neg    = min(len(neg_pool), len(pos_items) * neg_ratio)
            neg_items = np.random.choice(neg_pool, size=n_neg, replace=False)
            for item_id in neg_items:
                neg_rows.append({
                    "user_id":          user_id,
                    "item_id":          item_id,
                    "label_ctr":        0.0,
                    "label_cvr":        0.0,
                    "label_response":   0.0,
                    "label_price":      0.0,
                    "is_positive":      0,
                })

        negatives = pd.DataFrame(neg_rows)
        self._data = pd.concat([positives, negatives], ignore_index=True).sample(frac=1)

        logger.info(
            f"Ranker dataset: {positives['is_positive'].sum():,} positives, "
            f"{len(negatives):,} negatives"
        )

    def _load_parquet(self, path: str) -> pd.DataFrame:
        p = Path(path)
        if p.is_dir():
            files = list(p.rglob("*.parquet"))[:20]
            return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        return pd.read_parquet(path)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self._data.iloc[idx]
        # Simulate feature vector (in production: fetched from feature store)
        features = torch.randn(self.feature_dim)
        return {
            "features":       features,
            "position":       torch.tensor(int(np.random.randint(0, 20)), dtype=torch.long),
            "label_ctr":      torch.tensor(row["label_ctr"],      dtype=torch.float32),
            "label_cvr":      torch.tensor(row["label_cvr"],      dtype=torch.float32),
            "label_response": torch.tensor(row["label_response"], dtype=torch.float32),
            "label_price":    torch.tensor(row["label_price"],    dtype=torch.float32),
        }