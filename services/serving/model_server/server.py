#!/usr/bin/env python3
"""
=============================================================================
Unified Model Execution Server
Thread-safe loader for PyTorch model weights (Two-Tower and MMoE Ranker).
=============================================================================
"""

import torch
import numpy as np
from typing import List, Dict, Any, Tuple
from pathlib import Path
from services.recommender.candidate_gen.two_tower import TwoTowerModel
from services.recommender.ranking.mmoe_dcn import MMoEDCNRanker

class LocalModelServer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = Path("models_export")
        
        self.user_feats = {"user_id": 500, "user_city": 20, "user_segment": 5}
        self.item_feats = {"item_id": 1000, "item_category": 15, "merchant_id": 100}
        
        self.two_tower = None
        self.ranker = None
        
        self._load_two_tower_weights()
        self._load_mmoe_ranker_weights()

    def _load_two_tower_weights(self):
        path = self.model_dir / "two_tower.pt"
        if path.exists():
            try:
                model = TwoTowerModel(
                    user_embeddings_map=self.user_feats, item_embeddings_map=self.item_feats,
                    embedding_dim=16, user_dense_dim=4, item_dense_dim=4, projection_dim=16
                )
                model.load_state_dict(torch.load(path, map_location=self.device))
                model.to(self.device).eval()
                self.two_tower = model
                print("[+] Two-Tower candidate retriever initialized successfully.")
            except Exception as e:
                print(f"[❌ Error] Failed to compile Two-Tower state weights: {e}")

    def _load_mmoe_ranker_weights(self):
        path = self.model_dir / "mmoe_ranker.pt"
        if path.exists():
            try:
                combined_feats = {**self.user_feats, **self.item_feats}
                model = MMoEDCNRanker(
                    num_embeddings_map=combined_feats, embedding_dim=16, dense_dim=8,
                    cross_layers=2, low_rank=8, num_experts=4, expert_hidden=32
                )
                model.load_state_dict(torch.load(path, map_location=self.device))
                model.to(self.device).eval()
                self.ranker = model
                print("[+] MMoE Multi-Task Ranker initialized successfully.")
            except Exception as e:
                print(f"[❌ Error] Failed to compile MMoE Ranker state weights: {e}")

    def predict_user_embedding(self, user_features: Dict[str, Any]) -> np.ndarray:
        if not self.two_tower:
            np.random.seed(hash(user_features.get("user_id", "0")) % (2**32))
            return np.random.randn(16).astype(np.float32)

        sparse_in = {
            "user_id": torch.tensor([hash(user_features.get("user_id", "0")) % 500], device=self.device),
            "user_city": torch.tensor([hash(user_features.get("user_city", "0")) % 20], device=self.device),
            "user_segment": torch.tensor([hash(user_features.get("user_segment", "0")) % 5], device=self.device)
        }
        dense_in = torch.tensor([[
            float(user_features.get("user_view_count", 0)), float(user_features.get("user_purchase_count", 0)),
            float(user_features.get("user_conversion_rate", 0.0)), 0.0
        ]], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            return self.two_tower.user_tower(sparse_in, dense_in).cpu().numpy()[0]

    def score_ranking_batch(self, user_features: Dict[str, Any], item_features_list: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        batch_size = len(item_features_list)
        if batch_size == 0: return np.array([]), np.array([])
        
        if not self.ranker:
            ctr_scores, cvr_scores = [], []
            user_conv = float(user_features.get("user_conversion_rate", 0.05))
            for item in item_features_list:
                pop, price = float(item.get("popularity", 5)), float(item.get("base_price", 50.0))
                raw_ctr = (np.log(pop + 1) * 0.3) - (price * 0.001) + 0.5
                raw_cvr = raw_ctr * (0.1 + user_conv * 2.0)
                ctr_scores.append(1.0 / (1.0 + np.exp(-raw_ctr)))
                cvr_scores.append(1.0 / (1.0 + np.exp(-raw_cvr)))
            return np.array(ctr_scores), np.array(cvr_scores)

        sparse_batch = {
            "user_id": torch.full((batch_size,), hash(user_features.get("user_id", "0")) % 500, dtype=torch.long, device=self.device),
            "user_city": torch.full((batch_size,), hash(user_features.get("user_city", "0")) % 20, dtype=torch.long, device=self.device),
            "user_segment": torch.full((batch_size,), hash(user_features.get("user_segment", "0")) % 5, dtype=torch.long, device=self.device),
            "item_id": torch.zeros(batch_size, dtype=torch.long, device=self.device),
            "item_category": torch.zeros(batch_size, dtype=torch.long, device=self.device),
            "merchant_id": torch.zeros(batch_size, dtype=torch.long, device=self.device)
        }

        dense_user_list = [float(user_features.get("user_view_count", 0)), float(user_features.get("user_purchase_count", 0)), float(user_features.get("user_conversion_rate", 0.0)), 0.0]
        dense_batch_list = []
        for i, item in enumerate(item_features_list):
            sparse_batch["item_id"][i] = hash(item.get("item_id", "0")) % 1000
            sparse_batch["item_category"][i] = hash(item.get("category", "0")) % 15
            sparse_batch["merchant_id"][i] = hash(item.get("merchant_id", "0")) % 100
            dense_batch_list.append(dense_user_list + [float(item.get("popularity", 0)), float(item.get("base_price", 0.0)), 0.0, 0.0])

        with torch.no_grad():
            predictions = self.ranker(sparse_batch, torch.tensor(dense_batch_list, dtype=torch.float32, device=self.device))
            return predictions["ctr"].cpu().numpy().flatten(), predictions["cvr"].cpu().numpy().flatten()