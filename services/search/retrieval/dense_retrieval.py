#!/usr/bin/env python3
"""
=============================================================================
Semantic Search & LambdaMART LTR Ingestion Engine
Self-contained NumPy-based Approximate Nearest Neighbor (ANN) search and 
Learning-to-Rank (LTR) decision-tree evaluations using LightGBM.
=============================================================================
"""

import os
import numpy as np
from typing import List, Dict, Any, Tuple
from pathlib import Path

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

class DenseRetrievalEngine:
    def __init__(self):
        self.index_path = Path("models_export/faiss_index_meta.npz")
        self.ltr_path = Path("models_export/ltr_lambdamart_model.txt")
        
        self.item_ids: List[str] = []
        self.embeddings: np.ndarray = np.empty((0, 16), dtype=np.float32)
        self.ltr_model = None
        
        self._load_vector_index()
        self._load_ltr_booster()

    def _load_vector_index(self):
        if self.index_path.exists():
            try:
                data = np.load(self.index_path)
                self.item_ids = list(data["ids"])
                self.embeddings = data["vectors"]
                print(f"[+] Loaded {len(self.item_ids)} indexed item vectors successfully.")
            except Exception as e:
                print(f"[⚠️ Indexer Error] Failed to parse NPZ file structures: {e}")
        else:
            print("[⚠️ Search Warning] Pre-compiled vector index not found. Run /index/refresh endpoint.")

    def build_index(self, vectors: np.ndarray, item_ids: List[str]):
        """Builds and serializes a local nearest-neighbor search index."""
        self.embeddings = vectors
        self.item_ids = item_ids
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        # Normalize vectors for fast cosine similarity dot products
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        self.embeddings = np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms!=0)
        np.savez(self.index_path, ids=np.array(item_ids), vectors=self.embeddings)

    def _load_ltr_booster(self):
        if lgb and self.ltr_path.exists():
            try:
                self.ltr_model = lgb.Booster(model_file=str(self.ltr_path))
                print(f"[+] LightGBM LambdaMART LTR model loaded successfully.")
            except Exception as e:
                print(f"[⚠️ LTR Error] Failed to compile LightGBM booster: {e}")
        else:
            print("[⚠️ LTR Warning] LightGBM binary missing or uninstalled. Using math fallback.")

    def retrieve_candidates(self, user_emb: np.ndarray, k: int = 50) -> Tuple[List[str], List[float]]:
        """Queries nearest neighbor candidate vectors via fast Matrix Multiplication."""
        if len(self.item_ids) == 0:
            fallback_items = [f"item_{i}" for i in range(1, k + 1)]
            fallback_scores = np.linspace(0.9, 0.1, k).tolist()
            return fallback_items, fallback_scores
            
        # Normalize query vector
        q_norm = user_emb / (np.linalg.norm(user_emb) + 1e-9)
        
        # Fast Cosine Similarity (Dot Product over normalized arrays)
        similarities = np.dot(self.embeddings, q_norm)
        
        # Extract Top-K indices
        top_indices = np.argsort(similarities)[::-1][:k]
        
        return [self.item_ids[idx] for idx in top_indices], [float(similarities[idx]) for idx in top_indices]

    def rescore_with_ltr(self, user_features: Dict[str, Any], item_features_list: List[Dict[str, Any]]) -> List[Tuple[str, float]]:
        """Re-ranks candidates using the LambdaMART Learning-to-Rank tree ensemble."""
        if not item_features_list:
            return []

        features_matrix = []
        user_view_cnt = float(user_features.get("user_view_count", 0))
        user_purchase_cnt = float(user_features.get("user_purchase_count", 0))
        user_cvr = float(user_features.get("user_conversion_rate", 0.0))

        for item in item_features_list:
            price = float(item.get("base_price", 0.0))
            popularity = float(item.get("popularity", 0))
            
            features_matrix.append([
                user_view_cnt, user_purchase_cnt, user_cvr,
                price, popularity,
                price * user_cvr, popularity * user_view_cnt,
                0.0, 0.0, 0.0
            ])

        features_matrix = np.array(features_matrix, dtype=np.float32)

        if not self.ltr_model:
            # Deterministic linear fallback for LTR scoring if LightGBM is missing
            scores = (features_matrix[:, 4] * 0.1) + (features_matrix[:, 2] * 2.0) - (features_matrix[:, 3] * 0.001)
            scores = scores.tolist()
        else:
            scores = self.ltr_model.predict(features_matrix).tolist()

        scored_items = [(item["item_id"], float(score)) for item, score in zip(item_features_list, scores)]
        return sorted(scored_items, key=lambda x: x[1], reverse=True)