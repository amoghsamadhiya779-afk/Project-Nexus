"""
services/search/ltr/ltr_model.py
==================================
Learning-to-Rank with LightGBM LambdaMART + diversity re-ranking.

PLACE AT: nexus/services/search/ltr/ltr_model.py

Architecture:
  Stage 1: Dense retrieval (bi-encoder) → top-500 candidates
  Stage 2: LambdaMART LTR → rerank top-500 → top-50
  Stage 3: MMR diversity re-ranking → final top-20

Inspired by:
  - "LinkedIn Learning-to-Rank Signals" (2019)
  - "Semantic Re-Ranking with LTR at Airbnb" (2019)
  - "Improving Search Ranking at DoorDash" (2022)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler


# ─── Feature Engineering ──────────────────────────────────────────────────────

LTR_FEATURES = [
    # Query-item relevance signals
    "bm25_score",              # sparse BM25 text match
    "embedding_cosine_sim",    # dense embedding similarity
    "title_exact_match",       # exact query term match in title
    "category_match",          # query category = item category

    # Item quality signals
    "item_avg_rating",
    "item_review_count_log",   # log-transformed
    "item_sale_count_7d_log",
    "item_view_count_7d_log",
    "item_price",
    "item_is_promoted",

    # Personalisation signals (user×item)
    "user_item_embedding_dot",
    "user_category_affinity",  # user's historical preference for item's category
    "user_price_bucket_match", # user's price range = item price range

    # Context signals
    "position_bias_correction",  # inverse propensity weight
    "query_popularity",          # how common this query is
    "item_recency_score",        # freshness
]


def build_ltr_features(
    query: str,
    candidates: pd.DataFrame,    # item_id, plus all item feature columns
    user_features: Dict[str, Any],
    embedding_scores: np.ndarray, # cosine similarities from dense retrieval
    bm25_scores:      np.ndarray, # BM25 scores from sparse retrieval
) -> np.ndarray:
    """
    Build feature matrix for LTR model.
    Returns (n_candidates, n_features) float32 array.
    """
    n = len(candidates)
    features = np.zeros((n, len(LTR_FEATURES)), dtype=np.float32)

    # Query-item signals
    features[:, 0] = bm25_scores
    features[:, 1] = embedding_scores
    features[:, 2] = candidates.get("title_exact_match", pd.Series(np.zeros(n))).values
    features[:, 3] = candidates.get("category_match",    pd.Series(np.zeros(n))).values

    # Item quality
    features[:, 4] = candidates.get("avg_rating",     pd.Series(np.zeros(n))).values
    features[:, 5] = np.log1p(candidates.get("review_count", pd.Series(np.zeros(n))).values)
    features[:, 6] = np.log1p(candidates.get("item_sale_count_7d", pd.Series(np.zeros(n))).values)
    features[:, 7] = np.log1p(candidates.get("item_view_count_7d", pd.Series(np.zeros(n))).values)
    features[:, 8] = np.log1p(candidates.get("price", pd.Series(np.zeros(n))).values)
    features[:, 9] = candidates.get("is_promoted", pd.Series(np.zeros(n))).values.astype(float)

    # Personalisation
    features[:, 10] = user_features.get("embedding_dot_scores", np.zeros(n))
    features[:, 11] = user_features.get("category_affinity",    np.zeros(n))
    features[:, 12] = user_features.get("price_bucket_match",   np.zeros(n))

    # Context
    features[:, 13] = 1.0 / np.arange(1, n + 1)     # position prior
    features[:, 14] = user_features.get("query_popularity", 0.5)
    features[:, 15] = candidates.get("recency_score", pd.Series(np.zeros(n))).values

    return features


# ─── LTR Model ────────────────────────────────────────────────────────────────

class LTRModel:
    """
    LightGBM LambdaMART Learning-to-Rank model.

    LambdaMART is the most widely deployed LTR algorithm in industry:
    Microsoft, LinkedIn, Airbnb, and many others use it as their
    primary ranking model.

    Why LightGBM over XGBoost for LTR:
    - Native LambdaMART with group-aware training
    - Faster training on large datasets
    - Better memory efficiency
    """

    def __init__(self):
        self._model: Optional[lgb.Booster] = None
        self._scaler = StandardScaler()
        self._feature_names = LTR_FEATURES

    def fit(
        self,
        X_train:      np.ndarray,    # (n_samples, n_features)
        y_train:      np.ndarray,    # relevance labels (0, 1, 2, 3, 4)
        groups_train: np.ndarray,    # query group sizes
        X_val:        np.ndarray,
        y_val:        np.ndarray,
        groups_val:   np.ndarray,
        n_estimators: int   = 500,
        lr:           float = 0.05,
        max_depth:    int   = 6,
        num_leaves:   int   = 63,
    ) -> Dict[str, Any]:
        """
        Train LambdaMART model with NDCG optimisation.
        """
        X_train = self._scaler.fit_transform(X_train)
        X_val   = self._scaler.transform(X_val)

        train_dataset = lgb.Dataset(
            X_train, label=y_train, group=groups_train,
            feature_name=self._feature_names,
        )
        val_dataset = lgb.Dataset(
            X_val, label=y_val, group=groups_val,
            reference=train_dataset,
        )

        params = {
            "objective":        "lambdarank",
            "metric":           "ndcg",
            "ndcg_eval_at":     [5, 10, 20],
            "n_estimators":     n_estimators,
            "learning_rate":    lr,
            "max_depth":        max_depth,
            "num_leaves":       num_leaves,
            "min_child_samples":20,
            "subsample":        0.8,
            "colsample_bytree": 0.8,
            "reg_alpha":        0.1,
            "reg_lambda":       0.1,
            "n_jobs":           -1,
            "verbose":          -1,
            "lambdarank_truncation_level": 20,
        }

        callbacks = [
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=50),
        ]

        logger.info("Training LambdaMART LTR model...")
        self._model = lgb.train(
            params,
            train_dataset,
            valid_sets=[val_dataset],
            callbacks=callbacks,
        )

        # Feature importance
        importance = dict(zip(
            self._feature_names,
            self._model.feature_importance(importance_type="gain").tolist()
        ))
        top_features = sorted(importance.items(), key=lambda x: -x[1])[:5]
        logger.info(f"Top LTR features: {top_features}")

        return {
            "best_iteration": self._model.best_iteration,
            "feature_importance": importance,
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return relevance scores for ranking."""
        if self._model is None:
            raise RuntimeError("Model not trained")
        X_scaled = self._scaler.transform(X)
        return self._model.predict(X_scaled, num_iteration=self._model.best_iteration)

    def rank(
        self,
        query:      str,
        candidates: pd.DataFrame,
        user_feats: Dict[str, Any],
        emb_scores: np.ndarray,
        bm25_scores: np.ndarray,
        top_k:      int = 50,
    ) -> pd.DataFrame:
        """
        Full ranking pipeline: feature build → LTR score → sort.
        """
        features = build_ltr_features(
            query, candidates, user_feats, emb_scores, bm25_scores
        )
        scores = self.predict(features)
        candidates = candidates.copy()
        candidates["ltr_score"] = scores
        return candidates.nlargest(top_k, "ltr_score").reset_index(drop=True)

    def save(self, path: str) -> None:
        if self._model:
            self._model.save_model(f"{path}/ltr_model.txt")
            import joblib
            joblib.dump(self._scaler, f"{path}/ltr_scaler.pkl")
        logger.info(f"LTR model saved to {path}")

    def load(self, path: str) -> None:
        self._model = lgb.Booster(model_file=f"{path}/ltr_model.txt")
        import joblib
        self._scaler = joblib.load(f"{path}/ltr_scaler.pkl")

    def log_to_mlflow(self, metrics: Dict[str, Any]) -> None:
        mlflow.lightgbm.log_model(self._model, "ltr_model")
        mlflow.log_metrics(metrics)


# ─── MMR Diversity Re-ranker ──────────────────────────────────────────────────

class MMRReranker:
    """
    Maximal Marginal Relevance (MMR) for result diversity.

    Prevents showing 20 very similar items — balances relevance
    with diversity. Critical for marketplace UX where users want
    to see a variety of price points, sellers, and conditions.

    "The Use of MMR, Diversity-Based Reranking for Reordering
     Documents and Producing Summaries" — Carbonell & Goldstein, 1998.
    """

    def __init__(self, lambda_: float = 0.5):
        """lambda_: 0=max diversity, 1=max relevance, 0.5=balanced"""
        self.lambda_ = lambda_

    def rerank(
        self,
        candidates:  pd.DataFrame,      # with ltr_score and item_embedding columns
        score_col:   str = "ltr_score",
        embed_col:   str = "item_embedding",
        top_k:       int = 20,
    ) -> pd.DataFrame:
        """
        Greedily select items that maximise MMR score.
        """
        if len(candidates) <= top_k:
            return candidates

        scores     = candidates[score_col].values
        embeddings = np.vstack(candidates[embed_col].values)

        # L2 normalise for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)

        selected_indices: List[int] = []
        remaining        = list(range(len(candidates)))

        while len(selected_indices) < top_k and remaining:
            if not selected_indices:
                # Pick the highest-scored item first
                best_idx = max(remaining, key=lambda i: scores[i])
            else:
                # MMR: relevance - lambda * max similarity to already selected
                selected_embs = embeddings[selected_indices]

                best_mmr  = -float("inf")
                best_idx  = remaining[0]

                for idx in remaining:
                    relevance  = scores[idx]
                    sim_scores = embeddings[idx] @ selected_embs.T
                    max_sim    = sim_scores.max()

                    mmr = self.lambda_ * relevance - (1 - self.lambda_) * max_sim
                    if mmr > best_mmr:
                        best_mmr = mmr
                        best_idx = idx

            selected_indices.append(best_idx)
            remaining.remove(best_idx)

        return candidates.iloc[selected_indices].reset_index(drop=True)