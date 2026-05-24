"""
services/fraud_detection/models/fraud_model.py
================================================
Ensemble fraud/anomaly detection.

PLACE AT: nexus/services/fraud_detection/models/fraud_model.py

Approach (inspired by Grab's graph fraud and Uber's RADAR):
  Layer 1: Tabular features → XGBoost binary classifier
  Layer 2: Graph features  → GraphSAGE (see graph/graph_builder.py)
  Layer 3: Ensemble        → weighted average of both scores
  Layer 4: HITL queue      → scores above threshold sent for human review

Features:
  - Velocity: events/minute, distinct IPs, transaction amounts
  - Behavioural: session length, click patterns, device fingerprint
  - Graph: degree, clustering, PageRank, community membership
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    logger.warning("XGBoost not installed; using sklearn fallback")


FRAUD_FEATURES = [
    # Velocity features (last 1h window)
    "events_per_minute_1h",
    "distinct_items_viewed_1h",
    "distinct_sellers_contacted_1h",
    "purchase_count_1h",
    "failed_payment_count_1h",

    # Behavioural features
    "session_duration_seconds",
    "avg_time_between_events",
    "is_new_account",           # account age < 7 days
    "is_unverified",
    "device_changed_recently",

    # Amount features
    "total_spend_1h",
    "max_single_transaction",
    "price_deviation_from_category_median",

    # Network/graph features (populated by graph module)
    "graph_degree",
    "graph_pagerank",
    "graph_clustering_coef",
    "shared_device_with_flagged_user",
    "shared_ip_with_flagged_user",
]


@dataclass
class FraudPrediction:
    entity_id:    int
    entity_type:  str              # user | item | transaction
    fraud_score:  float            # 0–1
    is_flagged:   bool
    model_version: str
    top_signals:  List[Tuple[str, float]]  # (feature, importance)
    latency_ms:   float


class FraudDetector:
    """
    Real-time fraud scoring service.

    Scoring SLA: < 20ms p99 (synchronous, called in the request path)
    Batch scoring: runs hourly over all active users via Dagster
    """

    FLAG_THRESHOLD   = 0.70   # above this → send to HITL queue
    BLOCK_THRESHOLD  = 0.95   # above this → auto-block
    MODEL_VERSION    = "v1.0"

    def __init__(self):
        self._tabular_model = None
        self._feature_names = FRAUD_FEATURES

    def fit(
        self,
        X_train: np.ndarray,    # (n_samples, n_features)
        y_train: np.ndarray,    # 0=legit, 1=fraud
        X_val:   np.ndarray,
        y_val:   np.ndarray,
        scale_pos_weight: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Train XGBoost fraud classifier.
        scale_pos_weight handles class imbalance (fraud is rare).
        """
        # Estimate class weight if not provided
        if scale_pos_weight is None:
            n_legit = (y_train == 0).sum()
            n_fraud = (y_train == 1).sum()
            scale_pos_weight = n_legit / max(n_fraud, 1)
            logger.info(
                f"Fraud base rate: {n_fraud / len(y_train) * 100:.2f}% "
                f"→ scale_pos_weight={scale_pos_weight:.1f}"
            )

        if XGB_AVAILABLE:
            self._tabular_model = xgb.XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                eval_metric=["auc", "aucpr"],
                early_stopping_rounds=30,
                n_jobs=-1,
            )
            self._tabular_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=50,
            )
            val_auc = self._eval_auc(X_val, y_val)
        else:
            from sklearn.ensemble import GradientBoostingClassifier
            self._tabular_model = GradientBoostingClassifier(n_estimators=100)
            self._tabular_model.fit(X_train, y_train)
            val_auc = self._eval_auc(X_val, y_val)

        logger.info(f"Fraud model trained. Val AUC: {val_auc:.4f}")
        return {"val_auc": val_auc}

    def _eval_auc(self, X: np.ndarray, y: np.ndarray) -> float:
        from sklearn.metrics import roc_auc_score
        scores = self._tabular_model.predict_proba(X)[:, 1]
        return float(roc_auc_score(y, scores))

    def score(
        self,
        entity_id:   int,
        entity_type: str,
        features:    Dict[str, float],
        graph_score: float = 0.0,    # from GraphSAGE
        graph_weight: float = 0.3,
    ) -> FraudPrediction:
        """
        Score a single entity. Called synchronously in request path.
        """
        t0 = time.perf_counter()

        # Build feature vector
        x = np.array([
            features.get(f, 0.0) for f in self._feature_names
        ], dtype=np.float32).reshape(1, -1)

        if self._tabular_model is not None:
            tabular_score = float(
                self._tabular_model.predict_proba(x)[0, 1]
            )
        else:
            # Heuristic fallback (no trained model)
            tabular_score = min(
                features.get("events_per_minute_1h", 0) / 100 +
                features.get("failed_payment_count_1h", 0) * 0.2 +
                (0.4 if features.get("is_new_account") else 0) +
                features.get("shared_ip_with_flagged_user", 0) * 0.5,
                1.0
            )

        # Ensemble: tabular + graph
        fraud_score = (
            (1 - graph_weight) * tabular_score +
            graph_weight       * graph_score
        )

        # Top contributing features (for explainability)
        if XGB_AVAILABLE and hasattr(self._tabular_model, "feature_importances_"):
            importances = self._tabular_model.feature_importances_
            top_signals = sorted(
                zip(self._feature_names, importances * x.flatten()),
                key=lambda kv: -abs(kv[1])
            )[:5]
        else:
            top_signals = []

        latency_ms = (time.perf_counter() - t0) * 1000

        return FraudPrediction(
            entity_id    = entity_id,
            entity_type  = entity_type,
            fraud_score  = round(float(fraud_score), 4),
            is_flagged   = fraud_score >= self.FLAG_THRESHOLD,
            model_version= self.MODEL_VERSION,
            top_signals  = [(k, round(float(v), 4)) for k, v in top_signals],
            latency_ms   = round(latency_ms, 2),
        )

    def batch_score(
        self,
        entity_ids:   List[int],
        features_df:  pd.DataFrame,    # rows = entities, cols = FRAUD_FEATURES
        graph_scores: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Score a batch of entities (for hourly batch jobs).
        Returns DataFrame with fraud_score and is_flagged.
        """
        X = features_df[self._feature_names].fillna(0).values

        if self._tabular_model is not None:
            tabular_scores = self._tabular_model.predict_proba(X)[:, 1]
        else:
            tabular_scores = np.zeros(len(X))

        if graph_scores is None:
            graph_scores = np.zeros(len(X))

        fraud_scores = 0.7 * tabular_scores + 0.3 * graph_scores

        return pd.DataFrame({
            "entity_id":   entity_ids,
            "fraud_score": fraud_scores,
            "is_flagged":  fraud_scores >= self.FLAG_THRESHOLD,
            "auto_block":  fraud_scores >= self.BLOCK_THRESHOLD,
        })

    def save(self, path: str) -> None:
        import os
        os.makedirs(path, exist_ok=True)
        if XGB_AVAILABLE and self._tabular_model:
            self._tabular_model.save_model(f"{path}/fraud_xgb.json")
        else:
            import joblib
            joblib.dump(self._tabular_model, f"{path}/fraud_model.pkl")
        logger.info(f"Fraud model saved to {path}")

    def load(self, path: str) -> None:
        if XGB_AVAILABLE:
            self._tabular_model = xgb.XGBClassifier()
            self._tabular_model.load_model(f"{path}/fraud_xgb.json")
        else:
            import joblib
            self._tabular_model = joblib.load(f"{path}/fraud_model.pkl")