#!/usr/bin/env python3
"""
=============================================================================
Unified Ensemble Fraud Detector (GraphSAGE + Autoencoder + Isolation Forest)
Combines Graph Neural Networks for structural network anomalies with Autoencoders 
for tabular anomalies. Modeled after Uber RADAR and Grab's Defense architectures.
=============================================================================
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
from typing import Tuple, Dict

class SimpleGraphSAGE(nn.Module):
    """Message passing Convolution neural block for node classification."""
    def __init__(self, feature_dim: int, hidden_dim: int = 16):
        super().__init__()
        self.w_self = nn.Linear(feature_dim, hidden_dim)
        self.w_neighbor = nn.Linear(feature_dim, hidden_dim)
        self.act = nn.ReLU()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, node_features: torch.Tensor, adjacency_list: Dict[int, list]) -> torch.Tensor:
        num_nodes = node_features.shape[0]
        agg_neighbor_feats = torch.zeros_like(node_features)
        
        for node_idx, neighbors in adjacency_list.items():
            if neighbors:
                agg_neighbor_feats[node_idx] = torch.mean(node_features[neighbors], dim=0)
                
        h_self = self.w_self(node_features)
        h_neigh = self.w_neighbor(agg_neighbor_feats)
        
        node_embeddings = self.act(h_self + h_neigh)
        return self.classifier(node_embeddings)

class FraudAutoencoder(nn.Module):
    """Autoencoder anomaly reconstruction network for individual event profiling."""
    def __init__(self, input_dim: int = 10, latent_dim: int = 3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 6), nn.ReLU(),
            nn.Linear(6, latent_dim), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 6), nn.ReLU(),
            nn.Linear(6, input_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

class UnifiedEnsembleFraudDetector:
    """Ensemble orchestrator for robust, low false-positive anomaly scoring."""
    def __init__(self, feature_dim: int = 10):
        self.gnn = SimpleGraphSAGE(feature_dim=feature_dim)
        self.autoencoder = FraudAutoencoder(input_dim=feature_dim)
        self.iforest = IsolationForest(contamination=0.01, random_state=42)
        
    def fit_ensemble(self, X_train: np.ndarray, adjacency: Dict[int, list], labels: np.ndarray, epochs: int = 5):
        print("[*] Executing Unified Anomaly and Fraud detection training loop...")
        optimizer_gnn = torch.optim.Adam(self.gnn.parameters(), lr=0.01)
        optimizer_ae = torch.optim.Adam(self.autoencoder.parameters(), lr=0.01)
        bce_loss, mse_loss = nn.BCELoss(), nn.MSELoss()
        
        self.iforest.fit(X_train)
        
        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(-1)
        
        for epoch in range(epochs):
            self.gnn.train()
            optimizer_gnn.zero_grad()
            loss_gnn = bce_loss(self.gnn(X_tensor, adjacency), y_tensor)
            loss_gnn.backward()
            optimizer_gnn.step()
            
            self.autoencoder.train()
            optimizer_ae.zero_grad()
            loss_ae = mse_loss(self.autoencoder(X_tensor), X_tensor)
            loss_ae.backward()
            optimizer_ae.step()
            
            if (epoch + 1) % 2 == 0:
                print(f"    Epoch {epoch+1}/{epochs} - GNN Loss: {loss_gnn.item():.4f} | AE Recon-Loss: {loss_ae.item():.4f}")

    def predict_risk_score(self, x: np.ndarray, neighbors_list: Dict[int, list]) -> Tuple[float, Dict[str, float]]:
        x_tensor = torch.tensor(x, dtype=torch.float32)
        self.gnn.eval()
        self.autoencoder.eval()
        
        with torch.no_grad():
            gnn_score = float(self.gnn(x_tensor, neighbors_list)[0].item())
            reconstructed = self.autoencoder(x_tensor)
            ae_error = float(torch.mean((x_tensor - reconstructed) ** 2).item())
            
        if_score = float(-self.iforest.score_samples(x)[0])
        consensus_score = (gnn_score * 0.40) + (min(ae_error, 1.0) * 0.35) + (min(if_score, 1.0) * 0.25)
        
        return consensus_score, {
            "gnn_risk": gnn_score,
            "autoencoder_reconstruction_error": ae_error,
            "isolation_forest_anomaly_score": if_score
        }

if __name__ == "__main__":
    np.random.seed(42)
    features = np.random.normal(0, 1, (100, 10))
    features[5] = np.random.normal(5, 1, 10) # Inject anomaly
    adj = {i: [(i-1)%100, (i+1)%100] for i in range(100)}
    lbls = np.zeros(100)
    lbls[5] = 1 
    
    detector = UnifiedEnsembleFraudDetector()
    detector.fit_ensemble(features, adj, lbls, epochs=4)
    score, breakdown = detector.predict_risk_score(features[5:6], {0: [4, 6]})
    print(f"[+] Consensus Node-Risk Evaluated: {score:.4f}")