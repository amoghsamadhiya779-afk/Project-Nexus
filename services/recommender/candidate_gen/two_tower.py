import torch
import torch.nn as nn
import numpy as np

class TowerNetwork(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        
    def forward(self, x):
        return self.net(x)

class TwoTowerModel(nn.Module):
    def __init__(self, user_dim: int, item_dim: int, embedding_dim: int = 64, temperature: float = 0.07):
        super().__init__()
        self.user_tower = TowerNetwork(user_dim, embedding_dim)
        self.item_tower = TowerNetwork(item_dim, embedding_dim)
        self.temperature = temperature
        
    def forward(self, user_features, item_features):
        user_emb = self.user_tower(user_features)
        item_emb = self.item_tower(item_features)
        
        # Normalize embeddings for cosine similarity evaluation
        user_emb = nn.functional.normalize(user_emb, p=2, dim=1)
        item_emb = nn.functional.normalize(item_emb, p=2, dim=1)
        return user_emb, item_emb
