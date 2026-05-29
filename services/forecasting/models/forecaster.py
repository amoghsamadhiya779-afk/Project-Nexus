#!/usr/bin/env python3
"""
=============================================================================
Causal Multi-Horizon Demand Forecasting Model (N-BEATS Block Architecture)
Implements double residual mappings with polynomial trend and Fourier series 
seasonality solvers. Resilient against MLflow offline tracking states.
=============================================================================
"""

import os
import socket
import argparse
from urllib.parse import urlparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import mlflow
from typing import Tuple

from shared.utils.config import config

class NBeatsBlock(nn.Module):
    """
    N-BEATS structural basis projection block.
    Splits backcast outputs into polynomial trend and Fourier cyclical seasonality curves.
    """
    def __init__(self, backcast_len: int, forecast_len: int, hidden_dim: int = 128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(backcast_len, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        # Degree-2 polynomial coefficients projection for Trend
        self.theta_b_trend = nn.Linear(hidden_dim, 3)
        self.theta_f_trend = nn.Linear(hidden_dim, 3)
        
        # Sinusoidal Fourier parameters projection for Seasonality
        self.theta_b_season = nn.Linear(hidden_dim, 4)
        self.theta_f_season = nn.Linear(hidden_dim, 4)
        
        self.backcast_len = backcast_len
        self.forecast_len = forecast_len

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.fc(x)
        
        # 1. Polynomial Trend Calculations
        t_b = torch.linspace(-1, 1, self.backcast_len, device=x.device).unsqueeze(0)
        t_f = torch.linspace(0, 2, self.forecast_len, device=x.device).unsqueeze(0)
        
        v_b = torch.stack([t_b**0, t_b**1, t_b**2], dim=-1) # Vandermonde backcast representation
        v_f = torch.stack([t_f**0, t_f**1, t_f**2], dim=-1) # Vandermonde forecast representation
        
        coef_b_trend = self.theta_b_trend(h).unsqueeze(1)
        coef_f_trend = self.theta_f_trend(h).unsqueeze(1)
        
        backcast_trend = torch.sum(coef_b_trend * v_b, dim=-1)
        forecast_trend = torch.sum(coef_f_trend * v_f, dim=-1)
        
        # 2. Seasonality Series Solver
        cycle_period = 12.0
        f_b = torch.stack([
            torch.cos(2 * np.pi * t_b * 1 / cycle_period), torch.sin(2 * np.pi * t_b * 1 / cycle_period),
            torch.cos(2 * np.pi * t_b * 2 / cycle_period), torch.sin(2 * np.pi * t_b * 2 / cycle_period)
        ], dim=-1)
        
        f_f = torch.stack([
            torch.cos(2 * np.pi * t_f * 1 / cycle_period), torch.sin(2 * np.pi * t_f * 1 / cycle_period),
            torch.cos(2 * np.pi * t_f * 2 / cycle_period), torch.sin(2 * np.pi * t_f * 2 / cycle_period)
        ], dim=-1)
        
        coef_b_season = self.theta_b_season(h).unsqueeze(1)
        coef_f_season = self.theta_f_season(h).unsqueeze(1)
        
        backcast_season = torch.sum(coef_b_season * f_b, dim=-1)
        forecast_season = torch.sum(coef_f_season * f_f, dim=-1)
        
        return backcast_trend + backcast_season, forecast_trend + forecast_season


class NexusForecaster(nn.Module):
    """Deep Forecasting stack utilizing residual connections and sequential blocks."""
    def __init__(self, backcast_len: int = 24, forecast_len: int = 6, num_blocks: int = 2):
        super().__init__()
        self.backcast_len = backcast_len
        self.forecast_len = forecast_len
        self.blocks = nn.ModuleList([NBeatsBlock(backcast_len, forecast_len) for _ in range(num_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        forecast_total = torch.zeros(x.shape[0], self.forecast_len, device=x.device)
        residual = x
        
        for block in self.blocks:
            backcast, forecast = block(residual)
            residual = residual - backcast
            forecast_total = forecast_total + forecast
            
        return forecast_total


def is_mlflow_active(tracking_uri: str) -> bool:
    try:
        parsed = urlparse(tracking_uri)
        host = parsed.hostname or "localhost"
        port = parsed.port or 5000
        with socket.create_connection((host, port), timeout=1.5):
            return True
    except Exception:
        return False


def train_forecasting(epochs: int = 10, batch_size: int = 256, lr: float = 0.001, mlflow_uri: str = "http://localhost:5000"):
    print("\n" + "="*80)
    print("      TRAINING CAUSAL MULTI-HORIZON TIME-SERIES FORECASTING MODEL      ")
    print("="*80)
    
    mlflow_online = is_mlflow_active(mlflow_uri)
    
    backcast_len, forecast_len = 24, 6
    model = NexusForecaster(backcast_len, forecast_len)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    if mlflow_online:
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("forecasting_demand")
        mlflow.start_run(run_name="nbeats_training_run")
        mlflow.log_params({
            "learning_rate": lr,
            "batch_size": batch_size,
            "backcast_len": backcast_len,
            "forecast_len": forecast_len
        })
        print("[⚡ MLflow Ingest] Connected successfully. Logging metrics online...")
    else:
        print("[⚠️ MLflow Offline] Tracking server not responding. Operating in local offline mode.")

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Simulate base timeseries profiles
        t = np.linspace(0, 100, batch_size)
        noise = np.random.normal(0, 0.1, (batch_size, backcast_len))
        series = np.sin(t[:, None] * 0.1) + (t[:, None] * 0.01) + noise
        
        target_t = t[:, None] + np.arange(1, forecast_len + 1) * 0.1
        targets = np.sin(target_t * 0.1) + (target_t * 0.01) + np.random.normal(0, 0.1, (batch_size, forecast_len))
        
        x_tensor = torch.tensor(series, dtype=torch.float32, device=device)
        y_tensor = torch.tensor(targets, dtype=torch.float32, device=device)
        
        preds = model(x_tensor)
        loss = loss_fn(preds, y_tensor)
        
        loss.backward()
        optimizer.step()
        
        # Compute mean absolute percentage error (MAPE)
        mape = torch.mean(torch.abs((y_tensor - preds) / (y_tensor + 1e-5))) * 100.0
        
        loss_val = float(loss.item())
        mape_val = float(mape.item())
        
        if mlflow_online:
            mlflow.log_metric("mse_loss", loss_val, step=epoch)
            mlflow.log_metric("mape_metric", mape_val, step=epoch)
            
        if (epoch + 1) % 2 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:02d}/{epochs:02d} | MSE Loss: {loss_val:.6f} | MAPE: {mape_val:.2f}%")

    if mlflow_online:
        mlflow.end_run()

    out_dir = os.path.join(config.BASE_DATA_DIR, "models", "forecasting")
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, "demand_forecaster.pt"))
    print(f"[+] Forecasting model weights serialized to: {out_dir}/demand_forecaster.pt")


if __name__ == "__main__":
    train_forecasting()