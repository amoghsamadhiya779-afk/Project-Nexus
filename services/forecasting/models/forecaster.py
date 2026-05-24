"""
services/forecasting/models/forecaster.py
==========================================
Multi-horizon causal forecasting for supply/demand.

PLACE AT: nexus/services/forecasting/models/forecaster.py

Models:
  1. N-BEATS  — neural basis expansion (Oreshkin et al. 2019)
  2. Ensemble — N-BEATS + ARIMA + seasonal decomposition

Inspired by Uber's ML forecasting platform (M4 competition winner approach)
and DoorDash's demand forecasting for delivery time estimates.

Horizons: 1h, 6h, 24h, 7d, 30d (multi-horizon, single model)
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


# ─── N-BEATS Block ────────────────────────────────────────────────────────────

class NBEATSBlock(nn.Module):
    """
    Single N-BEATS block: FC stack + basis expansion.

    Each block produces:
    - backcast: prediction of the input window (for residual connection)
    - forecast: prediction of the output horizon

    Blocks are stacked in a doubly residual fashion:
    input to next block = input - backcast (remove explained variance)
    final forecast = sum of all block forecasts
    """

    def __init__(
        self,
        input_len:   int,
        output_len:  int,
        hidden_dim:  int = 256,
        n_layers:    int = 4,
        basis_type:  str = "generic",   # generic | trend | seasonality
        n_harmonics: int = 1,           # for seasonality basis
        n_polynomials: int = 2,         # for trend basis
    ):
        super().__init__()
        self.input_len  = input_len
        self.output_len = output_len
        self.basis_type = basis_type

        # Fully connected stack
        layers = [nn.Linear(input_len, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        self.fc = nn.Sequential(*layers)

        # Basis functions
        if basis_type == "generic":
            self.theta_b = nn.Linear(hidden_dim, input_len)    # backcast coefs
            self.theta_f = nn.Linear(hidden_dim, output_len)   # forecast coefs
        elif basis_type == "trend":
            p = n_polynomials + 1
            self.theta_b = nn.Linear(hidden_dim, p)
            self.theta_f = nn.Linear(hidden_dim, p)
            t_b = torch.arange(input_len)  / input_len
            t_f = torch.arange(output_len) / output_len
            self.register_buffer("T_b", torch.stack([t_b ** i for i in range(p)], dim=0))  # (p, T)
            self.register_buffer("T_f", torch.stack([t_f ** i for i in range(p)], dim=0))
        elif basis_type == "seasonality":
            k = n_harmonics
            self.theta_b = nn.Linear(hidden_dim, 4 * k)
            self.theta_f = nn.Linear(hidden_dim, 4 * k)
            t_b = 2 * np.pi * torch.arange(input_len)  / input_len
            t_f = 2 * np.pi * torch.arange(output_len) / output_len
            freqs = torch.arange(1, k + 1).float()
            cos_b = torch.cos(freqs.unsqueeze(1) * t_b.unsqueeze(0))   # (k, T)
            sin_b = torch.sin(freqs.unsqueeze(1) * t_b.unsqueeze(0))
            cos_f = torch.cos(freqs.unsqueeze(1) * t_f.unsqueeze(0))
            sin_f = torch.sin(freqs.unsqueeze(1) * t_f.unsqueeze(0))
            self.register_buffer("B_b", torch.cat([cos_b, sin_b], dim=0))  # (2k, T)
            self.register_buffer("B_f", torch.cat([cos_f, sin_f], dim=0))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.fc(x)  # (B, hidden)

        if self.basis_type == "generic":
            backcast = self.theta_b(h)
            forecast = self.theta_f(h)
        elif self.basis_type == "trend":
            theta_b  = self.theta_b(h).unsqueeze(-1)  # (B, p, 1)
            theta_f  = self.theta_f(h).unsqueeze(-1)
            backcast = (theta_b * self.T_b.unsqueeze(0)).sum(dim=1)
            forecast = (theta_f * self.T_f.unsqueeze(0)).sum(dim=1)
        elif self.basis_type == "seasonality":
            theta_b  = self.theta_b(h)  # (B, 4k)
            theta_f  = self.theta_f(h)
            k        = self.B_b.shape[0] // 2
            backcast = (theta_b[:, :k].unsqueeze(-1) * self.B_b[:k].unsqueeze(0) +
                        theta_b[:, k:].unsqueeze(-1) * self.B_b[k:].unsqueeze(0)).sum(1)
            forecast = (theta_f[:, :k].unsqueeze(-1) * self.B_f[:k].unsqueeze(0) +
                        theta_f[:, k:].unsqueeze(-1) * self.B_f[k:].unsqueeze(0)).sum(1)

        return backcast, forecast


class NBEATSModel(nn.Module):
    """
    Full N-BEATS model: stacked blocks with doubly residual connections.

    Stack architecture: [generic, trend, seasonality] × n_stacks
    This is the "interpretable" N-BEATS configuration.
    """

    def __init__(
        self,
        input_len:  int,
        output_len: int,
        hidden_dim: int = 256,
        n_stacks:   int = 30,
        n_layers:   int = 4,
    ):
        super().__init__()
        self.input_len  = input_len
        self.output_len = output_len

        # Alternate between basis types for interpretability
        basis_cycle = ["generic", "trend", "seasonality"]
        self.blocks = nn.ModuleList([
            NBEATSBlock(
                input_len, output_len,
                hidden_dim=hidden_dim,
                n_layers=n_layers,
                basis_type=basis_cycle[i % 3],
            )
            for i in range(n_stacks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, input_len) → returns (B, output_len) forecast"""
        residual = x
        forecast = torch.zeros(x.size(0), self.output_len, device=x.device)

        for block in self.blocks:
            backcast, block_forecast = block(residual)
            residual = residual - backcast   # doubly residual
            forecast = forecast + block_forecast

        return forecast


# ─── Forecasting Service ──────────────────────────────────────────────────────

class MarketplaceForecaster:
    """
    Multi-horizon demand forecasting service.

    Forecasts:
    - Item view count (next 1h, 6h, 24h, 7d)
    - Category demand (next 24h, 7d)
    - User session volume (next 1h, 6h)

    Training data: event time series from offline store.
    Inference: triggered every hour via Dagster.
    """

    HORIZONS = {
        "1h":  1,
        "6h":  6,
        "24h": 24,
        "7d":  24 * 7,
    }

    def __init__(
        self,
        input_len:  int = 168,  # 1 week of hourly history
        hidden_dim: int = 256,
        n_stacks:   int = 20,
    ):
        self.input_len = input_len
        # One model per horizon
        self.models: Dict[str, NBEATSModel] = {}
        for name, horizon in self.HORIZONS.items():
            self.models[name] = NBEATSModel(
                input_len=input_len,
                output_len=horizon,
                hidden_dim=hidden_dim,
                n_stacks=n_stacks,
            )

    def prepare_training_data(
        self,
        events_df:  pd.DataFrame,
        group_col:  str = "category",        # or "item_id"
        value_col:  str = "event_count",
        freq:       str = "1H",
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Build hourly time series from event log.
        Returns {entity_id: (X, y)} for each group.
        """
        ts = (
            events_df
            .set_index("event_timestamp")
            .groupby([group_col, pd.Grouper(freq=freq)])
            .size()
            .rename(value_col)
            .reset_index()
        )

        datasets = {}
        for entity, group in ts.groupby(group_col):
            series = group[value_col].values.astype(np.float32)
            if len(series) < self.input_len + max(self.HORIZONS.values()):
                continue  # not enough history

            # Normalise
            mu, sigma = series.mean(), series.std() + 1e-8
            series_norm = (series - mu) / sigma

            # Build sliding windows
            X_list, y_dict = [], {h: [] for h in self.HORIZONS}
            max_h = max(self.HORIZONS.values())
            for i in range(len(series_norm) - self.input_len - max_h):
                X_list.append(series_norm[i: i + self.input_len])
                for name, h in self.HORIZONS.items():
                    y_dict[name].append(
                        series_norm[i + self.input_len: i + self.input_len + h]
                    )

            if X_list:
                datasets[str(entity)] = {
                    "X":    np.array(X_list),
                    "y":    {k: np.array(v) for k, v in y_dict.items()},
                    "mu":   mu,
                    "sigma": sigma,
                }

        return datasets

    def train(
        self,
        datasets:   Dict[str, Any],
        horizon:    str = "24h",
        epochs:     int = 50,
        batch_size: int = 64,
        lr:         float = 1e-3,
    ) -> Dict[str, float]:
        """Train the model for a specific horizon."""
        import torch.optim as optim

        model  = self.models[horizon]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model  = model.to(device)
        opt    = optim.Adam(model.parameters(), lr=lr)

        # Collect all windows
        all_X, all_y = [], []
        for entity_data in datasets.values():
            all_X.append(entity_data["X"])
            all_y.append(entity_data["y"][horizon])

        if not all_X:
            logger.warning(f"No training data for horizon {horizon}")
            return {}

        X = torch.tensor(np.vstack(all_X), dtype=torch.float32)
        y = torch.tensor(np.vstack(all_y), dtype=torch.float32)

        n = len(X)
        losses = []

        model.train()
        for epoch in range(epochs):
            perm  = torch.randperm(n)
            epoch_loss = 0.0

            for i in range(0, n, batch_size):
                idx     = perm[i: i + batch_size]
                xb, yb  = X[idx].to(device), y[idx].to(device)
                pred    = model(xb)
                loss    = F.mse_loss(pred, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                epoch_loss += loss.item()

            if epoch % 10 == 0:
                logger.info(f"Forecaster {horizon} epoch {epoch}: loss={epoch_loss:.4f}")
            losses.append(epoch_loss)

        return {"final_loss": losses[-1], "horizon": horizon}

    def predict(
        self,
        entity_id:   str,
        history:     np.ndarray,   # (input_len,) last known values
        horizon:     str = "24h",
    ) -> np.ndarray:
        """
        Generate forecast for entity over the given horizon.
        Returns unnormalised prediction array.
        """
        model  = self.models[horizon]
        device = next(model.parameters()).device
        model.eval()

        # Normalise input
        mu, sigma = history.mean(), history.std() + 1e-8
        x_norm = (history[-self.input_len:] - mu) / sigma
        x_tensor = torch.tensor(x_norm, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            pred_norm = model(x_tensor).squeeze(0).cpu().numpy()

        # Denormalise
        return pred_norm * sigma + mu

    def save(self, path: str) -> None:
        import os
        os.makedirs(path, exist_ok=True)
        for name, model in self.models.items():
            torch.save(model.state_dict(), f"{path}/forecaster_{name}.pt")
        logger.info(f"Forecaster models saved to {path}")

    def load(self, path: str) -> None:
        for name, model in self.models.items():
            model_path = f"{path}/forecaster_{name}.pt"
            try:
                model.load_state_dict(torch.load(model_path, map_location="cpu"))
            except FileNotFoundError:
                logger.warning(f"Model not found: {model_path}")