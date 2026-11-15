"""PyTorch AutoEncoder anomaly detector.

The network learns to compress and rebuild "normal" rows. Rows it rebuilds badly
(high reconstruction error) are anomalies.

Score mapping: score = err / (err + threshold), so a row sitting exactly on the
percentile threshold scores 0.5, and `predict` is simply `score > 0.5`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from .base import BaseDetector, to_2d_array


class _AutoEncoderNet(nn.Module):
    """encoder -> bottleneck -> decoder.

    Tanh keeps hidden activations bounded, so a row far outside the training
    range cannot be reconstructed well (a ReLU net would just extrapolate).
    """

    def __init__(self, n_features: int, hidden_dim: int, bottleneck_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, n_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class AutoEncoderDetector(BaseDetector):
    name = "autoencoder"

    def __init__(
        self,
        hidden_dim: int | None = None,
        bottleneck_dim: int | None = None,
        epochs: int = 40,
        batch_size: int = 256,
        lr: float = 1e-3,
        threshold_percentile: float = 95.0,
        max_train_rows: int = 100_000,
        random_state: int = 42,
    ) -> None:
        super().__init__()
        if not 50 <= threshold_percentile < 100:
            raise ValueError("threshold_percentile must be in [50, 100)")
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.threshold_percentile = threshold_percentile
        self.max_train_rows = max_train_rows
        self.random_state = random_state

    # ------------------------------------------------------------------ fit
    def _build(self, n_features: int) -> None:
        hidden = self.hidden_dim or max(8, 2 * n_features)
        bottleneck = self.bottleneck_dim or max(1, n_features // 2)
        self.net_ = _AutoEncoderNet(n_features, hidden, bottleneck)

    def fit(self, X) -> "AutoEncoderDetector":
        arr = to_2d_array(X)
        torch.manual_seed(self.random_state)
        rng = np.random.default_rng(self.random_state)

        self.mean_ = arr.mean(axis=0)
        std = arr.std(axis=0)
        self.scale_ = np.where(std > 0, std, 1.0)
        scaled = ((arr - self.mean_) / self.scale_).astype(np.float32)

        train = scaled
        if len(train) > self.max_train_rows:
            train = train[rng.choice(len(train), self.max_train_rows, replace=False)]

        self._build(arr.shape[1])
        optimizer = torch.optim.Adam(self.net_.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        data = torch.from_numpy(train)

        self.loss_history_: list[float] = []
        self.net_.train()
        for _ in range(self.epochs):
            perm = torch.randperm(len(data))
            total = 0.0
            for i in range(0, len(data), self.batch_size):
                batch = data[perm[i:i + self.batch_size]]
                optimizer.zero_grad()
                loss = loss_fn(self.net_(batch), batch)
                loss.backward()
                optimizer.step()
                total += loss.item() * len(batch)
            self.loss_history_.append(total / len(data))

        self._fitted = True
        errors = self.reconstruction_error(arr)
        # Percentile cutoff on training errors; tiny floor avoids divide-by-zero.
        self.threshold_ = max(float(np.percentile(errors, self.threshold_percentile)), 1e-12)
        return self

    # ------------------------------------------------------------ inference
    def reconstruction_error(self, X) -> np.ndarray:
        """Mean squared reconstruction error per row, in standardised units."""
        self._check_fitted()
        arr = to_2d_array(X)
        scaled = torch.from_numpy(((arr - self.mean_) / self.scale_).astype(np.float32))
        self.net_.eval()
        with torch.no_grad():
            recon = self.net_(scaled)
        return ((recon - scaled) ** 2).mean(dim=1).numpy().astype(float)

    def score(self, X) -> np.ndarray:
        err = self.reconstruction_error(X)
        return err / (err + self.threshold_)

    def predict(self, X) -> np.ndarray:
        return self.reconstruction_error(X) > self.threshold_

    # --------------------------------------------------------- persistence
    def save(self, path: str | Path) -> None:
        """Save weights + everything needed to score new data (torch.save)."""
        self._check_fitted()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        first = self.net_.encoder[0]
        torch.save(
            {
                "state_dict": self.net_.state_dict(),
                "n_features": int(first.in_features),
                "hidden_dim": int(first.out_features),
                "bottleneck_dim": int(self.net_.encoder[2].out_features),
                "mean": self.mean_.tolist(),
                "scale": self.scale_.tolist(),
                "threshold": float(self.threshold_),
                "threshold_percentile": float(self.threshold_percentile),
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "AutoEncoderDetector":
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        det = cls(
            hidden_dim=ckpt["hidden_dim"],
            bottleneck_dim=ckpt["bottleneck_dim"],
            threshold_percentile=ckpt["threshold_percentile"],
        )
        det._build(ckpt["n_features"])
        det.net_.load_state_dict(ckpt["state_dict"])
        det.mean_ = np.asarray(ckpt["mean"])
        det.scale_ = np.asarray(ckpt["scale"])
        det.threshold_ = float(ckpt["threshold"])
        det.loss_history_ = []
        det._fitted = True
        return det
