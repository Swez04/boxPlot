"""Weighted-voting ensemble over the four detectors.

For every row, each detector casts a vote (anomaly / not) and gives a soft score.

    vote_share = sum(w_i * vote_i) / sum(w_i)
    soft_share = sum(w_i * score_i) / sum(w_i)
    confidence = 0.8 * vote_share + 0.2 * soft_share        (always in [0, 1])

Votes dominate, so "3 of 4 models agree" matters most; the soft share breaks ties
and ranks rows that have the same votes. A row is an anomaly when
confidence >= `threshold` (default 0.5).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from detectors import BaseDetector, DBSCANDetector, IQRDetector, IsolationForestDetector
from detectors.base import to_2d_array

DEFAULT_WEIGHTS = {"iqr": 1.0, "isolation_forest": 1.5, "dbscan": 1.0, "autoencoder": 1.5}
VOTE_MIX = 0.8


@dataclass
class EnsembleResult:
    confidence: pd.Series        # 0..1 per row
    is_anomaly: pd.Series        # bool per row
    votes: pd.DataFrame          # bool, one column per detector
    scores: pd.DataFrame         # 0..1, one column per detector

    @property
    def n_anomalies(self) -> int:
        return int(self.is_anomaly.sum())

    def top(self, n: int = 10) -> pd.Series:
        """Highest-confidence anomalies."""
        return self.confidence[self.is_anomaly].sort_values(ascending=False).head(n)


def default_detectors(random_state: int = 42, contamination: float = 0.05,
                      ae_epochs: int = 40) -> list[BaseDetector]:
    dets: list[BaseDetector] = [
        IQRDetector(),
        IsolationForestDetector(contamination=contamination, random_state=random_state),
        DBSCANDetector(),
    ]
    try:
        from detectors import AutoEncoderDetector
        dets.append(AutoEncoderDetector(epochs=ae_epochs, random_state=random_state))
    except ImportError:  # torch not installed -> 3-model ensemble
        pass
    return dets


class EnsembleDetector:
    def __init__(self, detectors: list[BaseDetector] | None = None,
                 weights: dict[str, float] | None = None, threshold: float = 0.5) -> None:
        self.detectors = detectors if detectors is not None else default_detectors()
        self.weights = {**DEFAULT_WEIGHTS, **(weights or {})}
        self.threshold = threshold
        if not self.detectors:
            raise ValueError("Need at least one detector")
        if not 0 < threshold <= 1:
            raise ValueError("threshold must be in (0, 1]")
        self._fitted = False

    def _w(self, det: BaseDetector) -> float:
        w = float(self.weights.get(det.name, 1.0))
        if w < 0:
            raise ValueError(f"Negative weight for {det.name}")
        return w

    def fit(self, X: pd.DataFrame) -> "EnsembleDetector":
        arr = to_2d_array(X)
        self.columns_ = list(X.columns) if isinstance(X, pd.DataFrame) else [f"x{i}" for i in range(arr.shape[1])]
        for det in self.detectors:
            det.fit(arr)
        self._fitted = True
        return self

    def detect(self, X: pd.DataFrame) -> EnsembleResult:
        if not self._fitted:
            raise RuntimeError("EnsembleDetector must be fitted before use")
        arr = to_2d_array(X)
        index = X.index if isinstance(X, (pd.DataFrame, pd.Series)) else pd.RangeIndex(len(arr))
        names = [d.name for d in self.detectors]
        votes = pd.DataFrame({d.name: d.predict(arr) for d in self.detectors}, index=index)
        scores = pd.DataFrame({d.name: d.score(arr) for d in self.detectors}, index=index)

        w = np.array([self._w(d) for d in self.detectors])
        if w.sum() == 0:
            raise ValueError("All weights are zero")
        w = w / w.sum()
        vote_share = votes[names].to_numpy(dtype=float) @ w
        soft_share = scores[names].to_numpy(dtype=float) @ w
        confidence = np.clip(VOTE_MIX * vote_share + (1 - VOTE_MIX) * soft_share, 0.0, 1.0)

        conf = pd.Series(confidence, index=index, name="confidence")
        return EnsembleResult(conf, conf >= self.threshold, votes, scores)

    def fit_detect(self, X: pd.DataFrame) -> EnsembleResult:
        return self.fit(X).detect(X)
