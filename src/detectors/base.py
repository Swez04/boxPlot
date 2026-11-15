"""Common interface shared by every anomaly detector in the platform.

Every detector exposes the same three methods so the Phase 3 ensemble can
treat them interchangeably:

    fit(X)    -> self
    score(X)  -> np.ndarray of anomaly scores in [0, 1]  (1 = most anomalous)
    predict(X)-> np.ndarray of bool                      (True = anomaly)
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


def to_2d_array(X) -> np.ndarray:
    """Convert a Series / DataFrame / list / ndarray to a clean 2-D float array."""
    if isinstance(X, (pd.Series, pd.DataFrame)):
        X = X.to_numpy()
    arr = np.asarray(X, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"Expected 1-D or 2-D input, got {arr.ndim}-D")
    if arr.shape[0] == 0:
        raise ValueError("Input is empty")
    if np.isnan(arr).any():
        raise ValueError("Input contains NaN values; drop or impute them first")
    return arr


class BaseDetector(ABC):
    """Abstract base class for all detectors."""

    name: str = "base"

    def __init__(self) -> None:
        self._fitted = False

    @abstractmethod
    def fit(self, X) -> "BaseDetector":
        ...

    @abstractmethod
    def score(self, X) -> np.ndarray:
        """Anomaly score in [0, 1] for each row (higher = more anomalous)."""

    @abstractmethod
    def predict(self, X) -> np.ndarray:
        """Boolean mask, True where the row is an anomaly."""

    def fit_predict(self, X) -> np.ndarray:
        return self.fit(X).predict(X)

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(f"{type(self).__name__} must be fitted before use")
