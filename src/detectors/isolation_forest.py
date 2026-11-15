import numpy as np
from sklearn.ensemble import IsolationForest

from .base import BaseDetector, to_2d_array


class IsolationForestDetector(BaseDetector):
    """Wrapper around scikit-learn's IsolationForest with scores scaled to [0, 1]."""

    name = "isolation_forest"

    def __init__(self, contamination: float = 0.05, n_estimators: int = 200,
                 random_state: int = 42) -> None:
        super().__init__()
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit(self, X) -> "IsolationForestDetector":
        arr = to_2d_array(X)
        self.model_ = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
        ).fit(arr)
        raw = -self.model_.score_samples(arr)  # higher = more anomalous
        self._lo, self._hi = float(raw.min()), float(raw.max())
        self._fitted = True
        return self

    def score(self, X) -> np.ndarray:
        self._check_fitted()
        raw = -self.model_.score_samples(to_2d_array(X))
        span = self._hi - self._lo
        if span == 0:
            return np.zeros_like(raw)
        return np.clip((raw - self._lo) / span, 0.0, 1.0)

    def predict(self, X) -> np.ndarray:
        self._check_fitted()
        return self.model_.predict(to_2d_array(X)) == -1
