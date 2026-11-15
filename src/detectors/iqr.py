import numpy as np

from .base import BaseDetector, to_2d_array


class IQRDetector(BaseDetector):
    """Tukey-fence detector. A row is anomalous if ANY column is outside its fences.

    Score = 1 - exp(-d), where d is the largest distance outside a fence,
    measured in IQR units (0 inside the fences).
    """

    name = "iqr"

    def __init__(self, k: float = 1.5) -> None:
        super().__init__()
        self.k = k

    def fit(self, X) -> "IQRDetector":
        arr = to_2d_array(X)
        q1, q3 = np.percentile(arr, [25, 75], axis=0)
        iqr = q3 - q1
        self.q1_, self.q3_, self.iqr_ = q1, q3, iqr
        self.lower_ = q1 - self.k * iqr
        self.upper_ = q3 + self.k * iqr
        self._fitted = True
        return self

    def _excess(self, arr: np.ndarray) -> np.ndarray:
        # Distance outside the fences, in IQR units. Constant columns (IQR=0)
        # use a unit of 1 so any deviation is still flagged.
        unit = np.where(self.iqr_ > 0, self.iqr_, 1.0)
        below = np.clip(self.lower_ - arr, 0, None) / unit
        above = np.clip(arr - self.upper_, 0, None) / unit
        return np.maximum(below, above).max(axis=1)

    def score(self, X) -> np.ndarray:
        self._check_fitted()
        return 1.0 - np.exp(-self._excess(to_2d_array(X)))

    def predict(self, X) -> np.ndarray:
        self._check_fitted()
        return self._excess(to_2d_array(X)) > 0
