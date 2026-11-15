import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .base import BaseDetector, to_2d_array


class DBSCANDetector(BaseDetector):
    """DBSCAN noise detector that can also score unseen rows.

    sklearn's DBSCAN has no predict(), so after fitting we keep the core
    samples. A row is noise (anomalous) when it is farther than `eps` from every
    core sample, which is exactly DBSCAN's own definition of noise.

    Score = 1 - exp(-d / eps), d = distance to the nearest core sample.
    """

    name = "dbscan"

    def __init__(self, eps: float = 0.5, min_samples: int = 5) -> None:
        super().__init__()
        self.eps = eps
        self.min_samples = min_samples

    def fit(self, X) -> "DBSCANDetector":
        arr = to_2d_array(X)
        self.scaler_ = StandardScaler().fit(arr)
        scaled = self.scaler_.transform(arr)
        # Real data is often full of repeated values (e.g. rainfall is mostly 0).
        # Fitting on every duplicate makes DBSCAN's neighbour lists explode, so we
        # fit on the unique rows weighted by how often they occur. This gives the
        # same noise/core decisions as fitting on the full data.
        unique, counts = np.unique(scaled, axis=0, return_counts=True)
        self.model_ = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(
            unique, sample_weight=counts
        )
        core = unique[self.model_.core_sample_indices_]
        self._nn = NearestNeighbors(n_neighbors=1).fit(core) if len(core) else None
        self._fitted = True
        return self

    def _distance_to_core(self, X) -> np.ndarray:
        scaled = self.scaler_.transform(to_2d_array(X))
        if self._nn is None:  # no dense region at all -> everything is noise
            return np.full(len(scaled), np.inf)
        dist, _ = self._nn.kneighbors(scaled)
        return dist[:, 0]

    def score(self, X) -> np.ndarray:
        self._check_fitted()
        d = self._distance_to_core(X)
        return 1.0 - np.exp(-d / self.eps)  # exp(-inf) = 0 -> score 1.0

    def predict(self, X) -> np.ndarray:
        self._check_fitted()
        return self._distance_to_core(X) > self.eps
