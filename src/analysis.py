"""Backwards-compatible wrappers around the new detector classes.

The original app imported these three functions; they keep working and now
delegate to src/detectors. Each returns the outlying values as a Series.
"""
import pandas as pd

from detectors import DBSCANDetector, IQRDetector, IsolationForestDetector


def _clean(data: pd.Series) -> pd.Series:
    return data.dropna()


def detect_outliers_iqr(data: pd.Series) -> pd.Series:
    data = _clean(data)
    return data[IQRDetector().fit_predict(data)]


def detect_outliers_isolation_forest(data: pd.Series, contamination: float = 0.05) -> pd.Series:
    data = _clean(data)
    return data[IsolationForestDetector(contamination=contamination).fit_predict(data)]


def detect_outliers_dbscan(data: pd.Series, eps: float = 0.5, min_samples: int = 5) -> pd.Series:
    data = _clean(data)
    return data[DBSCANDetector(eps=eps, min_samples=min_samples).fit_predict(data)]
