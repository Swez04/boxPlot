"""The original three functions must keep working for src/app.py."""
import numpy as np
import pandas as pd

from analysis import detect_outliers_dbscan, detect_outliers_iqr, detect_outliers_isolation_forest


def _series(data):
    return pd.Series(data, name="value")


def test_wrappers_return_series_subset(data_with_outliers):
    s = _series(data_with_outliers)
    for out in (detect_outliers_iqr(s),
                detect_outliers_isolation_forest(s, 0.05),
                detect_outliers_dbscan(s, 0.5, 5)):
        assert isinstance(out, pd.Series)
        assert set(out.index) <= set(s.index)
        assert {200, 201, 202} <= set(out.index)


def test_wrappers_ignore_nan(data_with_outliers):
    s = _series(data_with_outliers)
    s.iloc[5] = np.nan
    assert {200, 201, 202} <= set(detect_outliers_iqr(s).index)
