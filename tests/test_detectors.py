import numpy as np
import pandas as pd
import pytest

from detectors import DBSCANDetector, IQRDetector, IsolationForestDetector
from detectors.base import to_2d_array

OUTLIER_IDX = {200, 201, 202}
ALL = [IQRDetector, IsolationForestDetector, DBSCANDetector]


# ---------- shared contract (every detector must satisfy these) ----------
@pytest.mark.parametrize("cls", ALL)
def test_scores_in_unit_interval(cls, data_with_outliers):
    scores = cls().fit(data_with_outliers).score(data_with_outliers)
    assert scores.shape == (len(data_with_outliers),)
    assert scores.min() >= 0.0 and scores.max() <= 1.0


@pytest.mark.parametrize("cls", ALL)
def test_outliers_score_higher_than_inliers(cls, data_with_outliers):
    scores = cls().fit(data_with_outliers).score(data_with_outliers)
    assert scores[list(OUTLIER_IDX)].mean() > scores[:200].mean()


@pytest.mark.parametrize("cls", ALL)
def test_catches_obvious_outliers(cls, data_with_outliers):
    flagged = set(np.flatnonzero(cls().fit_predict(data_with_outliers)))
    assert OUTLIER_IDX <= flagged


@pytest.mark.parametrize("cls", ALL)
def test_predict_before_fit_raises(cls):
    with pytest.raises(RuntimeError):
        cls().predict(np.array([1.0, 2.0]))


@pytest.mark.parametrize("cls", ALL)
def test_nan_input_rejected(cls):
    with pytest.raises(ValueError):
        cls().fit(np.array([1.0, np.nan, 3.0]))


@pytest.mark.parametrize("cls", ALL)
def test_accepts_series_and_multivariate(cls, data_2d):
    det = cls().fit(pd.DataFrame(data_2d))
    assert det.predict(data_2d).dtype == bool
    flagged = set(np.flatnonzero(det.predict(data_2d)))
    assert {300, 301} <= flagged


@pytest.mark.parametrize("cls", ALL)
def test_scores_unseen_data(cls, data_with_outliers):
    det = cls().fit(data_with_outliers[:200])
    new = np.array([50.0, 500.0])
    s = det.score(new)
    assert s[1] > s[0]


# ---------- model-specific behaviour ----------
def test_iqr_matches_manual_fences():
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 100], dtype=float)
    det = IQRDetector().fit(x)
    q1, q3 = np.percentile(x, [25, 75])
    assert det.lower_[0] == pytest.approx(q1 - 1.5 * (q3 - q1))
    assert det.upper_[0] == pytest.approx(q3 + 1.5 * (q3 - q1))
    assert list(np.flatnonzero(det.predict(x))) == [9]


def test_iqr_clean_data_flags_nothing():
    x = np.linspace(0, 10, 100)
    assert not IQRDetector().fit_predict(x).any()


def test_iqr_constant_column_flags_deviation():
    det = IQRDetector().fit(np.full(50, 5.0))
    assert det.predict(np.array([5.0, 9.0])).tolist() == [False, True]


def test_isolation_forest_respects_contamination(data_with_outliers):
    n = len(data_with_outliers)
    flagged = IsolationForestDetector(contamination=0.05).fit_predict(data_with_outliers).sum()
    assert flagged == pytest.approx(0.05 * n, abs=3)


def test_isolation_forest_is_reproducible(data_with_outliers):
    a = IsolationForestDetector(random_state=7).fit(data_with_outliers).score(data_with_outliers)
    b = IsolationForestDetector(random_state=7).fit(data_with_outliers).score(data_with_outliers)
    np.testing.assert_allclose(a, b)


def test_dbscan_no_core_points_flags_everything():
    x = np.arange(5, dtype=float) * 100  # too sparse for min_samples=5
    det = DBSCANDetector(eps=0.1, min_samples=5).fit(x)
    assert det.predict(x).all()
    assert (det.score(x) == 1.0).all()


def test_dbscan_agrees_with_plain_sklearn_noise_labels(data_with_outliers):
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler

    scaled = StandardScaler().fit_transform(data_with_outliers.reshape(-1, 1))
    noise_sklearn = DBSCAN(eps=0.5, min_samples=5).fit(scaled).labels_ == -1
    det = DBSCANDetector(eps=0.5, min_samples=5).fit(data_with_outliers)
    np.testing.assert_array_equal(det.predict(data_with_outliers), noise_sklearn)


def test_dbscan_handles_heavily_duplicated_data():
    # 90% zeros, like rainfall data: must not blow up and must match plain DBSCAN.
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(3)
    x = np.concatenate([np.zeros(9000), rng.gamma(2, 10, 900), [400.0, 600.0]])
    scaled = StandardScaler().fit_transform(x.reshape(-1, 1))
    expected = DBSCAN(eps=0.1, min_samples=5).fit(scaled).labels_ == -1
    got = DBSCANDetector(eps=0.1, min_samples=5).fit_predict(x)
    np.testing.assert_array_equal(got, expected)
    assert got[-2:].all()


# ---------- helper ----------
def test_to_2d_array_shapes():
    assert to_2d_array([1, 2, 3]).shape == (3, 1)
    assert to_2d_array(pd.Series([1.0, 2.0])).shape == (2, 1)
    with pytest.raises(ValueError):
        to_2d_array([])
