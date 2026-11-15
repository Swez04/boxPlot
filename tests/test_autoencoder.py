import numpy as np
import pytest

torch = pytest.importorskip("torch")

from detectors import AutoEncoderDetector  # noqa: E402


def fast(**kw):
    return AutoEncoderDetector(epochs=30, batch_size=64, **kw)


def test_scores_in_unit_interval(data_with_outliers):
    s = fast().fit(data_with_outliers).score(data_with_outliers)
    assert s.min() >= 0.0 and s.max() <= 1.0


def test_flags_obvious_outliers(data_with_outliers):
    flagged = set(np.flatnonzero(fast().fit_predict(data_with_outliers)))
    assert {200, 201, 202} <= flagged


def test_multivariate_outliers(data_2d):
    flagged = set(np.flatnonzero(fast().fit_predict(data_2d)))
    assert {300, 301} <= flagged


def test_threshold_is_percentile_of_training_error(data_with_outliers):
    det = fast(threshold_percentile=90).fit(data_with_outliers)
    err = det.reconstruction_error(data_with_outliers)
    assert (err > det.threshold_).mean() == pytest.approx(0.10, abs=0.02)


def test_score_is_half_at_threshold(data_with_outliers):
    det = fast().fit(data_with_outliers)
    err = det.threshold_
    assert err / (err + det.threshold_) == 0.5
    assert (det.score(data_with_outliers) > 0.5).tolist() == det.predict(data_with_outliers).tolist()


def test_training_reduces_loss(data_2d):
    det = fast().fit(data_2d)
    assert det.loss_history_[-1] < det.loss_history_[0]


def test_reproducible(data_with_outliers):
    a = fast(random_state=3).fit(data_with_outliers).score(data_with_outliers)
    b = fast(random_state=3).fit(data_with_outliers).score(data_with_outliers)
    np.testing.assert_allclose(a, b, rtol=1e-5)


def test_save_load_roundtrip(tmp_path, data_2d):
    det = fast().fit(data_2d)
    path = tmp_path / "ae.pt"
    det.save(path)
    loaded = AutoEncoderDetector.load(path)
    np.testing.assert_allclose(det.score(data_2d), loaded.score(data_2d), rtol=1e-5)
    np.testing.assert_array_equal(det.predict(data_2d), loaded.predict(data_2d))


def test_unfitted_raises():
    with pytest.raises(RuntimeError):
        AutoEncoderDetector().predict(np.array([1.0]))


def test_bad_percentile_rejected():
    with pytest.raises(ValueError):
        AutoEncoderDetector(threshold_percentile=20)


def test_constant_column_does_not_crash():
    x = np.column_stack([np.full(100, 5.0), np.random.default_rng(0).normal(size=100)])
    assert fast().fit(x).score(x).shape == (100,)
