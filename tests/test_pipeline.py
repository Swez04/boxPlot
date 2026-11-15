"""Data utils, ensemble, SHAP explanations and charts."""
import numpy as np
import pandas as pd
import pytest

import visualization as viz
from data_utils import coerce_numeric, prepare_numeric, profile_dataset
from detectors import DBSCANDetector, IQRDetector, IsolationForestDetector
from ensemble import EnsembleDetector, default_detectors
from explain import explain_anomalies


@pytest.fixture
def frame():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "temp": rng.normal(20, 2, 400),
        "humidity": rng.normal(60, 5, 400),
        "pressure": rng.normal(1000, 3, 400),
    })
    df.loc[[10, 20, 30], :] = [[45, 10, 950], [-10, 95, 1050], [60, 30, 900]]
    return df


@pytest.fixture
def fast_ensemble():
    # Three classic detectors: keeps these tests fast and independent of torch.
    return EnsembleDetector([IQRDetector(), IsolationForestDetector(), DBSCANDetector()])


# ---------------------------------------------------------------- data utils
def test_coerce_numeric_parses_text_numbers():
    df = pd.DataFrame({"a": ["1", "2", None, "4"], "b": ["x", "y", "z", "w"]})
    out = coerce_numeric(df)
    assert pd.api.types.is_numeric_dtype(out["a"])
    assert not pd.api.types.is_numeric_dtype(out["b"])


def test_prepare_numeric_skips_ids_and_constants():
    df = pd.DataFrame({"user_id": range(50), "const": 1.0, "value": np.random.default_rng(0).normal(size=50)})
    assert list(prepare_numeric(df).columns) == ["value"]


def test_prepare_numeric_exclude_zeros_and_nan():
    df = pd.DataFrame({"v": [0, 0, 1.0, 2.0, np.nan, 3.0]})
    assert len(prepare_numeric(df, ["v"])) == 5
    assert len(prepare_numeric(df, ["v"], exclude_zeros=True)) == 3


def test_prepare_numeric_unknown_column():
    with pytest.raises(KeyError):
        prepare_numeric(pd.DataFrame({"a": [1, 2]}), ["nope"])


def test_profile_flags_zero_inflation_and_kinds():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "district_code": np.repeat([1, 2], 50),
        "date": pd.date_range("2024-01-01", periods=100).astype(str),
        "rain": np.where(rng.random(100) < 0.8, 0, rng.gamma(2, 10, 100)).astype(str),
        "name": ["a", "b"] * 50,
    })
    p = profile_dataset(df)
    kinds = {c["name"]: c["kind"] for c in p["columns"]}
    assert kinds == {"district_code": "identifier", "date": "datetime", "rain": "numeric", "name": "categorical"}
    assert p["recommended_columns"] == ["rain"]
    assert any("zeros" in s for s in p["suggestions"])


# ------------------------------------------------------------------ ensemble
def test_confidence_bounds_and_outliers_rank_top(frame, fast_ensemble):
    res = fast_ensemble.fit_detect(frame)
    assert res.confidence.between(0, 1).all()
    assert {10, 20, 30} <= set(res.top(10).index)
    assert res.is_anomaly[[10, 20, 30]].all()
    assert res.confidence[[10, 20, 30]].min() > res.confidence.drop([10, 20, 30]).median()


def test_all_models_agree_gives_high_confidence(frame, fast_ensemble):
    res = fast_ensemble.fit_detect(frame)
    unanimous = res.votes.all(axis=1)
    assert unanimous.any()
    assert (res.confidence[unanimous] >= 0.8).all()


def test_weights_change_outcome():
    votes_only = EnsembleDetector([IQRDetector(), DBSCANDetector()], weights={"iqr": 1, "dbscan": 0})
    x = pd.DataFrame({"v": np.r_[np.random.default_rng(0).normal(0, 1, 200), [8.0]]})
    r = votes_only.fit_detect(x)
    # dbscan weight 0 -> confidence driven by IQR alone
    assert r.votes["iqr"].iloc[-1] and r.confidence.iloc[-1] >= 0.8


def test_ensemble_validation(frame):
    with pytest.raises(ValueError):
        EnsembleDetector([])
    with pytest.raises(ValueError):
        EnsembleDetector([IQRDetector()], threshold=0)
    with pytest.raises(RuntimeError):
        EnsembleDetector([IQRDetector()]).detect(frame)
    with pytest.raises(ValueError):
        EnsembleDetector([IQRDetector()], weights={"iqr": 0}).fit_detect(frame)


def test_preserves_index(frame, fast_ensemble):
    shifted = frame.set_index(frame.index + 1000)
    res = fast_ensemble.fit_detect(shifted)
    assert set(res.top(3).index) <= set(shifted.index)


def test_default_detectors_include_four_when_torch_available():
    pytest.importorskip("torch")
    assert {d.name for d in default_detectors()} == {"iqr", "isolation_forest", "dbscan", "autoencoder"}


# ---------------------------------------------------------------------- SHAP
def test_shap_identifies_driving_feature():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(500, 3)), columns=["a", "b", "c"])
    X.loc[[5, 6, 7], "b"] = [25, -30, 28]          # only "b" is abnormal
    ens = EnsembleDetector([IQRDetector(), IsolationForestDetector(), DBSCANDetector()])
    res = ens.fit_detect(X)
    ex = explain_anomalies(X, res)
    assert ex.fidelity_r2 > 0.5
    # For the three planted outliers, "b" must be the feature SHAP blames.
    assert ex.shap_values.loc[[5, 6, 7]].abs().mean().idxmax() == "b"
    for row in (5, 6, 7):
        assert ex.top_features(row, k=1)[0]["feature"] == "b"
    assert list(ex.shap_values.columns) == ["a", "b", "c"]


def test_shap_no_anomalies_returns_empty(fast_ensemble):
    x = pd.DataFrame({"v": np.linspace(0, 1, 100)})
    res = fast_ensemble.fit_detect(x)
    if res.n_anomalies == 0:
        assert explain_anomalies(x, res).shap_values.empty


# -------------------------------------------------------------------- charts
def test_charts_build(frame, fast_ensemble):
    res = fast_ensemble.fit_detect(frame)
    ex = explain_anomalies(frame, res)
    top = res.top(1).index[0]
    figs = [
        viz.plot_boxplot_anomalies(frame, res.is_anomaly, "temp"),
        viz.plot_scatter(frame, res.confidence, res.is_anomaly),
        viz.plot_scatter(frame[["temp"]], res.confidence, res.is_anomaly),
        viz.plot_shap_bar(ex),
        viz.plot_shap_bar(ex, row=top),
        viz.plot_anomaly_heatmap(frame, res),
        viz.plot_model_agreement(res),
    ]
    assert all(len(f.data) > 0 for f in figs)


def test_mostly_empty_columns_are_not_auto_selected():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "good": rng.normal(size=200),
        "mostly_null": [1.0] * 10 + [np.nan] * 190,   # 95% missing
        "all_null": [np.nan] * 200,
    })
    assert list(prepare_numeric(df).columns) == ["good"]
    assert len(prepare_numeric(df)) == 200
    assert profile_dataset(df)["recommended_columns"] == ["good"]
