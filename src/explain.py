"""SHAP explanations for the ensemble.

The four models are very different (statistical fences, trees, clustering, a
neural net), so there is no single SHAP explainer for "the ensemble". Instead we
fit a RandomForest *surrogate* that learns to predict the ensemble's confidence
from the raw features, then run SHAP's exact TreeExplainer on the surrogate.
`fidelity_r2` says how faithfully the surrogate mimics the ensemble; if it is
low, treat the explanation with caution.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from ensemble import EnsembleResult


@dataclass
class Explanation:
    shap_values: pd.DataFrame   # rows = explained anomalies, cols = features
    values: pd.DataFrame        # the raw feature values of those rows
    base_value: float           # surrogate's average predicted confidence
    fidelity_r2: float          # how well the surrogate mimics the ensemble

    def top_features(self, row, k: int = 3) -> list[dict]:
        """The k features pushing this row's confidence up the most."""
        sv = self.shap_values.loc[row]
        order = sv.abs().sort_values(ascending=False).index[:k]
        return [
            {"feature": f, "value": float(self.values.loc[row, f]), "shap": round(float(sv[f]), 4)}
            for f in order
        ]

    def mean_abs(self) -> pd.Series:
        return self.shap_values.abs().mean().sort_values(ascending=False)


def explain_anomalies(
    X: pd.DataFrame,
    result: EnsembleResult,
    max_explain: int = 50,
    max_train_rows: int = 5000,
    random_state: int = 42,
) -> Explanation:
    """Explain the `max_explain` highest-confidence anomalies in `X`."""
    if not result.is_anomaly.any():
        empty = pd.DataFrame(columns=X.columns, dtype=float)
        return Explanation(empty, empty.copy(), float(result.confidence.mean()), float("nan"))

    rng = np.random.default_rng(random_state)

    def _sample(idx: pd.Index, n: int) -> pd.Index:
        return idx if len(idx) <= n else pd.Index(rng.choice(idx, n, replace=False))

    # Train on a mix of flagged and normal rows (up to half each), so the surrogate
    # sees both sides of the decision boundary even when anomalies are rare.
    flagged_idx = result.is_anomaly[result.is_anomaly].index
    normal_idx = result.is_anomaly[~result.is_anomaly].index
    half = max_train_rows // 2
    train_idx = _sample(flagged_idx, half).union(_sample(normal_idx, half))

    surrogate = RandomForestRegressor(
        n_estimators=100, max_depth=10, min_samples_leaf=2, n_jobs=-1, random_state=random_state
    ).fit(X.loc[train_idx], result.confidence.loc[train_idx])
    fidelity = float(r2_score(result.confidence.loc[train_idx], surrogate.predict(X.loc[train_idx])))

    targets = result.top(max_explain).index
    explainer = shap.TreeExplainer(surrogate)
    sv = explainer.shap_values(X.loc[targets])
    base = float(np.ravel(explainer.expected_value)[0])
    return Explanation(
        shap_values=pd.DataFrame(sv, index=targets, columns=X.columns),
        values=X.loc[targets].copy(),
        base_value=base,
        fidelity_r2=fidelity,
    )
