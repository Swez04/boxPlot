"""Measure detector quality on synthetic data with KNOWN injected anomalies.

    PYTHONPATH=src python scripts/benchmark.py

Normal rows come from correlated Gaussian clusters. Anomalies are of two kinds:
  - global:     far outside the data in every feature
  - contextual: each feature looks normal on its own, but the combination is impossible
                (breaks the correlation) -- univariate methods like IQR cannot see these.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ensemble import EnsembleDetector, default_detectors  # noqa: E402


def make_data(seed: int, n: int = 3000, rate: float = 0.03):
    rng = np.random.default_rng(seed)
    cov = np.array([[1, .9, .2], [.9, 1, .1], [.2, .1, 1]])
    X = rng.multivariate_normal([0, 0, 0], cov, n)
    X[n // 2:] += [6, 6, 3]                      # second cluster
    y = np.zeros(n, bool)
    k = int(n * rate)
    idx = rng.choice(n, k, replace=False)
    y[idx] = True
    g, c = idx[: k // 2], idx[k // 2:]
    X[g] = rng.uniform(-6, 12, (len(g), 3)) * rng.choice([-1, 1], (len(g), 3)) + 15 * rng.choice([-1, 1], (len(g), 1))
    X[c, 0], X[c, 1] = X[c, 0] + 2.5, X[c, 1] - 2.5   # same marginals roughly, correlation broken
    return pd.DataFrame(X, columns=["f1", "f2", "f3"]), y, set(g)


def prf(pred, y):
    tp = (pred & y).sum()
    p = tp / max(pred.sum(), 1)
    r = tp / max(y.sum(), 1)
    return p, r, 2 * p * r / max(p + r, 1e-9)


def main(seeds=(0, 1, 2, 3, 4)):
    rows = []
    for seed in seeds:
        X, y, glob = make_data(seed)
        ens = EnsembleDetector(default_detectors(seed, contamination=0.03, ae_epochs=30)).fit(X)
        res = ens.detect(X)
        preds = {c: res.votes[c].to_numpy() for c in res.votes} | {"ENSEMBLE": res.is_anomaly.to_numpy()}
        for name, pred in preds.items():
            p, r, f = prf(pred, y)
            rows.append({"model": name, "precision": p, "recall": r, "f1": f})
    out = pd.DataFrame(rows).groupby("model").mean().round(3).sort_values("f1", ascending=False)
    print(f"Mean over {len(seeds)} seeds (3000 rows, 3% injected anomalies, half contextual):\n")
    print(out.to_string())
    return out


if __name__ == "__main__":
    main()
