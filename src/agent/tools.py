"""The two tools the LLM agent can call. Plain functions, no MCP/LangChain imports,
so they are easy to test; `mcp_server.py` exposes them over MCP.

Tools return *facts* (numbers, SHAP drivers, a plain-English summary). The LLM
turns those into a conversational answer; it must not invent numbers.
"""
from __future__ import annotations

import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

from data_utils import load_dataframe, prepare_numeric, profile_dataset
from ensemble import EnsembleDetector, default_detectors
from explain import explain_anomalies

_CACHE: "OrderedDict[tuple, tuple]" = OrderedDict()
_CACHE_SIZE = 4


def data_dir() -> Path:
    """Directory the tools may read from (the LLM cannot read files outside it)."""
    return Path(os.environ.get("ANOMALY_DATA_DIR", "data/uploads")).resolve()


def _resolve(dataset_path: str) -> Path:
    p = Path(dataset_path)
    p = (data_dir() / p) if not p.is_absolute() else p
    p = p.resolve()
    if data_dir() not in p.parents and p != data_dir():
        raise PermissionError("dataset_path must be inside the configured data directory")
    if not p.is_file():
        raise FileNotFoundError(f"No dataset at {dataset_path}")
    return p


def _load(path: Path) -> pd.DataFrame:
    return load_dataframe(path, path.name)


def run_analysis(df: pd.DataFrame, columns=None, exclude_zeros: bool = False,
                 contamination: float = 0.05, seed: int = 42):
    """Prepare data, run the ensemble, explain the anomalies. Returns (X, result, explanation)."""
    X = prepare_numeric(df, columns, exclude_zeros)
    if len(X) < 20:
        raise ValueError(f"Only {len(X)} usable rows after cleaning; need at least 20")
    if X.shape[1] == 0:
        raise ValueError("No numeric columns to analyse")
    ens = EnsembleDetector(default_detectors(seed, contamination))
    result = ens.fit_detect(X)
    return X, result, explain_anomalies(X, result, random_state=seed)


def _cached_analysis(path: Path, columns, exclude_zeros: bool):
    key = (str(path), path.stat().st_mtime_ns, tuple(columns) if columns else None, exclude_zeros)
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key]
    out = run_analysis(_load(path), columns, exclude_zeros)
    _CACHE[key] = out
    while len(_CACHE) > _CACHE_SIZE:
        _CACHE.popitem(last=False)
    return out


# ------------------------------------------------------------------- tool 1
def dataset_analyzer(dataset_path: str) -> dict:
    """Detect column types and suggest how to analyse the dataset."""
    return profile_dataset(_load(_resolve(dataset_path)))


# ------------------------------------------------------------------- tool 2
def _describe_row(X: pd.DataFrame, result, explanation, row) -> dict:
    conf = float(result.confidence[row])
    voted = [m for m, v in result.votes.loc[row].items() if v]
    n_models = result.votes.shape[1]
    flagged = bool(result.is_anomaly[row])

    findings = []
    for col in X.columns:
        mean, std = X[col].mean(), X[col].std()
        value = float(X.loc[row, col])
        z = (value - mean) / std if std else 0.0
        findings.append({"feature": col, "value": round(value, 4), "z_score": round(float(z), 2),
                         "typical_range": [round(float(X[col].quantile(.05)), 4), round(float(X[col].quantile(.95)), 4)]})
    drivers = explanation.top_features(row, 3) if row in explanation.shap_values.index else []

    if flagged:
        text = (f"Row {row} was flagged as an anomaly with {conf:.0%} confidence; "
                f"{len(voted)} of {n_models} models agreed ({', '.join(voted)}).")
        if drivers:
            parts = []
            for d in drivers:
                f = next(f for f in findings if f["feature"] == d["feature"])
                direction = "above" if f["z_score"] > 0 else "below"
                parts.append(f"{d['feature']} = {d['value']:g} ({abs(f['z_score']):.1f} standard deviations {direction} "
                             f"average; typical range {f['typical_range'][0]:g} to {f['typical_range'][1]:g})")
            text += " The biggest contributors were " + "; ".join(parts) + "."
    else:
        text = (f"Row {row} was NOT flagged: confidence {conf:.0%} (below the cutoff), "
                f"{len(voted)} of {n_models} models flagged it.")
    return {"row": _py(row), "flagged": flagged, "confidence": round(conf, 4), "models_flagged": voted,
            "features": findings, "shap_drivers": drivers, "explanation": text}


def _py(v):
    return v.item() if isinstance(v, np.generic) else v


def anomaly_explainer(dataset_path: str, row_ids: list | None = None, top_n: int = 5,
                      columns: list[str] | None = None, exclude_zeros: bool = False) -> dict:
    """Explain flagged anomalies (the top_n most confident, or the given row_ids) in plain English."""
    path = _resolve(dataset_path)
    X, result, explanation = _cached_analysis(path, columns, exclude_zeros)
    top_n = max(1, min(int(top_n), 20))

    if row_ids:
        rows, missing = [], []
        for r in row_ids:
            match = next((i for i in X.index if str(i) == str(r)), None)
            (rows if match is not None else missing).append(match if match is not None else r)
    else:
        rows, missing = list(result.top(top_n).index), []

    # Rows outside the SHAP-explained set (e.g. not flagged) still get a description.
    explained = [_describe_row(X, result, explanation, r) for r in rows[:20]]
    return {
        "dataset": path.name,
        "columns_analyzed": list(X.columns),
        "rows_analyzed": int(len(X)),
        "anomalies_found": result.n_anomalies,
        "anomaly_rate": round(result.n_anomalies / len(X), 4),
        "shap_fidelity_r2": None if np.isnan(explanation.fidelity_r2) else round(explanation.fidelity_r2, 3),
        "overall_drivers": explanation.mean_abs().round(4).to_dict() if not explanation.shap_values.empty else {},
        "rows_not_found": [_py(m) for m in missing],
        "explanations": explained,
    }
