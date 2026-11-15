"""Data-drift detection with Evidently: has the data changed since the reference set?"""
from __future__ import annotations

import pandas as pd


def drift_report(reference: pd.DataFrame, current: pd.DataFrame, html_path: str | None = None) -> dict:
    """Compare `current` to `reference` on their shared numeric columns (K-S test per column)."""
    from evidently import Report
    from evidently.presets import DataDriftPreset

    cols = [c for c in reference.columns if c in current.columns]
    if not cols:
        raise ValueError("The two datasets share no numeric columns")
    snap = Report([DataDriftPreset()]).run(current[cols], reference[cols])
    if html_path:
        snap.save_html(html_path)

    per_column, summary = {}, {}
    for m in snap.dict()["metrics"]:
        name, value = m["metric_name"], m["value"]
        if name.startswith("DriftedColumnsCount"):
            summary = {"drifted_columns": int(value["count"]), "share": round(float(value["share"]), 4)}
        elif name.startswith("ValueDrift"):
            col = name.split("column=")[1].split(",")[0]
            per_column[col] = {"p_value": float(value), "drifted": bool(value < 0.05)}
    return {
        "columns_compared": cols,
        "dataset_drift": summary.get("share", 0) >= 0.5,
        **summary,
        "per_column": per_column,
    }
