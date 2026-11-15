"""MLflow experiment tracking. Never lets a tracking failure break a detection run."""
from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)
EXPERIMENT = "anomaly-platform"


def tracking_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow/mlflow.db")


def log_detection_run(dataset_name: str, X, result, explanation, params: dict,
                      model_paths: list[Path] | None = None) -> str | None:
    """Log one detection run. Returns the MLflow run id, or None if tracking is unavailable."""
    try:
        import mlflow

        Path("mlflow").mkdir(exist_ok=True)
        mlflow.set_tracking_uri(tracking_uri())
        mlflow.set_experiment(EXPERIMENT)
        with mlflow.start_run(run_name=f"detect-{dataset_name}") as run:
            mlflow.log_params({"dataset": dataset_name, "n_rows": len(X),
                               "columns": ",".join(map(str, X.columns)), **params})
            metrics = {
                "n_anomalies": result.n_anomalies,
                "anomaly_rate": result.n_anomalies / len(X),
                "mean_confidence": float(result.confidence.mean()),
            }
            for det, votes in result.votes.items():
                metrics[f"flag_rate_{det}"] = float(votes.mean())
            if explanation is not None and explanation.fidelity_r2 == explanation.fidelity_r2:  # not NaN
                metrics["shap_fidelity_r2"] = explanation.fidelity_r2
            mlflow.log_metrics(metrics)
            with tempfile.TemporaryDirectory() as tmp:
                flagged = result.confidence[result.is_anomaly].sort_values(ascending=False)
                out = Path(tmp) / "flagged_rows.csv"
                flagged.to_frame().join(X).to_csv(out)
                mlflow.log_artifact(str(out))
            for p in model_paths or []:
                if Path(p).exists():
                    mlflow.log_artifact(str(p), artifact_path="models")
            return run.info.run_id
    except Exception as exc:  # tracking must never take the API down
        log.warning("MLflow logging failed: %s", exc)
        return None
