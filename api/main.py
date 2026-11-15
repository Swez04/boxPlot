"""FastAPI service: /upload /detect /explain /report /chat /drift /health.

Run:  uvicorn api.main:app --reload
"""
from __future__ import annotations

import re
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

from agent import tools
from agent.graph import AnomalyAgent
from data_utils import load_dataframe, prepare_numeric, profile_dataset
from drift import drift_report
from tracking import log_detection_run

MAX_UPLOAD_MB = 50
ALLOWED_EXT = (".csv", ".xls", ".xlsx")
ID_RE = re.compile(r"^[0-9a-f]{32}$")

app = FastAPI(title="Anomaly Detection Platform", version="1.0.0")
_agent: AnomalyAgent | None = None
_logged_runs: dict[tuple, str | None] = {}


# ------------------------------------------------------------------ helpers
def _path_for(dataset_id: str) -> Path:
    if not ID_RE.match(dataset_id):
        raise HTTPException(400, "Invalid dataset_id")
    tools.data_dir().mkdir(parents=True, exist_ok=True)
    for ext in ALLOWED_EXT:
        p = tools.data_dir() / f"{dataset_id}{ext}"
        if p.is_file():
            return p
    raise HTTPException(404, f"Unknown dataset_id {dataset_id}")


def _analysis(dataset_id: str, columns, exclude_zeros: bool):
    path = _path_for(dataset_id)
    try:
        return path, tools._cached_analysis(path, columns, exclude_zeros)
    except (ValueError, KeyError) as exc:
        raise HTTPException(422, str(exc.args[0]) if exc.args else str(exc))


# ------------------------------------------------------------------- models
class DetectRequest(BaseModel):
    dataset_id: str
    columns: list[str] | None = None
    exclude_zeros: bool = False
    top_n: int = Field(10, ge=1, le=100)


class ChatRequest(BaseModel):
    dataset_id: str
    question: str = Field(min_length=1, max_length=2000)
    session_id: str = "default"


class DriftRequest(BaseModel):
    reference_id: str
    current_id: str
    columns: list[str] | None = None


# ---------------------------------------------------------------- endpoints
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    name = file.filename or ""
    if not name.lower().endswith(ALLOWED_EXT):
        raise HTTPException(400, "Upload a .csv, .xls or .xlsx file")
    content = await file.read()
    if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File larger than {MAX_UPLOAD_MB} MB")
    try:
        df = load_dataframe(content, name)
    except Exception as exc:
        raise HTTPException(400, f"Could not read the file: {exc}")
    if df.empty:
        raise HTTPException(400, "The file has no rows")

    dataset_id = uuid.uuid4().hex
    ext = Path(name).suffix.lower()
    tools.data_dir().mkdir(parents=True, exist_ok=True)
    (tools.data_dir() / f"{dataset_id}{ext}").write_bytes(content)
    return {"dataset_id": dataset_id, "filename": name, "profile": profile_dataset(df)}


@app.post("/detect")
def detect(req: DetectRequest):
    path, (X, result, explanation) = _analysis(req.dataset_id, req.columns, req.exclude_zeros)
    key = (str(path), path.stat().st_mtime_ns, tuple(req.columns or ()), req.exclude_zeros)
    if key not in _logged_runs:  # one MLflow run per distinct analysis, not per click
        _logged_runs[key] = log_detection_run(
            path.name, X, result, explanation,
            {"exclude_zeros": req.exclude_zeros, "models": ",".join(result.votes.columns)})
    top = result.top(req.top_n)
    return {
        "dataset_id": req.dataset_id,
        "mlflow_run_id": _logged_runs[key],
        "columns_analyzed": list(X.columns),
        "rows_analyzed": len(X),
        "anomalies_found": result.n_anomalies,
        "anomaly_rate": round(result.n_anomalies / len(X), 4),
        "top_anomalies": [
            {"row": r.item() if hasattr(r, "item") else r, "confidence": round(float(c), 4),
             "models_flagged": [m for m, v in result.votes.loc[r].items() if v],
             "values": {k: float(v) for k, v in X.loc[r].items()}}
            for r, c in top.items()
        ],
    }


@app.get("/explain/{dataset_id}")
def explain(dataset_id: str, rows: str | None = None, top_n: int = 5, exclude_zeros: bool = False):
    """`rows` is a comma-separated list of row ids; omit it for the top_n anomalies."""
    path = _path_for(dataset_id)
    row_ids = [r.strip() for r in rows.split(",") if r.strip()] if rows else None
    try:
        return tools.anomaly_explainer(path.name, row_ids, top_n, None, exclude_zeros)
    except (ValueError, KeyError) as exc:
        raise HTTPException(422, str(exc.args[0]) if exc.args else str(exc))


@app.get("/report/{dataset_id}")
def report(dataset_id: str, exclude_zeros: bool = False):
    """CSV of every flagged row with its confidence and which models voted for it."""
    _, (X, result, _) = _analysis(dataset_id, None, exclude_zeros)
    flagged = result.confidence[result.is_anomaly].sort_values(ascending=False)
    votes = result.votes.loc[flagged.index].add_prefix("flagged_by_")
    out = flagged.to_frame("confidence").join(X).join(votes)
    out.index.name = "row"
    return Response(out.to_csv(), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="anomalies_{dataset_id[:8]}.csv"'})


@app.get("/scored/{dataset_id}")
def scored(dataset_id: str, exclude_zeros: bool = False, max_normal: int = 5000):
    """Per-row confidence + votes for charting: every anomaly plus a sample of normal rows."""
    _, (X, result, _) = _analysis(dataset_id, None, exclude_zeros)
    keep = result.is_anomaly[result.is_anomaly].index
    normal = result.is_anomaly[~result.is_anomaly].index
    if len(normal) > max_normal:
        normal = normal.to_series().sample(max_normal, random_state=0).index
    idx = keep.union(normal)
    df = X.loc[idx].copy()
    df["confidence"] = result.confidence.loc[idx].round(4)
    df["is_anomaly"] = result.is_anomaly.loc[idx]
    df = df.join(result.votes.loc[idx].add_prefix("vote_"))
    df.index.name = "row"
    return {"columns": list(X.columns), "models": list(result.votes.columns),
            "records": df.reset_index().to_dict(orient="records")}


@app.post("/chat")
async def chat(req: ChatRequest):
    global _agent
    path = _path_for(req.dataset_id)
    if _agent is None:
        _agent = AnomalyAgent()
    try:
        return await _agent.ask(req.question, path.name, f"{req.dataset_id}:{req.session_id}")
    except RuntimeError as exc:  # e.g. ANTHROPIC_API_KEY missing
        raise HTTPException(503, str(exc))


@app.post("/drift")
def drift(req: DriftRequest):
    ref = prepare_numeric(load_dataframe(_path_for(req.reference_id), _path_for(req.reference_id).name), req.columns)
    cur = prepare_numeric(load_dataframe(_path_for(req.current_id), _path_for(req.current_id).name), req.columns)
    try:
        return drift_report(ref, cur)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
