# Intelligent Anomaly Detection Platform

Upload any numeric dataset, detect anomalies with an **ensemble of four models**, see *why* each
row was flagged (SHAP), and ask an **LLM agent** questions about the results in plain English.

```
Streamlit UI ──HTTP──▶ FastAPI ──▶ Ensemble (IQR · Isolation Forest · DBSCAN · PyTorch AutoEncoder)
   (no ML code)          │            ├─▶ SHAP explanations (surrogate + TreeExplainer)
                         │            ├─▶ MLflow run tracking
                         │            ├─▶ Evidently drift reports
                         └─▶ LangGraph agent ──▶ Claude ──▶ MCP tools (dataset_analyzer, anomaly_explainer)
```

## Quick start

**Docker (everything):**
```bash
cp .env.example .env          # add ANTHROPIC_API_KEY to enable the "Ask the AI" tab
docker compose up --build
```
UI <http://localhost:8501> · API docs <http://localhost:8000/docs> · MLflow <http://localhost:5000>

**Local:**
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pytest                                           # run the tests
uvicorn api.main:app --port 8000                 # terminal 1
API_URL=http://localhost:8000 streamlit run src/app.py   # terminal 2
export ANTHROPIC_API_KEY=...                     # only for the chat tab
```

## How it works

| Piece | File | What it does |
|---|---|---|
| Detectors | `src/detectors/` | `IQR`, `IsolationForest`, `DBSCAN`, `AutoEncoder` share one `fit / score / predict` interface; scores are 0–1 |
| Ensemble | `src/ensemble.py` | Weighted vote + soft score → a 0–1 **confidence** per row |
| Explainability | `src/explain.py` | SHAP on a random-forest surrogate of the ensemble (`fidelity_r2` says how faithful it is) |
| Data handling | `src/data_utils.py` | Reads CSV/Excel, parses numbers stored as text, skips ID / constant / mostly-empty columns, profiles the data |
| Agent | `src/agent/` | MCP server with two tools + a LangGraph loop that calls Claude |
| MLOps | `src/tracking.py`, `src/drift.py` | MLflow run logging; Evidently data-drift reports |
| API | `api/main.py` | `/upload /detect /explain /report /scored /chat /drift /health` |
| UI | `src/app.py` | Streamlit; talks only to the API |

**Confidence** = `0.8 × weighted vote share + 0.2 × weighted mean score`. A row is an anomaly at
confidence ≥ 0.5 (default weights: Isolation Forest and AutoEncoder 1.5, others 1.0).

**The chat agent** never computes anything itself. It calls the MCP tools, which return numbers
and SHAP drivers, and the model turns those into an explanation. File access is sandboxed to
`ANOMALY_DATA_DIR`. The MCP server also runs standalone, so Claude Desktop or any MCP client can use it:
`PYTHONPATH=src python -m agent.mcp_server`.

## Measured performance

Run `PYTHONPATH=src python scripts/benchmark.py`. It injects known anomalies (half global outliers,
half *contextual*: normal-looking per feature but with the correlation broken) into synthetic
data and reports means over 5 seeds:

| Model | Precision | Recall | F1 |
|---|---|---|---|
| **Ensemble** | 0.918 | 0.673 | **0.775** |
| AutoEncoder | 0.597 | 0.996 | 0.747 |
| DBSCAN | 0.996 | 0.520 | 0.683 |
| Isolation Forest | 0.673 | 0.673 | 0.673 |
| IQR | 1.000 | 0.498 | 0.665 |

The ensemble gives the best F1 and far higher precision than any single flexible model, but it is
**not** "95% accurate": it misses roughly a third of the injected anomalies. The AutoEncoder alone
catches almost everything but over-flags (its threshold is the 95th percentile of training error,
so it flags ~5% by design). Lower `threshold` in `EnsembleDetector` to trade precision for recall.
These are synthetic results; accuracy on your data depends on what "anomalous" means there.

## Real-data note: zero-heavy data (rainfall)

In `rainfall 2024.csv`, 86% of `actual_daily` is zero, so IQR fences collapse onto zero and every
rainy day looks like an outlier. Use **Exclude zeros** (sidebar, or `exclude_zeros` in the API):
flagged rows drop from 5.6% to 1.2% and the top results are genuine extreme-rain days.

## Tests

`pytest` runs 78 tests: detector contracts, DBSCAN on duplicate-heavy data, AutoEncoder
save/load, ensemble maths, SHAP attribution, the MCP server over real stdio, the agent loop
(scripted LLM), and every API endpoint. The live Claude call is **not** covered by tests, since
it needs an API key.

## Deploying

- **One server (simplest):** any VM with Docker → `docker compose up -d`, put a reverse proxy
  (Caddy / nginx) with HTTPS in front of port 8501. Don't expose 5000 (MLflow) or 8000 publicly
  without authentication; the API currently has none.
- **Managed hosts (Render, Railway, Fly.io):** deploy `docker/Dockerfile.api` and
  `docker/Dockerfile.ui` as two services, set `API_URL` on the UI, `ANTHROPIC_API_KEY` on the API,
  and mount a persistent volume at `/app/data/uploads`.

## Known limitations

- No authentication or per-user data isolation, and uploads are stored unencrypted on disk.
- Analyses are cached in memory per process (4 datasets); run one API worker.
- SHAP explains a surrogate of the ensemble, not the ensemble itself; check `fidelity_r2`.
- Numeric columns only; categorical/text columns are ignored.

## Author

Swez · swezbhardwaj@gmail.com
