import io

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
from fastapi.testclient import TestClient

from api import main


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("ANOMALY_DATA_DIR", str(tmp_path / "up"))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/mlflow.db")
    monkeypatch.chdir(tmp_path)
    main._logged_runs.clear()
    main._agent = None
    return TestClient(main.app)


def csv_bytes(shift=0.0, n=300):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"temp": rng.normal(20 + shift, 2, n), "pressure": rng.normal(1000, 4, n)})
    df.loc[[7, 8], :] = [[80, 900], [-30, 1100]]
    return df.to_csv(index=False).encode()


def upload(client, data=None, name="d.csv"):
    r = client.post("/upload", files={"file": (name, data or csv_bytes(), "text/csv")})
    assert r.status_code == 200, r.text
    return r.json()["dataset_id"]


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_upload_returns_profile(client):
    r = client.post("/upload", files={"file": ("d.csv", csv_bytes(), "text/csv")}).json()
    assert len(r["dataset_id"]) == 32
    assert r["profile"]["recommended_columns"] == ["temp", "pressure"]


def test_upload_rejects_bad_type_and_empty(client):
    assert client.post("/upload", files={"file": ("x.exe", b"abc", "application/x")}).status_code == 400
    assert client.post("/upload", files={"file": ("e.csv", b"a,b\n", "text/csv")}).status_code == 400


def test_excel_upload(client):
    buf = io.BytesIO()
    pd.read_csv(io.BytesIO(csv_bytes())).to_excel(buf, index=False)
    ds = upload(client, buf.getvalue(), "d.xlsx")
    assert client.post("/detect", json={"dataset_id": ds}).status_code == 200


def test_detect_finds_planted_anomalies_and_logs_mlflow(client):
    ds = upload(client)
    r = client.post("/detect", json={"dataset_id": ds, "top_n": 5}).json()
    assert {a["row"] for a in r["top_anomalies"]} >= {7, 8}
    assert r["mlflow_run_id"]
    again = client.post("/detect", json={"dataset_id": ds}).json()
    assert again["mlflow_run_id"] == r["mlflow_run_id"]  # not logged twice


def test_explain_and_report(client):
    ds = upload(client)
    e = client.get(f"/explain/{ds}", params={"rows": "7,150"}).json()
    by = {x["row"]: x for x in e["explanations"]}
    assert by[7]["flagged"] and not by[150]["flagged"]
    rep = client.get(f"/report/{ds}")
    assert rep.headers["content-type"].startswith("text/csv")
    out = pd.read_csv(io.StringIO(rep.text))
    assert {7, 8} <= set(out["row"]) and "confidence" in out and "flagged_by_iqr" in out


def test_drift_endpoint(client):
    ref, cur = upload(client), upload(client, csv_bytes(shift=6))
    r = client.post("/drift", json={"reference_id": ref, "current_id": cur}).json()
    assert r["per_column"]["temp"]["drifted"] and not r["per_column"]["pressure"]["drifted"]


def test_unknown_and_malformed_ids(client):
    assert client.post("/detect", json={"dataset_id": "0" * 32}).status_code == 404
    assert client.post("/detect", json={"dataset_id": "../../etc/passwd"}).status_code == 400
    assert client.get("/report/not-an-id").status_code == 400


def test_detect_unknown_column_is_422(client):
    ds = upload(client)
    assert client.post("/detect", json={"dataset_id": ds, "columns": ["nope"]}).status_code == 422


def test_chat_without_api_key_is_503(client, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    ds = upload(client)
    r = client.post("/chat", json={"dataset_id": ds, "question": "hi"})
    assert r.status_code == 503 and "ANTHROPIC_API_KEY" in r.json()["detail"]


def test_scored_endpoint_for_charts(client):
    ds = upload(client)
    r = client.get(f"/scored/{ds}").json()
    df = pd.DataFrame(r["records"])
    assert {"row", "confidence", "is_anomaly", "temp", "pressure", "vote_iqr"} <= set(df)
    assert df.loc[df.row.isin([7, 8]), "is_anomaly"].all()
