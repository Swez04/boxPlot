"""Streamlit UI. It holds no ML code: every action is a call to the FastAPI service.

    uvicorn api.main:app --port 8000          # terminal 1
    API_URL=http://localhost:8000 streamlit run src/app.py   # terminal 2
"""
import os
import sys
import uuid
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import streamlit as st

import visualization as viz
from api_client import ApiClient, ApiError

st.set_page_config(page_title="Anomaly Detection Platform", page_icon="📊", layout="wide")
st.title("📊 Intelligent Anomaly Detection")

api_url = st.sidebar.text_input("API URL", os.environ.get("API_URL", "http://localhost:8000"))
client = ApiClient(api_url)
if not client.health():
    st.error(f"Can't reach the API at {api_url}. Start it with `uvicorn api.main:app --port 8000`.")
    st.stop()

ss = st.session_state
ss.setdefault("session_id", uuid.uuid4().hex[:8])
ss.setdefault("datasets", {})      # filename -> {dataset_id, profile}
ss.setdefault("chat", {})          # dataset_id -> [(role, text)]

# ---------------------------------------------------------------- sidebar
upload = st.sidebar.file_uploader("Upload dataset", type=["csv", "xls", "xlsx"])
if upload is not None and upload.name not in ss.datasets:
    try:
        with st.spinner("Uploading and profiling..."):
            ss.datasets[upload.name] = client.upload(upload.name, upload.getvalue())
    except ApiError as exc:
        st.sidebar.error(str(exc))

if not ss.datasets:
    st.info("Upload a CSV or Excel file in the sidebar to begin.")
    st.stop()

current = st.sidebar.selectbox("Dataset", list(ss.datasets))
ds = ss.datasets[current]
dataset_id, profile = ds["dataset_id"], ds["profile"]
exclude_zeros = st.sidebar.checkbox("Exclude zeros", help="Recommended for zero-heavy data like rainfall.")

tab_overview, tab_charts, tab_explain, tab_chat, tab_drift = st.tabs(
    ["Overview", "Charts", "Explain", "Ask the AI", "Drift"])

# ---------------------------------------------------------------- overview
with tab_overview:
    st.subheader("Dataset profile")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{profile['n_rows']:,}")
    c2.metric("Columns", profile["n_cols"])
    c3.metric("Analysable numeric columns", len(profile["recommended_columns"]))
    for tip in profile["suggestions"]:
        st.warning(tip)
    st.dataframe(pd.DataFrame(profile["columns"]), width="stretch", hide_index=True)

    chosen = st.multiselect("Columns to analyse", profile["recommended_columns"],
                            default=profile["recommended_columns"][:3])
    if st.button("Detect anomalies", type="primary"):
        try:
            with st.spinner("Running 4 models..."):
                ss[f"det:{dataset_id}"] = client.detect(dataset_id, chosen or None, exclude_zeros, top_n=20)
        except ApiError as exc:
            st.error(str(exc))

    det = ss.get(f"det:{dataset_id}")
    if det:
        a, b, c = st.columns(3)
        a.metric("Rows analysed", f"{det['rows_analyzed']:,}")
        b.metric("Anomalies", f"{det['anomalies_found']:,}")
        c.metric("Anomaly rate", f"{det['anomaly_rate']:.1%}")
        top = pd.DataFrame([{"row": t["row"], "confidence": t["confidence"],
                             "models": ", ".join(t["models_flagged"]), **t["values"]} for t in det["top_anomalies"]])
        st.dataframe(top, width="stretch", hide_index=True)
        st.download_button("Download full report (CSV)", client.report(dataset_id, exclude_zeros),
                           file_name="anomalies.csv", mime="text/csv")
        if det.get("mlflow_run_id"):
            st.caption(f"Logged to MLflow run {det['mlflow_run_id'][:8]}")

det = ss.get(f"det:{dataset_id}")
needs_run = "Run **Detect anomalies** on the Overview tab first."

# ------------------------------------------------------------------ charts
with tab_charts:
    if not det:
        st.info(needs_run)
    else:
        data = client.scored(dataset_id, exclude_zeros)
        df = pd.DataFrame(data["records"]).set_index("row")
        X, conf, flag = df[data["columns"]], df["confidence"], df["is_anomaly"]
        col = st.selectbox("Column", data["columns"])
        st.plotly_chart(viz.plot_boxplot_anomalies(X, flag, col), width="stretch", config=viz.PLOTLY_CONFIG)
        st.plotly_chart(viz.plot_scatter(X, conf, flag), width="stretch", config=viz.PLOTLY_CONFIG)
        votes = df[[f"vote_{m}" for m in data["models"]]].rename(columns=lambda c: c[5:])
        res = SimpleNamespace(votes=votes, top=lambda n: conf[flag].sort_values(ascending=False).head(n))
        st.plotly_chart(viz.plot_model_agreement(res), width="stretch", config=viz.PLOTLY_CONFIG)

# ----------------------------------------------------------------- explain
with tab_explain:
    if not det:
        st.info(needs_run)
    else:
        rows = st.text_input("Explain specific rows (comma-separated), or leave empty for the top anomalies")
        ex = client.explain(dataset_id, [r for r in rows.split(",") if r.strip()] or None, 5, exclude_zeros)
        if ex["shap_fidelity_r2"] is not None and ex["shap_fidelity_r2"] < 0.5:
            st.warning(f"SHAP surrogate fidelity is low ({ex['shap_fidelity_r2']}); treat explanations as approximate.")
        if ex["overall_drivers"]:
            shim = SimpleNamespace(mean_abs=lambda: pd.Series(ex["overall_drivers"]).sort_values(ascending=False))
            st.plotly_chart(viz.plot_shap_bar(shim), width="stretch", config=viz.PLOTLY_CONFIG)
        for e in ex["explanations"]:
            st.markdown(f"**Row {e['row']}** · confidence {e['confidence']:.0%}")
            st.write(e["explanation"])
        for r in ex["rows_not_found"]:
            st.error(f"Row {r} not found.")

# -------------------------------------------------------------------- chat
with tab_chat:
    history = ss.chat.setdefault(dataset_id, [])
    for role, text in history:
        st.chat_message(role).write(text)
    if question := st.chat_input("Ask about this dataset, e.g. 'Why is row 12 an anomaly?'"):
        history.append(("user", question))
        st.chat_message("user").write(question)
        try:
            with st.spinner("Thinking..."):
                reply = client.chat(dataset_id, question, ss.session_id)
            answer = reply["answer"]
            if reply["tools_used"]:
                answer += f"\n\n*Tools used: {', '.join(reply['tools_used'])}*"
        except ApiError as exc:
            answer = f"⚠️ {exc}"
        history.append(("assistant", answer))
        st.chat_message("assistant").write(answer)

# ------------------------------------------------------------------- drift
with tab_drift:
    others = {k: v["dataset_id"] for k, v in ss.datasets.items() if k != current}
    if not others:
        st.info("Upload a second dataset to compare it with the selected one.")
    else:
        ref_name = st.selectbox("Reference (baseline) dataset", list(others))
        if st.button("Check drift"):
            try:
                r = client.drift(others[ref_name], dataset_id)
                (st.error if r["dataset_drift"] else st.success)(
                    f"{r['drifted_columns']} of {len(r['columns_compared'])} columns drifted.")
                st.dataframe(pd.DataFrame(r["per_column"]).T, width="stretch")
            except ApiError as exc:
                st.error(str(exc))
