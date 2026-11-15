"""Plot builders. Every function RETURNS a figure (never calls .show()) so Streamlit,
the API and tests can all use them. Plotly figures are interactive by default:
hover tooltips, zoom/pan, and a camera button that downloads a PNG."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

PLOTLY_CONFIG = {
    "displaylogo": False,
    "toImageButtonOptions": {"format": "png", "filename": "anomaly_chart", "scale": 2},
}
NORMAL_COLOR = "#4c78a8"
ANOMALY_COLOR = "#e45756"


# ------------------------------------------------------------ legacy box plots
def plot_box(data, column):
    """Static matplotlib box plot (kept for the original app)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x=data[column].dropna(), color="lightblue",
                flierprops=dict(marker="o", markerfacecolor="red", markersize=6), ax=ax)
    ax.set_title(f"Box Plot of {column}")
    ax.set_xlabel(column)
    ax.grid(True)
    fig.tight_layout()
    return fig


def plot_interactive(data, column):
    return px.box(data, y=column, points="outliers",
                  title=f"Interactive Box Plot of {column}")


# ----------------------------------------------------------- ensemble charts
def plot_boxplot_anomalies(X: pd.DataFrame, is_anomaly: pd.Series, column: str,
                           max_points: int = 5000) -> go.Figure:
    """Box plot whose points are coloured by the ensemble's verdict."""
    df = pd.DataFrame({column: X[column], "verdict": is_anomaly.map({True: "anomaly", False: "normal"})})
    # Always keep every anomaly; thin the normals so the chart stays responsive.
    normal = df[df.verdict == "normal"]
    if len(normal) > max_points:
        normal = normal.sample(max_points, random_state=0)
    df = pd.concat([normal, df[df.verdict == "anomaly"]])
    fig = px.box(df, y=column, color="verdict", points="all",
                 color_discrete_map={"normal": NORMAL_COLOR, "anomaly": ANOMALY_COLOR},
                 title=f"{column}: normal vs anomalous values")
    return fig


def plot_scatter(X: pd.DataFrame, confidence: pd.Series, is_anomaly: pd.Series,
                 x: str | None = None, y: str | None = None, max_points: int = 8000) -> go.Figure:
    """Scatter of two features coloured by confidence. One feature -> value vs row order."""
    cols = list(X.columns)
    x = x or cols[0]
    df = X.copy()
    df["confidence"] = confidence
    df["anomaly"] = is_anomaly
    normal = df[~df["anomaly"]]
    if len(normal) > max_points:
        normal = normal.sample(max_points, random_state=0)
    df = pd.concat([normal, df[df["anomaly"]]])
    df["row"] = df.index.astype(str)

    if y is None and len(cols) > 1:
        y = cols[1]
    if y is None:  # single feature: plot value against position
        df["_position"] = range(len(df)) if not pd.api.types.is_numeric_dtype(df.index) else df.index
        fig = px.scatter(df, x="_position", y=x, color="confidence", symbol="anomaly",
                         color_continuous_scale="Turbo", range_color=(0, 1),
                         hover_data={"row": True, "confidence": ":.2f"},
                         title=f"{x} by row (colour = anomaly confidence)")
        fig.update_xaxes(title="row")
        return fig
    return px.scatter(df, x=x, y=y, color="confidence", symbol="anomaly",
                      color_continuous_scale="Turbo", range_color=(0, 1),
                      hover_data={"row": True, "confidence": ":.2f"},
                      title=f"{x} vs {y} (colour = anomaly confidence)")


def plot_shap_bar(explanation, row=None, top_n: int = 10) -> go.Figure:
    """Mean |SHAP| across explained anomalies, or the SHAP values of one row."""
    if row is None:
        s = explanation.mean_abs().head(top_n).sort_values()
        title, xlabel = "Which features drive anomalies (mean |SHAP|)", "mean |SHAP value|"
        colors = [NORMAL_COLOR] * len(s)
    else:
        sv = explanation.shap_values.loc[row]
        s = sv.reindex(sv.abs().sort_values(ascending=False).index[:top_n]).sort_values()
        title, xlabel = f"Why row {row} was flagged (SHAP)", "SHAP value (→ raises anomaly confidence)"
        colors = [ANOMALY_COLOR if v > 0 else NORMAL_COLOR for v in s]
    fig = go.Figure(go.Bar(x=s.values, y=s.index.astype(str), orientation="h", marker_color=colors,
                           hovertemplate="%{y}: %{x:.4f}<extra></extra>"))
    fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title="feature")
    return fig


def plot_anomaly_heatmap(X: pd.DataFrame, result, top_n: int = 25) -> go.Figure:
    """Top anomalies (rows) x features (columns), coloured by z-score of the value.

    Red = unusually high, blue = unusually low, so you can see *how* each row is odd.
    """
    idx = result.top(top_n).index
    if len(idx) == 0:
        fig = go.Figure()
        fig.update_layout(title="No anomalies detected")
        return fig
    std = X.std().replace(0, 1)
    z = ((X.loc[idx] - X.mean()) / std).clip(-6, 6)
    z.index = [f"{i} (conf {result.confidence[i]:.2f})" for i in idx]
    fig = px.imshow(z, aspect="auto", color_continuous_scale="RdBu_r", zmin=-6, zmax=6,
                    labels=dict(color="z-score"), title=f"Top {len(idx)} anomalies: how far each feature is from normal")
    return fig


def plot_model_agreement(result, top_n: int = 25) -> go.Figure:
    """Which detectors voted for each top anomaly."""
    idx = result.top(top_n).index
    if len(idx) == 0:
        fig = go.Figure()
        fig.update_layout(title="No anomalies detected")
        return fig
    v = result.votes.loc[idx].astype(int)
    v.index = [str(i) for i in idx]
    fig = px.imshow(v, aspect="auto", color_continuous_scale=[(0, "#eeeeee"), (1, ANOMALY_COLOR)],
                    zmin=0, zmax=1, title="Model agreement (red = model flagged the row)")
    fig.update_coloraxes(showscale=False)
    return fig
