"""MCP server exposing the anomaly tools.

Run it standalone (stdio) so any MCP client, including Claude Desktop, can use it:

    PYTHONPATH=src python -m agent.mcp_server
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from agent import tools

mcp = FastMCP("anomaly-platform")


@mcp.tool()
def dataset_analyzer(dataset_path: str) -> dict:
    """Profile an uploaded dataset: column types (numeric / identifier / datetime /
    categorical / text), missing and zero percentages, skew, and suggestions for how to
    analyse it. Call this first when the user asks about a dataset."""
    return tools.dataset_analyzer(dataset_path)


@mcp.tool()
def anomaly_explainer(dataset_path: str, row_ids: list[str] | None = None, top_n: int = 5,
                      columns: list[str] | None = None, exclude_zeros: bool = False) -> dict:
    """Run the 4-model anomaly ensemble on a dataset and explain flagged rows in plain English.

    With no row_ids it explains the top_n most confident anomalies. With row_ids it explains
    those specific rows (and says so if a row was NOT flagged). Returns confidence, which
    models agreed, per-feature z-scores and SHAP drivers. Set exclude_zeros for zero-heavy
    data such as rainfall."""
    return tools.anomaly_explainer(dataset_path, row_ids, top_n, columns, exclude_zeros)


if __name__ == "__main__":
    mcp.run()  # stdio transport
