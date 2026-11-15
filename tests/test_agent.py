import asyncio

import pytest

pytest.importorskip("torch")

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from agent import tools
from agent.graph import AnomalyAgent, local_tool_list


class FakeToolLLM(GenericFakeChatModel):
    """Scripted chat model that tolerates bind_tools()."""

    def bind_tools(self, *_, **__):
        return self


def call(name, args, id_="c1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": id_, "type": "tool_call"}])


# ------------------------------------------------------------------- tools
def test_dataset_analyzer(csv_dataset):
    p = tools.dataset_analyzer(csv_dataset)
    kinds = {c["name"]: c["kind"] for c in p["columns"]}
    assert kinds == {"sensor_id": "identifier", "temp": "numeric", "pressure": "numeric"}
    assert p["recommended_columns"] == ["temp", "pressure"]


def test_anomaly_explainer_top(csv_dataset):
    out = tools.anomaly_explainer(csv_dataset, top_n=3)
    assert out["columns_analyzed"] == ["temp", "pressure"]
    assert out["anomalies_found"] >= 2
    flagged_rows = {e["row"] for e in out["explanations"]}
    assert {7, 8} <= flagged_rows
    e = next(e for e in out["explanations"] if e["row"] == 7)
    assert e["flagged"] and "anomaly" in e["explanation"] and e["shap_drivers"]


def test_anomaly_explainer_specific_rows_including_normal(csv_dataset):
    out = tools.anomaly_explainer(csv_dataset, row_ids=["7", "150", "99999"])
    by_row = {e["row"]: e for e in out["explanations"]}
    assert by_row[7]["flagged"]
    assert not by_row[150]["flagged"] and "NOT flagged" in by_row[150]["explanation"]
    assert out["rows_not_found"] == ["99999"]


def test_path_sandbox(csv_dataset, tmp_path):
    with pytest.raises(PermissionError):
        tools.dataset_analyzer("/etc/passwd")
    with pytest.raises(PermissionError):
        tools.dataset_analyzer("../../etc/passwd")
    with pytest.raises(FileNotFoundError):
        tools.dataset_analyzer("missing.csv")


def test_too_few_rows(tmp_path, monkeypatch):
    monkeypatch.setenv("ANOMALY_DATA_DIR", str(tmp_path))
    (tmp_path / "tiny.csv").write_text("a\n1\n2\n3\n")
    with pytest.raises(ValueError):
        tools.anomaly_explainer("tiny.csv")


# ------------------------------------------------------------- agent loop
def test_agent_loop_calls_tool_then_answers(csv_dataset):
    llm = FakeToolLLM(messages=iter([
        call("anomaly_explainer", {"dataset_path": csv_dataset, "row_ids": ["7"]}),
        AIMessage(content="Row 7 is unusual: temperature is far above normal."),
    ]))
    agent = AnomalyAgent(llm=llm, tool_list=local_tool_list())
    res = asyncio.run(agent.ask("Why is row 7 an anomaly?", csv_dataset, "t1"))
    assert res["tools_used"] == ["anomaly_explainer"]
    assert "Row 7" in res["answer"]


def test_agent_follow_up_only_reports_this_turns_tools(csv_dataset):
    llm = FakeToolLLM(messages=iter([
        call("dataset_analyzer", {"dataset_path": csv_dataset}),
        AIMessage(content="Two numeric columns."),
        AIMessage(content="No tools needed for this one."),
    ]))
    agent = AnomalyAgent(llm=llm, tool_list=local_tool_list())

    async def run():
        a = await agent.ask("What is in the data?", csv_dataset, "t2")
        b = await agent.ask("Thanks, summarise.", csv_dataset, "t2")
        return a, b

    a, b = asyncio.run(run())
    assert a["tools_used"] == ["dataset_analyzer"]
    assert b["tools_used"] == []


def test_agent_needs_api_key_without_llm(csv_dataset, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    agent = AnomalyAgent(tool_list=local_tool_list())
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        asyncio.run(agent.ask("hi", csv_dataset, "t3"))


# --------------------------------------------------------- real MCP server
def test_mcp_server_over_stdio(csv_dataset):
    """Spawn the real MCP server as a subprocess and call a tool through it."""
    from agent.graph import mcp_tool_list

    async def run():
        found = await mcp_tool_list()
        names = {t.name for t in found}
        analyzer = next(t for t in found if t.name == "dataset_analyzer")
        result = await analyzer.ainvoke({"dataset_path": csv_dataset})
        return names, result

    names, result = asyncio.run(run())
    assert names == {"dataset_analyzer", "anomaly_explainer"}
    assert "temp" in str(result)
