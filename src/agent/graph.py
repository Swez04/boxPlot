"""LangGraph agent loop: the LLM decides when to call tools, tools run, repeat."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agent import tools as local_tools

DEFAULT_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5-5")

SYSTEM_PROMPT = """You are an anomaly-detection analyst inside a data platform.
The user has uploaded a dataset located at: {dataset_path}

Use your tools to answer. Call dataset_analyzer to understand the data, and
anomaly_explainer to find and explain anomalies (pass row_ids when the user asks about
specific rows). Rules:
- Only state numbers that appear in tool results; never invent values or rows.
- Explain in plain English for a non-expert: what is unusual, how unusual, and why the
  models flagged it. Mention confidence and how many models agreed.
- If a row was not flagged, say so and explain why using its confidence.
- If the SHAP fidelity is low (< 0.5), say the explanation is approximate.
- Keep answers concise."""


def local_tool_list() -> list[StructuredTool]:
    """The tools wrapped directly (no MCP hop). Used in tests and as a fallback."""
    return [
        StructuredTool.from_function(local_tools.dataset_analyzer, name="dataset_analyzer",
                                     description=local_tools.dataset_analyzer.__doc__),
        StructuredTool.from_function(local_tools.anomaly_explainer, name="anomaly_explainer",
                                     description=local_tools.anomaly_explainer.__doc__),
    ]


async def mcp_tool_list() -> list[Any]:
    """Load the tools from the MCP server, started as a stdio subprocess."""
    from langchain_mcp_adapters.client import MultiServerMCPClient

    src = str(Path(__file__).resolve().parents[1])
    client = MultiServerMCPClient({
        "anomaly": {
            "command": sys.executable,
            "args": ["-m", "agent.mcp_server"],
            "transport": "stdio",
            "env": {**os.environ, "PYTHONPATH": src},
        }
    })
    return await client.get_tools()


def make_llm(model: str | None = None):
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is not set; the chat agent needs it")
    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(model=model or DEFAULT_MODEL, temperature=0, max_tokens=1500)


def build_agent(llm, tool_list: list[Any], checkpointer=None):
    """Compile the agent graph: START -> agent -> (tools -> agent)* -> END."""
    llm_with_tools = llm.bind_tools(tool_list)

    async def call_model(state: MessagesState):
        return {"messages": [await llm_with_tools.ainvoke(state["messages"])]}

    g = StateGraph(MessagesState)
    g.add_node("agent", call_model)
    g.add_node("tools", ToolNode(tool_list))
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", tools_condition)
    g.add_edge("tools", "agent")
    return g.compile(checkpointer=checkpointer or MemorySaver())


class AnomalyAgent:
    """Chat wrapper: one conversation thread per (session, dataset)."""

    def __init__(self, llm=None, tool_list: list[Any] | None = None, use_mcp: bool = True) -> None:
        self._llm = llm
        self._tools = tool_list
        self._use_mcp = use_mcp
        self._graph = None
        self._seen_threads: set[str] = set()

    async def _ensure(self) -> None:
        if self._graph is not None:
            return
        llm = self._llm or make_llm()
        tool_list = self._tools
        if tool_list is None:
            tool_list = await mcp_tool_list() if self._use_mcp else local_tool_list()
        self._graph = build_agent(llm, tool_list)

    async def ask(self, question: str, dataset_path: str, thread_id: str, max_steps: int = 12) -> dict:
        await self._ensure()
        messages: list = []
        if thread_id not in self._seen_threads:
            messages.append(SystemMessage(SYSTEM_PROMPT.format(dataset_path=dataset_path)))
            self._seen_threads.add(thread_id)
        messages.append(HumanMessage(question))
        out = await self._graph.ainvoke(
            {"messages": messages},
            config={"configurable": {"thread_id": thread_id}, "recursion_limit": max_steps * 2 + 1},
        )
        msgs = out["messages"]
        last_q = max(i for i, m in enumerate(msgs) if isinstance(m, HumanMessage))  # this turn only
        calls = [tc["name"] for m in msgs[last_q:] if isinstance(m, AIMessage) for tc in (m.tool_calls or [])]
        answer = out["messages"][-1].content
        if isinstance(answer, list):  # Claude returns content blocks
            answer = "".join(b.get("text", "") for b in answer if isinstance(b, dict))
        return {"answer": answer, "tools_used": calls}
