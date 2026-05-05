import json

from evoagent.core.tools import (
    execute_tool,
    execute_tool_with_meta,
    get_tool_specs,
    register_semantic_retriever,
    reset_tool_stats,
    resolve_tool_names,
)
from evoagent.agents.specialists import SearchAgent, ToolAgent
from evoagent.core.llm import LLMClient


def test_tool_specs_resolve() -> None:
    specs = get_tool_specs(("get_current_time", "fetch_url_text", "mcp_call_tool"))
    names = [s["function"]["name"] for s in specs]
    assert "get_current_time" in names
    assert "fetch_url_text" in names
    assert "mcp_call_tool" in names
    assert all("source" in s for s in specs)


def test_mcp_tool_without_server_returns_error() -> None:
    out = execute_tool("mcp_call_tool", {"tool_name": "x", "arguments": {}})
    assert "MCP server is not configured" in out


def test_fetch_url_blocks_localhost_by_default() -> None:
    out = execute_tool("fetch_url_text", {"url": "http://127.0.0.1:1234"})
    assert "private or local network targets are blocked" in out


def test_tool_agent_enables_function_call() -> None:
    agent = ToolAgent(LLMClient())
    assert agent.enable_function_call is True
    assert "integration" in agent.tool_categories


def test_tool_filter_excludes_low_success_rate_tool() -> None:
    reset_tool_stats()
    execute_tool_with_meta("mcp_call_tool", {"tool_name": "x", "arguments": {}})
    selected = resolve_tool_names(categories=("integration",), min_success_rate=0.9)
    assert "mcp_call_tool" not in selected


def test_semantic_memory_search_tool_works_with_registered_retriever() -> None:
    register_semantic_retriever(
        lambda query, top_k, use_negative: [
            {
                "id": "episode:test",
                "fact": f"hit:{query}",
                "metadata": {"k": top_k, "negative": use_negative},
                "score": 0.9,
            }
        ]
    )

    out = execute_tool("semantic_memory_search", {"query": "python async", "top_k": 2})
    payload = json.loads(out)
    assert payload["query"] == "python async"
    assert payload["top_k"] == 2
    assert payload["hits"][0]["id"] == "episode:test"


def test_search_agent_enables_retrieval_function_call() -> None:
    agent = SearchAgent(LLMClient())
    assert agent.enable_function_call is True
    assert "retrieval" in agent.tool_categories
