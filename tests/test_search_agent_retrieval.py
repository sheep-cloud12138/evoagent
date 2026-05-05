import asyncio

from evoagent.agents.specialists import SearchAgent
from evoagent.core.llm import LLMClient
from evoagent.core.models import TaskRequest


def test_search_agent_selects_semantic_retrieval_tool(monkeypatch) -> None:
    agent = SearchAgent(LLMClient())

    def fake_generate_with_tools_trace(prompt, tool_names, temperature=None, profile=LLMClient.Profile.STANDARD):
        _ = prompt, temperature, profile
        assert "semantic_memory_search" in tool_names
        return (
            "已完成检索",
            [
                {
                    "name": "semantic_memory_search",
                    "category": "retrieval",
                    "success": True,
                    "result": "{\"hits\":[]}",
                }
            ],
        )

    monkeypatch.setattr(agent.llm, "generate_with_tools_trace", fake_generate_with_tools_trace)

    task = TaskRequest(task_id="s1", query="总结之前关于并发控制的经验", context={})
    out = asyncio.run(agent.run(task, "检索相关经验并归纳"))

    assert out.result == "已完成检索"
    assert out.metadata.get("tool_events", [])[0].get("category") == "retrieval"
