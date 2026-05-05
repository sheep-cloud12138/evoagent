import asyncio

from evoagent.agents.specialists import ToolAgent
from evoagent.core.llm import LLMClient
from evoagent.core.models import AgentOutput, Difficulty, TaskCapability, TaskDecision, TaskFeatures, TaskRequest
from evoagent.core.orchestrator import SubAgentOrchestrator
from evoagent.core.tools import execute_tool_with_meta


def test_write_text_file_tool_writes_real_file(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    evt = execute_tool_with_meta(
        "write_text_file",
        {
            "path": "hello_world.cpp",
            "content": "#include <iostream>\n",
            "overwrite": True,
            "create_parents": True,
        },
    )
    assert evt["success"] is True
    assert (tmp_path / "hello_world.cpp").exists()


def test_filesystem_capability_uses_file_agent_plan() -> None:
    orchestrator = SubAgentOrchestrator(LLMClient())
    task = TaskRequest(task_id="t1", query="在当前文件夹写一个hello wrold的cpp程序", context={})
    decision = TaskDecision(
        difficulty=Difficulty.SIMPLE,
        score=0.2,
        reasoning="test",
        features=TaskFeatures(required_capabilities=[TaskCapability.FILESYSTEM]),
    )
    plan = orchestrator._spawn_plan(task, decision)
    assert [agent.name for agent, _ in plan] == ["file-agent", "code-agent"]


def test_complex_query_uses_explicit_capability_agents() -> None:
    orchestrator = SubAgentOrchestrator(LLMClient())
    task = TaskRequest(task_id="t1r", query="实现端到端系统并落地外部依赖", context={})
    decision = TaskDecision(
        difficulty=Difficulty.COMPLEX,
        score=0.9,
        reasoning="test",
        features=TaskFeatures(
            required_capabilities=[
                TaskCapability.WEB,
                TaskCapability.INTEGRATION,
                TaskCapability.CODING,
            ],
        ),
    )
    plan = orchestrator._spawn_plan(task, decision)
    assert [agent.name for agent, _ in plan] == ["search-agent", "integration-agent", "code-agent"]


def test_tool_agent_uses_generic_function_call_trace(monkeypatch) -> None:
    agent = ToolAgent(LLMClient())

    def fake_generate_with_tools_trace(prompt, tool_names, temperature=None, profile=LLMClient.Profile.STANDARD):
        _ = prompt, temperature, profile
        assert "write_text_file" in tool_names
        return (
            "已完成通用工具执行",
            [
                {
                    "name": "write_text_file",
                    "category": "filesystem",
                    "success": True,
                    "result": "ok",
                }
            ],
        )

    monkeypatch.setattr(
        agent.llm,
        "generate_with_tools_trace",
        fake_generate_with_tools_trace,
    )

    task = TaskRequest(task_id="t2", query="在当前文件夹写一个leetcode 第一题的标准解法", context={})
    out = asyncio.run(agent.run(task, "执行通用工具流程"))

    assert out.result == "已完成通用工具执行"
    events = out.metadata.get("tool_events", [])
    assert isinstance(events, list) and len(events) >= 1
    assert events[0].get("category") == "filesystem"
    assert events[0].get("success") is True
    assert out.metadata.get("failed") is None or out.metadata.get("failed") is False


def test_tool_agent_omits_filesystem_tools_without_explicit_local_file_request(monkeypatch) -> None:
    agent = ToolAgent(LLMClient())

    def fake_generate_with_tools_trace(prompt, tool_names, temperature=None, profile=LLMClient.Profile.STANDARD):
        _ = prompt, temperature, profile
        assert "write_text_file" not in tool_names
        assert "read_text_file" not in tool_names
        assert "list_directory" not in tool_names
        return ("ok", [])

    monkeypatch.setattr(agent.llm, "generate_with_tools_trace", fake_generate_with_tools_trace)

    task = TaskRequest(task_id="t2b", query="帮我撰写一份行业技术趋势研究报告", context={})
    out = asyncio.run(agent.run(task, "识别需要调用的工具链与依赖"))
    assert out.result == "ok"


def test_tool_agent_handles_no_tool_event(monkeypatch) -> None:
    agent = ToolAgent(LLMClient())
    monkeypatch.setattr(
        agent.llm,
        "generate_with_tools_trace",
        lambda prompt, tool_names, temperature=None, profile=LLMClient.Profile.STANDARD: (
            "无需工具，直接回答",
            [],
        ),
    )

    task = TaskRequest(task_id="t2r", query="解释算法复杂度", context={})
    out = asyncio.run(agent.run(task, "通用推理"))

    assert out.result == "无需工具，直接回答"
    events = out.metadata.get("tool_events", [])
    assert events == []


def test_tool_success_survives_llm_fallback(monkeypatch) -> None:
    agent = ToolAgent(LLMClient())

    def fake_generate_with_tools_trace(
        prompt, tool_names, temperature=None, profile=LLMClient.Profile.STANDARD
    ):
        _ = prompt, tool_names, temperature, profile
        return (
            "[Fallback-LLM] offline",
            [
                {
                    "name": "read_text_file",
                    "source": "builtin",
                    "category": "filesystem",
                    "success": True,
                    "result": "README says EvoAgent uses LangGraph and FastAPI.",
                }
            ],
        )

    monkeypatch.setattr(
        agent.llm,
        "generate_with_tools_trace",
        fake_generate_with_tools_trace,
    )

    task = TaskRequest(task_id="t2c", query="读取当前文件夹 README.md", context={})
    out = asyncio.run(agent.run(task, "读取文件并返回真实结果"))

    assert out.metadata.get("failed") is None
    assert out.metadata.get("tool_evidence_only") is True
    assert "工具执行事实" in out.result
    assert "LangGraph and FastAPI" in out.result
    assert out.confidence > 0.2


def test_execution_aggregate_uses_generic_fallback_merge(monkeypatch) -> None:
    orchestrator = SubAgentOrchestrator(LLMClient())
    monkeypatch.setattr(
        orchestrator.llm,
        "generate",
        lambda *args, **kwargs: "[Fallback-LLM] offline",
    )

    task = TaskRequest(task_id="t3", query="在当前文件夹写一个hello wrold的cpp程序", context={})
    outputs = [
        AgentOutput(
            agent_name="tool-agent",
            objective="test",
            result="已完成",
            confidence=0.9,
            metadata={"failed": False, "tool_events": []},
        )
    ]
    answer = orchestrator.aggregate(task, outputs)
    assert "最终整合结果（确定性兜底）" in answer


def test_aggregate_promotes_tool_evidence(monkeypatch) -> None:
    orchestrator = SubAgentOrchestrator(LLMClient())
    captured = {"prompt": ""}

    def fake_generate(prompt, *args, **kwargs):
        _ = args, kwargs
        captured["prompt"] = prompt
        return "ok"

    monkeypatch.setattr(orchestrator.llm, "generate", fake_generate)

    task = TaskRequest(task_id="t4", query="分析读取到的项目文件", context={})
    outputs = [
        AgentOutput(
            agent_name="file-agent",
            objective="read files",
            result="模型总结",
            confidence=0.7,
            metadata={
                "tool_events": [
                    {
                        "name": "read_text_file",
                        "source": "builtin",
                        "category": "filesystem",
                        "success": True,
                        "result": "pyproject depends on fastapi and langgraph",
                    }
                ]
            },
        )
    ]

    assert orchestrator.aggregate(task, outputs) == "ok"
    assert "工具事实" in captured["prompt"]
    assert "fastapi and langgraph" in captured["prompt"]


def test_aggregate_replaces_answer_that_ignores_successful_tools(monkeypatch) -> None:
    orchestrator = SubAgentOrchestrator(LLMClient())
    monkeypatch.setattr(
        orchestrator.llm,
        "generate",
        lambda *args, **kwargs: "无法读取这些文件，请提供文件内容。",
    )

    task = TaskRequest(task_id="t5", query="分析读取到的项目文件", context={})
    outputs = [
        AgentOutput(
            agent_name="file-agent",
            objective="read files",
            result="模型总结",
            confidence=0.7,
            metadata={
                "tool_events": [
                    {
                        "name": "read_text_file",
                        "source": "builtin",
                        "category": "filesystem",
                        "success": True,
                        "result": "README says EvoAgent has layered memory.",
                    }
                ]
            },
        )
    ]

    answer = orchestrator.aggregate(task, outputs)

    assert "最终整合结果（确定性兜底）" in answer
    assert "工具事实摘录" in answer
    assert "layered memory" in answer
