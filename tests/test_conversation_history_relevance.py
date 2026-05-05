import asyncio

from evoagent.agents.base import BaseSubAgent
from evoagent.core.llm import LLMClient
from evoagent.core.models import AgentOutput, AgentOutput, TaskRequest
from evoagent.core.orchestrator import SubAgentOrchestrator


class _ProbeAgent(BaseSubAgent):
    name = "probe-agent"


def test_subagent_filters_irrelevant_history(monkeypatch) -> None:
    agent = _ProbeAgent(LLMClient())
    captured = {"prompt": ""}

    def fake_generate(prompt, temperature=None, profile=LLMClient.Profile.STANDARD):
        _ = temperature, profile
        captured["prompt"] = prompt
        return "ok"

    monkeypatch.setattr(agent.llm, "generate", fake_generate)

    task = TaskRequest(
        task_id="t1",
        query="请给出两步计划",
        context={
            "conversation_history": [
                {"role": "user", "content": "现在几点"},
                {"role": "assistant", "content": "当前时间：2026-04-20 14:39:00"},
            ]
        },
    )

    _ = asyncio.run(agent.run(task, "直接完成任务并给出最终结果"))
    assert "现在几点" not in captured["prompt"]
    assert "当前时间" not in captured["prompt"]


def test_subagent_keeps_history_for_reference_query(monkeypatch) -> None:
    agent = _ProbeAgent(LLMClient())
    captured = {"prompt": ""}

    def fake_generate(prompt, temperature=None, profile=LLMClient.Profile.STANDARD):
        _ = temperature, profile
        captured["prompt"] = prompt
        return "ok"

    monkeypatch.setattr(agent.llm, "generate", fake_generate)

    task = TaskRequest(
        task_id="t2",
        query="我刚才说我叫什么？",
        context={
            "conversation_history": [
                {"role": "user", "content": "我叫小王"},
                {"role": "assistant", "content": "收到，我记住了"},
            ]
        },
    )

    _ = asyncio.run(agent.run(task, "直接完成任务并给出最终结果"))
    assert "我叫小王" in captured["prompt"]


def test_aggregate_filters_irrelevant_history(monkeypatch) -> None:
    orchestrator = SubAgentOrchestrator(LLMClient())
    captured = {"prompt": ""}

    def fake_generate(prompt, temperature=None, profile=LLMClient.Profile.STANDARD, force_models=None):
        _ = temperature, profile, force_models
        captured["prompt"] = prompt
        return "ok"

    monkeypatch.setattr(orchestrator.llm, "generate", fake_generate)

    task = TaskRequest(
        task_id="t3",
        query="请给出两步计划",
        context={
            "conversation_history": [
                {"role": "user", "content": "现在几点"},
                {"role": "assistant", "content": "当前时间：2026-04-20 14:39:00"},
            ]
        },
    )

    outputs = [AgentOutput(agent_name="code-agent", objective="x", result="两步计划：1. ... 2. ...")]
    _ = orchestrator.aggregate(task, outputs)

    assert "现在几点" not in captured["prompt"]
    assert "当前时间" not in captured["prompt"]
