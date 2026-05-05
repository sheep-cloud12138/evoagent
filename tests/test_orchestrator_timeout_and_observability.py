import asyncio
import json

from evoagent.agents.base import BaseSubAgent
from evoagent.core.config import settings
from evoagent.core.llm import LLMClient
from evoagent.core.models import AgentOutput, Difficulty, TaskDecision, TaskFeatures, TaskRequest
from evoagent.core.observability import RealtimeExecutionObserver
from evoagent.core.orchestrator import SubAgentOrchestrator


class _FastAgent(BaseSubAgent):
    name = "fast-agent"

    async def run(self, task: TaskRequest, objective: str) -> AgentOutput:
        return AgentOutput(agent_name=self.name, objective=objective, result="FAST", metadata={"tool_events": []})


class _SlowAgent(BaseSubAgent):
    name = "slow-agent"

    async def run(self, task: TaskRequest, objective: str) -> AgentOutput:
        await asyncio.sleep(0.2)
        return AgentOutput(agent_name=self.name, objective=objective, result="SLOW")


def _decision() -> TaskDecision:
    return TaskDecision(
        difficulty=Difficulty.COMPLEX,
        score=0.9,
        reasoning="test",
        features=TaskFeatures(estimated_steps=5, needs_external_tools=True),
    )


def test_orchestrator_timeout_returns_partial_results_and_logs(monkeypatch, tmp_path) -> None:
    observer = RealtimeExecutionObserver(
        enabled=True,
        event_log_path=tmp_path / "obs" / "events.jsonl",
        metrics_path=tmp_path / "obs" / "metrics.json",
        emit_stdout=False,
    )
    orchestrator = SubAgentOrchestrator(LLMClient(), observer=observer)

    monkeypatch.setattr(settings, "orchestrator_task_timeout_seconds", 0.05)
    monkeypatch.setattr(
        orchestrator,
        "_spawn_plan",
        lambda task, decision: [(_FastAgent(orchestrator.llm), "fast"), (_SlowAgent(orchestrator.llm), "slow")],
    )

    task = TaskRequest(task_id="t1", query="complex", context={})
    outputs = asyncio.run(orchestrator.execute(task, _decision()))

    assert len(outputs) == 2
    assert any(o.result == "FAST" for o in outputs)
    assert any(bool(o.metadata.get("timed_out", False)) for o in outputs)

    event_lines = (tmp_path / "obs" / "events.jsonl").read_text(encoding="utf-8").strip().splitlines()
    parsed = [json.loads(line) for line in event_lines if line.strip()]
    assert any(e.get("event") == "subagent_start" for e in parsed)
    assert any(e.get("event") == "subagent_end" for e in parsed)
