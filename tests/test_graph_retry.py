import asyncio

from evoagent.app import EvoAgentSystem
from evoagent.core.config import settings
from evoagent.core.models import AgentOutput, Difficulty, TaskDecision, TaskFeatures


def test_graph_retries_failed_agent_run_once(monkeypatch) -> None:
    system = EvoAgentSystem()
    calls = {"execute": 0}
    decision = TaskDecision(
        difficulty=Difficulty.SIMPLE,
        score=0.1,
        reasoning="test",
        features=TaskFeatures(estimated_steps=1, confidence=0.9),
        confidence=0.9,
    )

    async def fake_execute(task, passed_decision):
        _ = task, passed_decision
        calls["execute"] += 1
        if calls["execute"] == 1:
            return [
                AgentOutput(
                    agent_name="reasoning-agent",
                    objective="answer",
                    result="[subagent-error] temporary failure",
                    confidence=0.2,
                    metadata={"failed": True},
                )
            ]
        return [
            AgentOutput(
                agent_name="reasoning-agent",
                objective="answer",
                result="recovered answer",
                confidence=0.9,
                metadata={"failed": False},
            )
        ]

    monkeypatch.setattr(settings, "graph_retry_enabled", True)
    monkeypatch.setattr(settings, "graph_max_retries", 1)
    monkeypatch.setattr(system.router, "decide", lambda task: decision)
    monkeypatch.setattr(system.skill_artifacts, "find_relevant", lambda query: (None, 0.0))
    monkeypatch.setattr(system.orchestrator, "execute", fake_execute)
    monkeypatch.setattr(system.feedback, "run_post_task", lambda **kwargs: {})

    result = asyncio.run(system.run("retry this", context={"session_id": "graph-retry"}))

    assert result.final_answer == "recovered answer"
    assert calls["execute"] == 2
    assert result.metadata["retry_count"] == 1
    assert any(step.get("step") == "graph_retry" for step in result.metadata["trace"])
