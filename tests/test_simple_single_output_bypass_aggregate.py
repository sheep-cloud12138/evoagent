import asyncio

from evoagent.app import EvoAgentSystem
from evoagent.core.models import AgentOutput, Difficulty, TaskDecision, TaskFeatures


def test_simple_single_output_skips_aggregate(monkeypatch) -> None:
    system = EvoAgentSystem()

    decision = TaskDecision(
        difficulty=Difficulty.SIMPLE,
        score=0.1,
        reasoning="test",
        features=TaskFeatures(estimated_steps=1, needs_external_tools=False, confidence=0.9),
        confidence=0.9,
    )

    async def fake_execute(task, passed_decision):
        _ = task, passed_decision
        return [
            AgentOutput(
                agent_name="code-agent",
                objective="直接完成任务并给出最终结果",
                result="两步计划：1. 明确目标 2. 执行并复盘",
                confidence=0.8,
                metadata={"failed": False},
            )
        ]

    monkeypatch.setattr(system.router, "decide", lambda task: decision)
    monkeypatch.setattr(system.orchestrator, "execute", fake_execute)

    def aggregate_should_not_be_called(task, outputs):
        _ = task, outputs
        raise AssertionError("aggregate should be bypassed for simple single output")

    monkeypatch.setattr(system.orchestrator, "aggregate", aggregate_should_not_be_called)

    result = asyncio.run(system.run("请给出两步计划", context={"session_id": "test-simple-bypass"}))
    assert result.final_answer == "两步计划：1. 明确目标 2. 执行并复盘"
    assert any(step.get("step") == "aggregate_skipped_simple_single" for step in result.metadata.get("trace", []))
