import asyncio

from evoagent.app import EvoAgentSystem
from evoagent.core.config import settings
from evoagent.core.models import AgentOutput, Difficulty, TaskDecision, TaskFeatures


def _complex_decision() -> TaskDecision:
    return TaskDecision(
        difficulty=Difficulty.COMPLEX,
        score=0.82,
        reasoning="test",
        features=TaskFeatures(
            estimated_steps=5,
            needs_external_tools=True,
            confidence=0.4,
        ),
        confidence=0.4,
    )


def test_old_orchestrator_reflection_is_not_applied(monkeypatch) -> None:
    system = EvoAgentSystem()
    monkeypatch.setattr(settings, "reflection_enabled", True)
    monkeypatch.setattr(settings, "reflection_max_rounds", 1)
    monkeypatch.setattr(system.router, "decide", lambda task: _complex_decision())
    monkeypatch.setattr(
        system.skill_artifacts, "find_relevant", lambda query: (None, 0.0)
    )
    monkeypatch.setattr(system.feedback, "run_post_task", lambda **kwargs: {})

    async def fake_execute(task, decision):
        _ = task, decision
        return [
            AgentOutput(
                agent_name="search-agent",
                objective="find evidence",
                result="No tool-backed execution result was produced yet.",
                confidence=0.55,
                metadata={"failed": False},
            ),
            AgentOutput(
                agent_name="reasoning-agent",
                objective="draft answer",
                result="Current plan is still a proposal and needs verification.",
                confidence=0.6,
                metadata={"failed": False},
            ),
        ]

    monkeypatch.setattr(system.orchestrator, "execute", fake_execute)
    monkeypatch.setattr(
        system.orchestrator, "aggregate", lambda task, outputs: "baseline answer"
    )

    def should_not_run(*args, **kwargs):
        raise AssertionError("old orchestrator reflection should not call llm.generate")

    monkeypatch.setattr(system.llm, "generate", should_not_run)

    result = asyncio.run(
        system.run(
            "请处理一个复杂排障任务", context={"session_id": "reflection-revise"}
        )
    )

    assert result.final_answer == "baseline answer"
    assert not any(
        step.get("step") == "reflection" for step in result.metadata.get("trace", [])
    )


def test_reflection_disabled_keeps_baseline(monkeypatch) -> None:
    system = EvoAgentSystem()
    monkeypatch.setattr(settings, "reflection_enabled", False)
    monkeypatch.setattr(system.router, "decide", lambda task: _complex_decision())
    monkeypatch.setattr(
        system.skill_artifacts, "find_relevant", lambda query: (None, 0.0)
    )
    monkeypatch.setattr(system.feedback, "run_post_task", lambda **kwargs: {})

    async def fake_execute(task, decision):
        _ = task, decision
        return [
            AgentOutput(
                agent_name="reasoning-agent",
                objective="draft answer",
                result="Baseline",
                confidence=0.7,
                metadata={"failed": False},
            ),
            AgentOutput(
                agent_name="search-agent",
                objective="support",
                result="Support",
                confidence=0.7,
                metadata={"failed": False},
            ),
        ]

    monkeypatch.setattr(system.orchestrator, "execute", fake_execute)
    monkeypatch.setattr(
        system.orchestrator, "aggregate", lambda task, outputs: "baseline answer"
    )

    def should_not_run(*args, **kwargs):
        raise AssertionError("old orchestrator reflection should not call llm.generate")

    monkeypatch.setattr(system.llm, "generate", should_not_run)

    result = asyncio.run(
        system.run("请处理一个复杂任务", context={"session_id": "reflection-disabled"})
    )

    assert result.final_answer == "baseline answer"
    assert not any(
        step.get("step") == "reflection" for step in result.metadata.get("trace", [])
    )
