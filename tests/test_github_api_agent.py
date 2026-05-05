from evoagent.core.llm import LLMClient
from evoagent.core.models import Difficulty, TaskCapability, TaskDecision, TaskFeatures, TaskRequest
from evoagent.core.orchestrator import SubAgentOrchestrator


def test_external_research_query_uses_generic_medium_plan() -> None:
    orchestrator = SubAgentOrchestrator(LLMClient())
    decision = TaskDecision(
        difficulty=Difficulty.MEDIUM,
        score=0.5,
        reasoning="test",
        features=TaskFeatures(
            required_capabilities=[TaskCapability.RETRIEVAL, TaskCapability.WEB, TaskCapability.PLANNING],
        ),
    )
    task = TaskRequest(task_id="t1", query="查询一个外部技术榜单并整理结果", context={})
    plan = orchestrator._spawn_plan(task, decision)

    # Removed source-specific API subagents; routing is now capability-driven.
    assert [agent.name for agent, _ in plan] == ["search-agent", "reasoning-agent"]
