from evoagent.core.llm import LLMClient
from evoagent.core.models import Difficulty, TaskCapability, TaskDecision, TaskFeatures, TaskRequest
from evoagent.core.orchestrator import SubAgentOrchestrator


def _simple_decision() -> TaskDecision:
    return TaskDecision(
        difficulty=Difficulty.SIMPLE,
        score=0.2,
        reasoning="test",
        features=TaskFeatures(),
    )


def test_simple_task_without_capability_uses_reasoning_agent() -> None:
    orchestrator = SubAgentOrchestrator(LLMClient())
    task = TaskRequest(
        task_id="t1",
        query="总结一句话",
    )
    plan = orchestrator._spawn_plan(task, _simple_decision())
    assert len(plan) == 1
    assert plan[0][0].name == "reasoning-agent"


def test_github_skill_query_uses_capability_plan_not_keyword_shortcut() -> None:
    orchestrator = SubAgentOrchestrator(LLMClient())
    decision = TaskDecision(
        difficulty=Difficulty.SIMPLE,
        score=0.2,
        reasoning="test",
        features=TaskFeatures(
            required_capabilities=[TaskCapability.WEB, TaskCapability.SKILL],
        ),
    )
    task = TaskRequest(
        task_id="t1b",
        query="查询一天github上最近热门的十个skills并给你 自己下载",
    )
    plan = orchestrator._spawn_plan(task, decision)
    assert [agent.name for agent, _ in plan] == ["search-agent", "skill-agent"]


def test_code_generation_task_uses_single_code_agent_even_when_medium() -> None:
    orchestrator = SubAgentOrchestrator(LLMClient())
    decision = TaskDecision(
        difficulty=Difficulty.MEDIUM,
        score=0.6,
        reasoning="test",
        features=TaskFeatures(
            estimated_steps=4,
            needs_external_tools=True,
            has_historical_pattern=False,
            historical_pattern_score=0.0,
            confidence=0.6,
            required_capabilities=[TaskCapability.CODING],
        ),
        confidence=0.6,
        fallback_plan="test",
    )
    task = TaskRequest(
        task_id="t2",
        query="写一个能处理并发请求的 Python 爬虫，要有错误处理",
    )

    plan = orchestrator._spawn_plan(task, decision)

    assert len(plan) == 1
    assert plan[0][0].name == "code-agent"
