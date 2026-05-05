from evoagent.core.feedback import FeedbackLoop
from evoagent.core.models import AgentOutput
from evoagent.core.models import TaskRequest
from evoagent.core.router import DifficultyAssessor


class _DummyEpisodic:
    def add(self, **kwargs):
        return None


class _DummyMemory:
    def __init__(self):
        self.episodic = _DummyEpisodic()
        self.semantic = self

    def upsert_fact(self, **kwargs):
        return None

    def distill_episode_to_semantic(
        self,
        task_id: str,
        summary: str,
        score: float,
        success: bool,
        failure_reason: str | None = None,
    ):
        return None


class _DummyEvolution:
    def __init__(self):
        self.registry = type("R", (), {"decay_and_archive": lambda self, **k: 0})()
        self.called = False

    def identify_gap(self, quality_score: float, path_length: int, confidence: float) -> bool:
        return False

    def generate_skill(self, task_text: str, failure_reason: str):
        self.called = True
        return type("D", (), {"name": "x", "description": "d", "tags": ["t"], "code": "def execute(input_text: str) -> str:\n    return input_text", "tests": "from skill_impl import execute\n\ndef test_x():\n    assert execute('a') == 'a'"})()

    def validate_and_register(self, draft):
        return True, "registered"


def test_failure_marker_forces_evolution() -> None:
    evo = _DummyEvolution()
    fb = FeedbackLoop(router=DifficultyAssessor(), memory=_DummyMemory(), evolution=evo)
    actions = fb.run_post_task(
        task_id="t",
        task_text="q",
        final_answer="执行 failed: tool returned invalid schema",
        quality_score=0.9,
        path_length=1,
        confidence=0.9,
    )
    assert actions["skill_evolution"] == "registered"
    assert evo.called


def test_low_value_fallback_skips_skill_evolution() -> None:
    evo = _DummyEvolution()
    fb = FeedbackLoop(router=DifficultyAssessor(), memory=_DummyMemory(), evolution=evo)
    actions = fb.run_post_task(
        task_id="t-low",
        task_text="q-low",
        final_answer="最终整合结果（确定性兜底）\n任务: x\n结论: 无可用降级方案",
        quality_score=0.4,
        path_length=2,
        confidence=0.2,
    )
    assert actions["skill_evolution"] == "skipped_low_value_fallback"
    assert not evo.called


def test_provider_timeout_skips_skill_evolution() -> None:
    evo = _DummyEvolution()
    fb = FeedbackLoop(router=DifficultyAssessor(), memory=_DummyMemory(), evolution=evo)
    actions = fb.run_post_task(
        task_id="t-timeout",
        task_text="q-timeout",
        final_answer="[subagent-timeout] 任务级超时触发，使用部分结果继续聚合。",
        quality_score=0.2,
        path_length=2,
        confidence=0.2,
    )
    assert actions["skill_evolution"] == "skipped_provider_failure"
    assert not evo.called


def test_tool_failure_is_classified_and_router_not_polluted() -> None:
    evo = _DummyEvolution()
    fb = FeedbackLoop(router=DifficultyAssessor(), memory=_DummyMemory(), evolution=evo)
    decision = fb.router.decide(TaskRequest(task_id="x", query="查询工具调用", context={}))
    outputs = [
        AgentOutput(
            agent_name="tool-agent",
            objective="obj",
            result="failed",
            metadata={
                "tool_events": [
                    {
                        "name": "mcp_call_tool",
                        "source": "mcp",
                        "category": "integration",
                        "success": False,
                        "error": "timeout",
                    }
                ]
            },
        )
    ]
    actions = fb.run_post_task(
        task_id="t2",
        task_text="q2",
        final_answer="调用失败",
        quality_score=0.3,
        path_length=1,
        confidence=0.2,
        decision=decision,
        outputs=outputs,
        semantic_hints=[{"id": "x", "fact": "y", "metadata": {}}],
        trace=[],
    )
    assert actions["failure_class"] == "tool_failure"
    assert actions["router"] == "skipped_non_routing_failure"
