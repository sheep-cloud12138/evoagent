from evoagent.core.models import Difficulty, TaskCapability, TaskRequest
from evoagent.core.router import DifficultyAssessor


def test_router_complex_detection() -> None:
    router = DifficultyAssessor()
    task = TaskRequest(
        task_id="t1",
        query="设计一个并行系统并集成数据库和API",
        context={},
    )
    decision = router.decide(task)
    assert decision.difficulty in {Difficulty.MEDIUM, Difficulty.COMPLEX}
    assert 0 <= decision.score <= 1
    assert 0 <= decision.confidence <= 1
    assert decision.fallback_plan
    assert TaskCapability.PLANNING in decision.features.required_capabilities


def test_router_simple_detection() -> None:
    router = DifficultyAssessor()
    task = TaskRequest(task_id="t2", query="总结一句话", context={"has_similar_success": True})
    decision = router.decide(task)
    assert decision.difficulty == Difficulty.SIMPLE


def test_router_low_confidence_auto_escalate(monkeypatch) -> None:
    router = DifficultyAssessor()
    monkeypatch.setattr(router, "_decision_confidence", lambda *args, **kwargs: 0.05)
    task = TaskRequest(task_id="t3", query="总结一句话", context={"has_similar_success": True})
    decision = router.decide(task)
    assert decision.escalated_due_to_low_confidence is True
    assert decision.original_difficulty == Difficulty.SIMPLE
    assert decision.difficulty == Difficulty.MEDIUM


def test_router_cold_start_uses_prior_history_score() -> None:
    router = DifficultyAssessor()
    task = TaskRequest(task_id="t4", query="总结一句话", context={"episode_count": 0, "historical_success_rate": 0.0})
    decision = router.decide(task)
    assert decision.features.historical_pattern_score > 0.0


def test_router_structural_fallback_keeps_report_complex(monkeypatch) -> None:
    router = DifficultyAssessor()
    monkeypatch.setattr(router, "_infer_features_with_llm", lambda query: None)
    task = TaskRequest(task_id="t5", query="请撰写一份 2026 年行业技术趋势研究报告", context={})
    decision = router.decide(task)
    assert decision.difficulty == Difficulty.COMPLEX
    assert decision.features.needs_external_tools is True
    assert TaskCapability.WEB in decision.features.required_capabilities
    assert TaskCapability.PLANNING in decision.features.required_capabilities


def test_router_structural_fallback_keeps_system_design_complex(monkeypatch) -> None:
    router = DifficultyAssessor()
    monkeypatch.setattr(router, "_infer_features_with_llm", lambda query: None)
    task = TaskRequest(task_id="t6", query="我要做一个内容发布系统，帮我设计技术方案并给出关键代码", context={})
    decision = router.decide(task)
    assert decision.difficulty in {Difficulty.MEDIUM, Difficulty.COMPLEX}
    assert decision.features.estimated_steps >= 5
    assert TaskCapability.CODING in decision.features.required_capabilities
    assert TaskCapability.PLANNING in decision.features.required_capabilities


def test_router_structural_fallback_detects_filesystem_capability(monkeypatch) -> None:
    router = DifficultyAssessor()
    monkeypatch.setattr(router, "_infer_features_with_llm", lambda query: None)
    task = TaskRequest(task_id="t7", query="在当前文件夹保存一个总结文档", context={})
    decision = router.decide(task)
    assert TaskCapability.FILESYSTEM in decision.features.required_capabilities
    assert decision.features.needs_external_tools is True
