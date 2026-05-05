from __future__ import annotations

from uuid import uuid4

from evoagent.core import graph as graph_module
from evoagent.core.config import settings
from evoagent.core.graph import reflection_node, should_continue
from evoagent.llm.structured import Plan, ReflectionResult, StepSpec


def _state(confidence: float = 0.9) -> dict:
    return {
        "run_id": "r",
        "user_query": "hello",
        "plan": {"steps": [{"sequence": 1, "role": "researcher", "description": "d"}]},
        "steps": [{"status": "succeeded"}],
        "current_step": 1,
        "worker_outputs": [{"role": "researcher", "output": "ok", "confidence": confidence}],
        "confidence": confidence,
        "metadata": {},
    }


def test_reflection_not_triggered_when_confidence_high(monkeypatch) -> None:
    monkeypatch.setattr(settings, "reflection_enabled", True)
    assert should_continue(_state(0.9)) == "reporter"


def test_reflection_triggered_when_confidence_low(monkeypatch) -> None:
    monkeypatch.setattr(settings, "reflection_enabled", True)
    assert should_continue(_state(0.2)) == "reflection"


def test_reflection_done_flag_prevents_double_reflection(monkeypatch) -> None:
    monkeypatch.setattr(settings, "reflection_enabled", True)
    state = _state(0.2)
    state["metadata"]["reflection_done"] = True
    assert should_continue(state) == "reporter"


def test_reflection_result_no_revision_passes_through(monkeypatch) -> None:
    monkeypatch.setattr(
        graph_module.StructuredLLMService,
        "generate_structured_sync",
        lambda self, prompt, response_model, system=None: ReflectionResult(
            needs_revision=False,
            issues=[],
            revised_outputs=None,
            confidence=0.8,
        ),
    )
    state = reflection_node(_state(0.2))
    assert state["worker_outputs"][0]["output"] == "ok"
    assert state["metadata"]["reflection_done"] is True


def test_full_graph_run_with_reflection_enabled(monkeypatch) -> None:
    monkeypatch.setattr(settings, "reflection_enabled", True)

    def fake_structured(self, prompt, response_model, system=None):
        if response_model is Plan:
            return Plan(
                reasoning="test",
                steps=[
                    StepSpec(
                        sequence=1,
                        role="researcher",
                        description="research",
                        expected_output="notes",
                    )
                ],
            )
        return ReflectionResult(needs_revision=False, issues=[], revised_outputs=None, confidence=0.8)

    monkeypatch.setattr(graph_module.StructuredLLMService, "generate_structured_sync", fake_structured)
    monkeypatch.setattr(graph_module.LLMService, "generate", lambda self, *args, **kwargs: "ok")
    run_id = str(uuid4())
    result = graph_module.app.invoke(
        {"run_id": run_id, "user_query": "hello", "metadata": {}, "current_step": 0, "worker_outputs": [], "confidence": 0.1},
        config={"configurable": {"thread_id": run_id}},
    )
    assert result["metadata"]["reflection_done"] is True
