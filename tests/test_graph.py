from __future__ import annotations

from uuid import uuid4

from evoagent.core import graph as graph_module
from evoagent.core.graph import app, planner_node, should_continue
from evoagent.llm.structured import Plan, StepSpec


def test_graph_compiles_without_error() -> None:
    assert app is not None


def test_planner_node_returns_valid_plan_structure(monkeypatch) -> None:
    def fake_structured(self, prompt, response_model, system=None):
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

    monkeypatch.setattr(graph_module.StructuredLLMService, "generate_structured_sync", fake_structured)
    state = planner_node({"run_id": "r1", "user_query": "hello"})
    assert state["plan"]["steps"][0]["role"] == "researcher"
    assert state["steps"][0]["status"] == "pending"


def test_should_continue_routing_logic(monkeypatch) -> None:
    monkeypatch.setattr(graph_module.settings, "reflection_enabled", False)
    state = {
        "current_step": 0,
        "plan": {"steps": [{"sequence": 1, "role": "researcher"}]},
        "steps": [],
        "worker_outputs": [],
        "confidence": 0.9,
        "metadata": {},
    }
    assert should_continue(state) == "supervisor"
    state["current_step"] = 1
    assert should_continue(state) == "reporter"


def test_full_graph_run_with_mocked_llm(monkeypatch) -> None:
    def fake_structured(self, prompt, response_model, system=None):
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

    monkeypatch.setattr(graph_module.StructuredLLMService, "generate_structured_sync", fake_structured)
    monkeypatch.setattr(graph_module.LLMService, "generate", lambda self, *args, **kwargs: "worker answer")
    run_id = str(uuid4())
    result = app.invoke(
        {"run_id": run_id, "user_query": "hello", "metadata": {}, "current_step": 0, "worker_outputs": []},
        config={"configurable": {"thread_id": run_id}},
    )
    assert result["final_answer"] == "worker answer"
    assert result["current_step"] == 1
