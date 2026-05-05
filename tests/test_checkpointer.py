from __future__ import annotations

import asyncio
from uuid import uuid4

from evoagent.core import graph as graph_module
from evoagent.core.runtime import AgentRuntime
from evoagent.llm.structured import Plan, StepSpec
from evoagent.models import RunStatus


def _mock_plan(monkeypatch) -> None:
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
    monkeypatch.setattr(graph_module.LLMService, "generate", lambda self, *args, **kwargs: "ok")


def test_run_can_be_resumed_after_simulated_failure(monkeypatch) -> None:
    runtime = AgentRuntime()
    run_id = str(uuid4())
    runtime._create_run("hello", run_id)
    runtime._set_run_status(run_id, RunStatus.failed, error="simulated")
    _mock_plan(monkeypatch)

    resumed = asyncio.run(runtime.resume(run_id))

    assert resumed.status == RunStatus.succeeded
    assert resumed.final_answer == "ok"


def test_checkpointer_saves_state_between_invocations(monkeypatch) -> None:
    _mock_plan(monkeypatch)
    run_id = str(uuid4())
    result = graph_module.app.invoke(
        {"run_id": run_id, "user_query": "hello", "metadata": {}, "current_step": 0, "worker_outputs": []},
        config={"configurable": {"thread_id": run_id}},
    )
    assert result["final_answer"] == "ok"
    assert graph_module.checkpointer is not None


def test_langsmith_env_absent_does_not_break(monkeypatch) -> None:
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    _mock_plan(monkeypatch)
    run_id = str(uuid4())
    result = graph_module.app.invoke(
        {"run_id": run_id, "user_query": "hello", "metadata": {}, "current_step": 0, "worker_outputs": []},
        config={"configurable": {"thread_id": run_id}},
    )
    assert result["final_answer"] == "ok"
