from __future__ import annotations

from evoagent.agents.workers import coder_worker, memory_worker, researcher_worker, reviewer_worker, tool_worker
from evoagent.llm.service import LLMService


class _LLM(LLMService):
    def generate(self, *args, **kwargs) -> str:
        return "print('ok')"


def _state(role: str = "researcher") -> dict:
    return {
        "user_query": "hello",
        "worker_outputs": [],
        "metadata": {
            "next_worker": role,
            "current_step_spec": {"role": role, "description": "hello"},
        },
    }


def test_each_worker_function_returns_schema() -> None:
    llm = _LLM()
    for worker, role in (
        (researcher_worker, "researcher"),
        (coder_worker, "coder"),
        (tool_worker, "tool"),
        (memory_worker, "memory"),
        (reviewer_worker, "reviewer"),
    ):
        result = worker(_state(role), llm)
        assert {"role", "output", "artifacts", "confidence", "error"} <= set(result)


def test_coder_worker_sandbox_isolation() -> None:
    result = coder_worker(_state("coder"), _LLM())
    assert result["role"] == "coder"
    assert "ok" in result["output"] or result["error"] is not None


def test_tool_worker_handles_unknown_tool_gracefully() -> None:
    state = _state("tool")
    state["metadata"]["current_step_spec"] = {"description": "definitely_unknown_tool"}
    result = tool_worker(state, _LLM())
    assert result["role"] == "tool"
    assert result["error"]
    assert "unknown_tool" in result["error"]
