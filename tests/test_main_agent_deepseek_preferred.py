from evoagent.core.llm import LLMClient
from evoagent.core.models import AgentOutput, TaskRequest
from evoagent.core.orchestrator import SubAgentOrchestrator


def test_aggregate_prefers_main_agent_model(monkeypatch) -> None:
    orchestrator = SubAgentOrchestrator(LLMClient())
    task = TaskRequest(task_id="t1", query="test")
    outputs = [AgentOutput(agent_name="x", objective="o", result="r")]

    captured = {}

    def fake_generate(prompt, temperature=None, profile=LLMClient.Profile.STANDARD, force_models=None):
        captured["force_models"] = force_models
        return "ok"

    monkeypatch.setattr(orchestrator.llm, "generate", fake_generate)
    answer = orchestrator.aggregate(task, outputs)

    assert answer == "ok"
    assert captured["force_models"] is not None
    assert isinstance(captured["force_models"], list)
    assert captured["force_models"][0].startswith("deepseek/")
