import asyncio

from evoagent.agents.base import BaseSubAgent
from evoagent.core.llm import LLMClient
from evoagent.core.models import TaskRequest


class _FallbackLLM(LLMClient):
    def generate(
        self,
        prompt: str,
        temperature: float | None = None,
        profile: str = LLMClient.Profile.STANDARD,
        force_models: list[str] | None = None,
    ) -> str:
        _ = prompt, temperature, profile, force_models
        return "[Fallback-LLM] timeout"


def test_fallback_output_is_marked_failed() -> None:
    agent = BaseSubAgent(_FallbackLLM())
    task = TaskRequest(task_id="t", query="hello", context={})

    output = asyncio.run(agent.run(task, "answer"))

    assert output.metadata["failed"] is True
    assert output.confidence == 0.2
