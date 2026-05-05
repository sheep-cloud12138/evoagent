from __future__ import annotations

from evoagent.core.models import AgentOutput, TaskDecision, TaskRequest
from evoagent.core.orchestrator import SubAgentOrchestrator


class AgentSupervisor:
    """Supervisor boundary around planning, worker dispatch, review, and reporting."""

    def __init__(self, orchestrator: SubAgentOrchestrator) -> None:
        self.orchestrator = orchestrator

    async def execute(
        self, task: TaskRequest, decision: TaskDecision
    ) -> list[AgentOutput]:
        return await self.orchestrator.execute(task, decision)

    def aggregate(self, task: TaskRequest, outputs: list[AgentOutput]) -> str:
        return self.orchestrator.aggregate(task, outputs)

    def deterministic_aggregate(
        self, task: TaskRequest, outputs: list[AgentOutput]
    ) -> str:
        return self.orchestrator._deterministic_aggregate(task, outputs)

    def maybe_reflect(
        self,
        task: TaskRequest,
        decision: TaskDecision,
        outputs: list[AgentOutput],
        draft_answer: str,
    ) -> tuple[str, bool, str]:
        return self.orchestrator.maybe_reflect(task, decision, outputs, draft_answer)
