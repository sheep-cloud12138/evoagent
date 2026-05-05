from __future__ import annotations

from evoagent.core.gateway import AgentGateway
from evoagent.core.models import ExecutionResult
from evoagent.core.runtime import AgentRuntime
from evoagent.core.services import EvoAgentServices
from evoagent.core.weather import fetch_weather_report


class EvoAgentSystem(EvoAgentServices):
    """Public in-process system facade.

    All task execution now enters the LangGraph runtime. The inherited service
    bundle keeps the original router, orchestrator, memory, feedback, and skill
    components available to graph nodes and tests.
    """

    def __init__(self) -> None:
        super().__init__()
        self.langgraph_runtime = AgentRuntime()
        self.gateway = AgentGateway(self.run)

    def fetch_weather_report(self, location_text: str) -> str:
        # Keep tests and external callers able to monkeypatch evoagent.app.fetch_weather_report.
        return fetch_weather_report(location_text)

    async def run(self, query: str, context: dict | None = None) -> ExecutionResult:
        return await self.langgraph_runtime.run_result(
            query,
            context=context,
            services=self,
        )
