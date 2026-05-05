from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from evoagent.core.models import ExecutionResult


RunCallable = Callable[[str, dict[str, Any] | None], Awaitable[ExecutionResult]]


class AgentGateway:
    """In-process gateway used by CLI/API frontends to submit agent runs."""

    def __init__(self, run_callable: RunCallable) -> None:
        self._run_callable = run_callable

    async def run(
        self, query: str, context: dict[str, Any] | None = None
    ) -> ExecutionResult:
        return await self._run_callable(query, context)

    def run_sync(
        self, query: str, context: dict[str, Any] | None = None
    ) -> ExecutionResult:
        return asyncio.run(self.run(query, context=context))
