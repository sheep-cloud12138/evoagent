from __future__ import annotations

from collections.abc import Iterator

from sqlmodel import Session

from evoagent.core.database import engine
from evoagent.core.runtime import AgentRuntime

_runtime: AgentRuntime | None = None


def get_db_session() -> Iterator[Session]:
    with Session(engine) as session:
        yield session


def get_runtime() -> AgentRuntime:
    global _runtime
    if _runtime is None:
        _runtime = AgentRuntime()
    return _runtime
