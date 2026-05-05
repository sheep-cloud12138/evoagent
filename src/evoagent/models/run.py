from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import ConfigDict
from sqlalchemy import Column, JSON
from sqlmodel import Field, SQLModel


def _uuid() -> str:
    return str(uuid4())


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


class RunStatus(str, Enum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    cancelled = "cancelled"


class StepStatus(str, Enum):
    pending = "pending"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    skipped = "skipped"


class ArtifactType(str, Enum):
    text = "text"
    code = "code"
    file = "file"
    tool_result = "tool_result"
    memory_ref = "memory_ref"


class Run(SQLModel, table=True):
    model_config = ConfigDict(populate_by_name=True)

    run_id: str = Field(default_factory=_uuid, primary_key=True)
    user_query: str
    status: RunStatus = Field(default=RunStatus.queued)
    plan: dict[str, Any] | None = Field(
        default=None, sa_column=Column(JSON, nullable=True)
    )
    final_answer: str | None = None
    error: str | None = None
    started_at: datetime | None = None
    ended_at: datetime | None = None
    metadata_: dict[str, Any] = Field(
        default_factory=dict,
        alias="metadata",
        sa_column=Column("metadata", JSON, nullable=False),
    )

    def __repr__(self) -> str:
        return f"Run(run_id={self.run_id!r}, status={self.status.value!r})"


class Step(SQLModel, table=True):
    step_id: str = Field(default_factory=_uuid, primary_key=True)
    run_id: str = Field(foreign_key="run.run_id", index=True)
    role: str
    sequence: int
    input: dict[str, Any] = Field(
        default_factory=dict, sa_column=Column(JSON, nullable=False)
    )
    output: dict[str, Any] | None = Field(
        default=None, sa_column=Column(JSON, nullable=True)
    )
    status: StepStatus = Field(default=StepStatus.pending)
    error: str | None = None
    duration_ms: int | None = None
    created_at: datetime = Field(default_factory=_now)

    def __repr__(self) -> str:
        return f"Step(step_id={self.step_id!r}, run_id={self.run_id!r}, status={self.status.value!r})"


class Artifact(SQLModel, table=True):
    model_config = ConfigDict(populate_by_name=True)

    artifact_id: str = Field(default_factory=_uuid, primary_key=True)
    run_id: str = Field(foreign_key="run.run_id", index=True)
    step_id: str | None = Field(default=None, foreign_key="step.step_id", index=True)
    type: ArtifactType
    content: str
    metadata_: dict[str, Any] = Field(
        default_factory=dict,
        alias="metadata",
        sa_column=Column("metadata", JSON, nullable=False),
    )
    created_at: datetime = Field(default_factory=_now)

    def __repr__(self) -> str:
        return f"Artifact(artifact_id={self.artifact_id!r}, type={self.type.value!r})"


class ToolCall(SQLModel, table=True):
    call_id: str = Field(default_factory=_uuid, primary_key=True)
    step_id: str = Field(foreign_key="step.step_id", index=True)
    run_id: str = Field(foreign_key="run.run_id", index=True)
    tool_name: str
    input: dict[str, Any] = Field(
        default_factory=dict, sa_column=Column(JSON, nullable=False)
    )
    output: dict[str, Any] | None = Field(
        default=None, sa_column=Column(JSON, nullable=True)
    )
    status: str = Field(default="pending")
    error: str | None = None
    duration_ms: int | None = None
    created_at: datetime = Field(default_factory=_now)

    def __repr__(self) -> str:
        return f"ToolCall(call_id={self.call_id!r}, tool_name={self.tool_name!r}, status={self.status!r})"


def _install_metadata_property(model: type[SQLModel]) -> None:
    def getter(self: SQLModel) -> dict[str, Any]:
        return self.metadata_

    def setter(self: SQLModel, value: dict[str, Any]) -> None:
        self.metadata_ = value

    setattr(model, "metadata", property(getter, setter))


_install_metadata_property(Run)
_install_metadata_property(Artifact)
