from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from pydantic import ConfigDict, field_validator
from sqlalchemy import Column, JSON
from sqlmodel import Field, SQLModel


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


class SkillManifest(SQLModel, table=True):
    model_config = ConfigDict(populate_by_name=True)

    skill_id: str = Field(primary_key=True)
    name: str
    version: str
    description: str
    input_schema: dict[str, Any] = Field(
        default_factory=dict, sa_column=Column(JSON, nullable=False)
    )
    output_schema: dict[str, Any] = Field(
        default_factory=dict, sa_column=Column(JSON, nullable=False)
    )
    permissions: list[str] = Field(
        default_factory=list, sa_column=Column(JSON, nullable=False)
    )
    tags: list[str] = Field(
        default_factory=list, sa_column=Column(JSON, nullable=False)
    )
    status: str = "candidate"
    success_count: int = 0
    failure_count: int = 0
    success_rate: float = 0.0
    min_successes_to_activate: int = 3
    created_at: datetime = Field(default_factory=_now)
    last_used_at: datetime | None = None
    source_code: str
    test_code: str
    author: str = "evolution"

    @field_validator("version")
    @classmethod
    def _validate_semver(cls, value: str) -> str:
        if not re.fullmatch(r"\d+\.\d+\.\d+", value):
            raise ValueError("version must be semver, e.g. 1.2.3")
        return value

    @field_validator("status")
    @classmethod
    def _validate_status(cls, value: str) -> str:
        allowed = {"candidate", "testing", "active", "deprecated", "rejected"}
        if value not in allowed:
            raise ValueError(f"status must be one of {sorted(allowed)}")
        return value

    def __repr__(self) -> str:
        return f"SkillManifest(skill_id={self.skill_id!r}, status={self.status!r})"
