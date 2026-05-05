from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class StepSpec(BaseModel):
    sequence: int
    role: Literal["researcher", "coder", "tool", "memory", "reviewer", "reporter"]
    description: str
    depends_on: list[int] = Field(default_factory=list)
    expected_output: str = ""


class Plan(BaseModel):
    steps: list[StepSpec]
    reasoning: str = ""


class ReflectionResult(BaseModel):
    needs_revision: bool
    issues: list[str] = Field(default_factory=list)
    revised_outputs: list[dict] | None = None
    confidence: float = 0.7


class WorkerOutput(BaseModel):
    role: str
    output: str
    confidence: float
    artifacts: list[str] = Field(default_factory=list)
    error: str | None = None
