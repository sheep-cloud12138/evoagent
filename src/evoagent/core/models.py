from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Difficulty(str, Enum):
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"


class TaskCapability(str, Enum):
    RETRIEVAL = "retrieval"
    CODING = "coding"
    FILESYSTEM = "filesystem"
    WEB = "web"
    INTEGRATION = "integration"
    SKILL = "skill"
    PLANNING = "planning"
    UTILITY = "utility"


class TaskRequest(BaseModel):
    task_id: str
    query: str
    context: dict[str, Any] = Field(default_factory=dict)


class TaskFeatures(BaseModel):
    estimated_steps: int = 1
    needs_external_tools: bool = False
    has_historical_pattern: bool = False
    historical_pattern_score: float = 0.0
    confidence: float = 0.7
    required_capabilities: list[TaskCapability] = Field(default_factory=list)


class TaskDecision(BaseModel):
    difficulty: Difficulty
    score: float
    reasoning: str
    features: TaskFeatures
    confidence: float = 0.7
    fallback_plan: str = "若当前路径失败，升级难度并重试。"
    escalated_due_to_low_confidence: bool = False
    original_difficulty: Difficulty | None = None


class AgentOutput(BaseModel):
    agent_name: str
    objective: str
    result: str
    confidence: float = 0.7
    metadata: dict[str, Any] = Field(default_factory=dict)


class RunStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


class RunToolCall(BaseModel):
    name: str
    source: str = "unknown"
    category: str = "unknown"
    success: bool = False
    error: str = ""
    result_preview: str = ""
    step_id: str | None = None


class RunArtifact(BaseModel):
    artifact_id: str
    type: str
    name: str
    uri: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RunMemoryRef(BaseModel):
    memory_id: str
    kind: str = "semantic"
    score: float | None = None
    source: str | None = None
    preview: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class RunStep(BaseModel):
    step_id: str
    name: str
    role: str
    status: StepStatus = StepStatus.SUCCEEDED
    input: dict[str, Any] = Field(default_factory=dict)
    output: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    duration_ms: float = 0.0


class AgentRun(BaseModel):
    run_id: str
    task_id: str
    user_query: str
    status: RunStatus
    plan: list[str] = Field(default_factory=list)
    steps: list[RunStep] = Field(default_factory=list)
    tool_calls: list[RunToolCall] = Field(default_factory=list)
    artifacts: list[RunArtifact] = Field(default_factory=list)
    memory_refs: list[RunMemoryRef] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    started_at: str | None = None
    ended_at: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SkillManifest(BaseModel):
    name: str
    version: str
    description: str
    tags: list[str] = Field(default_factory=list)
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)
    permissions: list[str] = Field(default_factory=list)
    applicability: str = ""
    status: str = "candidate"


class ExecutionResult(BaseModel):
    task_id: str
    decision: TaskDecision
    outputs: list[AgentOutput]
    final_answer: str
    quality_score: float
    latency_seconds: float
    metadata: dict[str, Any] = Field(default_factory=dict)
    run: AgentRun | None = None
