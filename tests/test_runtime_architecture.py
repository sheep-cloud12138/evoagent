import asyncio
import json

import pytest

from evoagent.core.gateway import AgentGateway
from evoagent.core.models import (
    AgentOutput,
    Difficulty,
    ExecutionResult,
    RunStatus,
    TaskDecision,
    TaskFeatures,
)
from evoagent.core.runtime import RunRuntime
from evoagent.core.workspace import WorkspaceSandbox
from evoagent.skills.runtime import SkillArtifactManager


def _decision() -> TaskDecision:
    return TaskDecision(
        difficulty=Difficulty.MEDIUM,
        score=0.5,
        reasoning="test",
        features=TaskFeatures(estimated_steps=3, needs_external_tools=True),
    )


def test_run_runtime_normalizes_trace_outputs_tools_memory_and_artifacts() -> None:
    result = ExecutionResult(
        task_id="task-1",
        decision=_decision(),
        outputs=[
            AgentOutput(
                agent_name="search-agent",
                objective="collect evidence",
                result="evidence",
                confidence=0.8,
                metadata={
                    "tool_events": [
                        {
                            "name": "fetch_url_text",
                            "source": "builtin",
                            "category": "web",
                            "success": True,
                            "result": "page text",
                        }
                    ]
                },
            )
        ],
        final_answer="answer",
        quality_score=0.86,
        latency_seconds=0.2,
        metadata={
            "trace": [{"step": "routing", "elapsed_ms": 1.5}],
            "semantic_hints": [
                {
                    "id": "mem-1",
                    "fact": "prior fact",
                    "score": 0.91,
                    "metadata": {"source": "semantic_store"},
                }
            ],
            "file_output": {"path": "/tmp/out.md", "filename": "out.md", "bytes": 6},
        },
    )

    finalized = RunRuntime().finalize_result("test query", {"session_id": "s1"}, result)

    assert finalized.run is not None
    assert finalized.run.status == RunStatus.SUCCEEDED
    assert finalized.run.user_query == "test query"
    assert finalized.run.steps[0].role == "planner"
    assert finalized.run.tool_calls[0].name == "fetch_url_text"
    assert finalized.run.memory_refs[0].memory_id == "mem-1"
    assert any(item.type == "file" for item in finalized.run.artifacts)
    assert finalized.metadata["run"]["run_id"] == finalized.task_id


def test_agent_gateway_delegates_to_runtime_callable() -> None:
    async def fake_run(query, context=None):
        return ExecutionResult(
            task_id="gateway-task",
            decision=_decision(),
            outputs=[],
            final_answer=f"{query}:{context['session_id']}",
            quality_score=0.9,
            latency_seconds=0.01,
        )

    result = asyncio.run(AgentGateway(fake_run).run("hello", {"session_id": "s2"}))

    assert result.final_answer == "hello:s2"


def test_workspace_sandbox_keeps_paths_inside_run_workspace(tmp_path) -> None:
    sandbox = WorkspaceSandbox(tmp_path)

    target = sandbox.resolve("run-1", "nested/file.txt")
    artifact = sandbox.artifact("run-1")

    assert target.parent.name == "nested"
    assert artifact.type == "workspace"
    with pytest.raises(ValueError):
        sandbox.resolve("run-1", "../escape.txt")


def test_skill_artifacts_persist_manifest(tmp_path) -> None:
    class Draft:
        name = "demo skill"
        description = "Demo skill"
        tags = ["demo"]
        code = "def execute(input_text: str) -> str:\n    return input_text\n"
        tests = "from skill_impl import execute\n\ndef test_x():\n    assert execute('a') == 'a'\n"

    manager = SkillArtifactManager(tmp_path)
    manager.persist(Draft(), version="1.0.0", status="candidate")

    items = json.loads((tmp_path / "index.json").read_text(encoding="utf-8"))

    assert items[0]["manifest"]["name"] == "demo skill"
    assert items[0]["manifest"]["version"] == "1.0.0"
    assert items[0]["manifest"]["input_schema"]["required"] == ["input_text"]
    assert items[0]["manifest"]["permissions"] == ["pure_python"]
