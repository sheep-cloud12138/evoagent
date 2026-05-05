from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import inspect
from sqlmodel import Session, create_engine, select

from evoagent.core.database import create_all
from evoagent.models import (
    Artifact,
    ArtifactType,
    Run,
    RunStatus,
    Step,
    StepStatus,
    ToolCall,
)


def _engine():
    engine = create_engine("sqlite:///:memory:")
    create_all(engine)
    return engine


def test_create_run_with_json_fields() -> None:
    engine = _engine()

    with Session(engine) as session:
        run = Run(
            user_query="build a test plan",
            plan={"steps": [{"role": "planner"}]},
            metadata={"source": "unit-test"},
        )
        session.add(run)
        session.commit()
        session.refresh(run)

        stored = session.get(Run, run.run_id)

    assert stored is not None
    assert stored.run_id
    assert stored.status == RunStatus.queued
    assert stored.plan == {"steps": [{"role": "planner"}]}
    assert stored.metadata == {"source": "unit-test"}
    assert repr(stored) == f"Run(run_id={stored.run_id!r}, status='queued')"


def test_update_run_and_step_status() -> None:
    engine = _engine()

    with Session(engine) as session:
        run = Run(user_query="ship it")
        step = Step(run_id=run.run_id, role="coder", sequence=1, input={"task": "write code"})
        session.add(run)
        session.add(step)
        session.commit()

        run.status = RunStatus.running
        run.started_at = datetime.now(tz=timezone.utc)
        step.status = StepStatus.running
        session.add(run)
        session.add(step)
        session.commit()

        run.status = RunStatus.succeeded
        run.final_answer = "done"
        run.ended_at = datetime.now(tz=timezone.utc)
        step.status = StepStatus.succeeded
        step.output = {"result": "ok"}
        step.duration_ms = 12
        session.add(run)
        session.add(step)
        session.commit()

        stored_run = session.get(Run, run.run_id)
        stored_step = session.get(Step, step.step_id)

    assert stored_run is not None
    assert stored_step is not None
    assert stored_run.status == RunStatus.succeeded
    assert stored_run.final_answer == "done"
    assert stored_run.started_at is not None
    assert stored_run.ended_at is not None
    assert stored_step.status == StepStatus.succeeded
    assert stored_step.output == {"result": "ok"}
    assert stored_step.duration_ms == 12


def test_step_artifact_and_tool_call_foreign_keys() -> None:
    engine = _engine()

    with Session(engine) as session:
        run = Run(user_query="use tools")
        step = Step(
            run_id=run.run_id,
            role="tool",
            sequence=1,
            input={"tool": "search"},
        )
        artifact = Artifact(
            run_id=run.run_id,
            step_id=step.step_id,
            type=ArtifactType.tool_result,
            content="search result",
            metadata={"provider": "mock"},
        )
        tool_call = ToolCall(
            run_id=run.run_id,
            step_id=step.step_id,
            tool_name="search",
            input={"query": "evoagent"},
            output={"items": []},
        )
        session.add(run)
        session.add(step)
        session.add(artifact)
        session.add(tool_call)
        session.commit()

        steps = list(session.exec(select(Step).where(Step.run_id == run.run_id)))
        artifacts = list(session.exec(select(Artifact).where(Artifact.run_id == run.run_id)))
        calls = list(session.exec(select(ToolCall).where(ToolCall.step_id == step.step_id)))

    assert len(steps) == 1
    assert steps[0].run_id == run.run_id
    assert steps[0].status == StepStatus.pending
    assert len(artifacts) == 1
    assert artifacts[0].step_id == step.step_id
    assert artifacts[0].metadata == {"provider": "mock"}
    assert len(calls) == 1
    assert calls[0].run_id == run.run_id
    assert calls[0].output == {"items": []}
    assert calls[0].status == "pending"

    foreign_keys = {
        table: inspect(engine).get_foreign_keys(table)
        for table in ("step", "artifact", "toolcall")
    }
    assert any(fk["referred_table"] == "run" for fk in foreign_keys["step"])
    assert any(fk["referred_table"] == "run" for fk in foreign_keys["artifact"])
    assert any(fk["referred_table"] == "step" for fk in foreign_keys["artifact"])
    assert any(fk["referred_table"] == "run" for fk in foreign_keys["toolcall"])
    assert any(fk["referred_table"] == "step" for fk in foreign_keys["toolcall"])
