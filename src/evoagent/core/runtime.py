from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from sqlmodel import Session, select

from evoagent.core.database import engine, init_db
from evoagent.models import (
    Artifact,
    ArtifactType,
    Run,
    RunStatus as DbRunStatus,
    Step,
    StepStatus as DbStepStatus,
)
from evoagent.core.models import (
    AgentRun,
    ExecutionResult,
    RunArtifact,
    RunMemoryRef,
    RunStatus,
    RunStep,
    RunToolCall,
    StepStatus,
)


class RunRuntime:
    """Normalizes existing execution output into a durable run record."""

    @staticmethod
    def _now() -> str:
        return datetime.now(tz=timezone.utc).isoformat()

    @staticmethod
    def _role_for_step(name: str) -> str:
        lowered = name.lower()
        if "route" in lowered or "routing" in lowered:
            return "planner"
        if "sub_agent" in lowered or "agent" in lowered:
            return "worker"
        if "aggregate" in lowered or "reflection" in lowered:
            return "reporter"
        if "semantic" in lowered or "memory" in lowered:
            return "memory"
        if "skill" in lowered:
            return "skill"
        if "file" in lowered:
            return "artifact"
        return "runtime"

    @staticmethod
    def _status_from_error(error: str | None) -> StepStatus:
        return StepStatus.FAILED if error else StepStatus.SUCCEEDED

    def _trace_steps(self, trace: list[Any]) -> list[RunStep]:
        steps: list[RunStep] = []
        for idx, raw in enumerate(trace):
            if not isinstance(raw, dict):
                continue
            name = str(raw.get("step", f"step_{idx + 1}"))
            error = str(raw.get("error", "")).strip() or None
            try:
                duration_ms = float(raw.get("elapsed_ms", 0.0))
            except Exception:
                duration_ms = 0.0
            output = {
                key: value
                for key, value in raw.items()
                if key not in {"step", "elapsed_ms", "error"}
            }
            steps.append(
                RunStep(
                    step_id=f"trace-{idx + 1}",
                    name=name,
                    role=self._role_for_step(name),
                    status=self._status_from_error(error),
                    output=output,
                    error=error,
                    duration_ms=duration_ms,
                )
            )
        return steps

    @staticmethod
    def _agent_steps(result: ExecutionResult) -> list[RunStep]:
        steps: list[RunStep] = []
        for idx, output in enumerate(result.outputs):
            failed = bool(output.metadata.get("failed", False))
            timed_out = bool(output.metadata.get("timed_out", False))
            errors = []
            if failed:
                errors.append("failed")
            if timed_out:
                errors.append("timed_out")
            steps.append(
                RunStep(
                    step_id=f"agent-{idx + 1}",
                    name=output.agent_name,
                    role="worker",
                    status=StepStatus.FAILED if failed else StepStatus.SUCCEEDED,
                    input={"objective": output.objective},
                    output={
                        "confidence": output.confidence,
                        "preview": output.result[:800],
                    },
                    error=",".join(errors) if errors else None,
                )
            )
        return steps

    @staticmethod
    def _tool_calls(result: ExecutionResult) -> list[RunToolCall]:
        calls: list[RunToolCall] = []
        for output_idx, output in enumerate(result.outputs):
            events = output.metadata.get("tool_events", [])
            if not isinstance(events, list):
                continue
            for event in events:
                if not isinstance(event, dict):
                    continue
                calls.append(
                    RunToolCall(
                        name=str(event.get("name", "")),
                        source=str(event.get("source", "unknown")),
                        category=str(event.get("category", "unknown")),
                        success=bool(event.get("success", False)),
                        error=str(event.get("error", ""))[:240],
                        result_preview=str(event.get("result", ""))[:300],
                        step_id=f"agent-{output_idx + 1}",
                    )
                )
        return calls

    @staticmethod
    def _artifacts(metadata: dict[str, Any], final_answer: str) -> list[RunArtifact]:
        artifacts = [
            RunArtifact(
                artifact_id="final-answer",
                type="answer",
                name="final_answer",
                metadata={"chars": len(final_answer)},
            )
        ]
        file_output = metadata.get("file_output")
        if isinstance(file_output, dict) and file_output.get("path"):
            artifacts.append(
                RunArtifact(
                    artifact_id="file-output",
                    type="file",
                    name=str(file_output.get("filename") or "output"),
                    uri=str(file_output.get("path")),
                    metadata={k: v for k, v in file_output.items() if k != "path"},
                )
            )
        matched_skill = metadata.get("matched_skill")
        if isinstance(matched_skill, dict):
            artifacts.append(
                RunArtifact(
                    artifact_id="matched-skill",
                    type="skill",
                    name=str(matched_skill.get("name", "skill")),
                    uri=str(matched_skill.get("skill_path", "")) or None,
                    metadata=matched_skill,
                )
            )
        return artifacts

    @staticmethod
    def _memory_refs(metadata: dict[str, Any]) -> list[RunMemoryRef]:
        refs: list[RunMemoryRef] = []
        hints = metadata.get("semantic_hints", [])
        if not isinstance(hints, list):
            return refs
        for idx, item in enumerate(hints):
            if not isinstance(item, dict):
                continue
            raw_meta = item.get("metadata", {})
            meta = raw_meta if isinstance(raw_meta, dict) else {}
            score = item.get("score")
            try:
                score_value = float(score) if score is not None else None
            except Exception:
                score_value = None
            refs.append(
                RunMemoryRef(
                    memory_id=str(item.get("id", f"semantic-{idx + 1}")),
                    kind=str(meta.get("kind", "semantic")),
                    score=score_value,
                    source=str(meta.get("source", "")) or None,
                    preview=str(item.get("fact", item.get("document", "")))[:300],
                    metadata=meta,
                )
            )
        return refs

    @staticmethod
    def _errors(
        result: ExecutionResult, steps: list[RunStep], tool_calls: list[RunToolCall]
    ) -> list[str]:
        errors: list[str] = []
        if result.final_answer.startswith("[Fallback-LLM]"):
            errors.append("fallback_llm")
        for step in steps:
            if step.error:
                errors.append(f"{step.name}:{step.error}")
        for call in tool_calls:
            if not call.success and call.error:
                errors.append(f"tool:{call.name}:{call.error}")
        return errors

    def finalize_result(
        self,
        query: str,
        context: dict[str, Any],
        result: ExecutionResult,
    ) -> ExecutionResult:
        metadata = dict(result.metadata or {})
        trace = metadata.get("trace", [])
        trace_steps = self._trace_steps(trace if isinstance(trace, list) else [])
        agent_steps = self._agent_steps(result)
        steps = trace_steps + agent_steps
        tool_calls = self._tool_calls(result)
        errors = self._errors(result, steps, tool_calls)
        status = (
            RunStatus.FAILED
            if errors and result.quality_score < 0.72
            else RunStatus.SUCCEEDED
        )
        run = AgentRun(
            run_id=str(metadata.get("run_id") or result.task_id or uuid4()),
            task_id=result.task_id,
            user_query=query,
            status=status,
            plan=[str(output.objective) for output in result.outputs],
            steps=steps,
            tool_calls=tool_calls,
            artifacts=self._artifacts(metadata, result.final_answer),
            memory_refs=self._memory_refs(metadata),
            errors=errors,
            started_at=None,
            ended_at=self._now(),
            metadata={
                "session_id": context.get("session_id"),
                "mode": metadata.get("mode", "agent_runtime"),
                "difficulty": result.decision.difficulty.value,
                "quality_score": result.quality_score,
            },
        )
        metadata["run_id"] = run.run_id
        metadata["run"] = run.model_dump(mode="json")
        result.metadata = metadata
        result.run = run
        return result


class AgentRuntime:
    """LangGraph runtime that persists runs, steps, and artifacts."""

    def __init__(self) -> None:
        init_db()

    @staticmethod
    def _now() -> datetime:
        return datetime.now(tz=timezone.utc)

    @staticmethod
    def _graph_config(run_id: str) -> dict[str, Any]:
        return {
            "configurable": {"thread_id": run_id},
            "run_name": f"evoagent-run-{run_id}",
            "tags": ["evoagent", run_id],
        }

    def _get_run(self, run_id: str) -> Run | None:
        with Session(engine) as session:
            return session.get(Run, run_id)

    def _create_run(self, user_query: str, run_id: str) -> Run:
        with Session(engine) as session:
            existing = session.get(Run, run_id)
            if existing is not None:
                return existing
            run = Run(
                run_id=run_id,
                user_query=user_query,
                status=DbRunStatus.running,
                started_at=self._now(),
                metadata={},
            )
            session.add(run)
            session.commit()
            session.refresh(run)
            return run

    def _set_run_status(
        self,
        run_id: str,
        status: DbRunStatus,
        *,
        final_answer: str | None = None,
        error: str | None = None,
        plan: dict | None = None,
        metadata: dict | None = None,
    ) -> Run:
        with Session(engine) as session:
            run = session.get(Run, run_id)
            if run is None:
                raise RuntimeError(f"Run not found: {run_id}")
            run.status = status
            run.error = error
            if final_answer is not None:
                run.final_answer = final_answer
            if plan is not None:
                run.plan = plan
            if metadata is not None:
                run.metadata_ = metadata
            if status in {
                DbRunStatus.succeeded,
                DbRunStatus.failed,
                DbRunStatus.cancelled,
            }:
                run.ended_at = self._now()
            session.add(run)
            session.commit()
            session.refresh(run)
            return run

    def _persist_outputs(self, run_id: str, state: dict[str, Any]) -> None:
        steps = state.get("steps", [])
        outputs = state.get("worker_outputs", [])
        if not isinstance(steps, list):
            steps = []
        if not isinstance(outputs, list):
            outputs = []

        with Session(engine) as session:
            existing_sequences = {
                int(step.sequence)
                for step in session.exec(select(Step).where(Step.run_id == run_id))
            }
            for idx, step_payload in enumerate(steps):
                if not isinstance(step_payload, dict):
                    continue
                sequence = int(step_payload.get("sequence", idx + 1))
                if sequence in existing_sequences:
                    continue
                output = step_payload.get("output")
                if not isinstance(output, dict):
                    output = (
                        outputs[idx]
                        if idx < len(outputs) and isinstance(outputs[idx], dict)
                        else {}
                    )
                failed = (
                    bool(output.get("error")) or step_payload.get("status") == "failed"
                )
                step = Step(
                    run_id=run_id,
                    role=str(step_payload.get("role", output.get("role", "worker"))),
                    sequence=sequence,
                    input=step_payload,
                    output=output,
                    status=DbStepStatus.failed if failed else DbStepStatus.succeeded,
                    error=str(output.get("error")) if output.get("error") else None,
                    duration_ms=int(float(step_payload.get("duration_ms") or 0)),
                )
                session.add(step)
                session.flush()
                artifacts = output.get("artifacts", [])
                if not isinstance(artifacts, list):
                    artifacts = [artifacts]
                if not artifacts and output.get("output"):
                    artifacts = [str(output.get("output"))]
                for artifact_payload in artifacts:
                    content = ""
                    metadata: dict[str, Any] = {}
                    artifact_type = ArtifactType.text
                    if isinstance(artifact_payload, dict):
                        content = str(artifact_payload.get("content", ""))
                        metadata = (
                            artifact_payload.get("metadata", {})
                            if isinstance(artifact_payload.get("metadata", {}), dict)
                            else {}
                        )
                        raw_type = str(artifact_payload.get("type", "text"))
                        artifact_type = (
                            ArtifactType(raw_type)
                            if raw_type in ArtifactType._value2member_map_
                            else ArtifactType.text
                        )
                    else:
                        content = str(artifact_payload)
                    session.add(
                        Artifact(
                            run_id=run_id,
                            step_id=step.step_id,
                            type=artifact_type,
                            content=content,
                            metadata=metadata,
                        )
                    )
            if state.get("final_answer"):
                session.add(
                    Artifact(
                        run_id=run_id,
                        step_id=None,
                        type=ArtifactType.text,
                        content=str(state.get("final_answer", "")),
                        metadata={"name": "final_answer"},
                    )
                )
            session.commit()

    async def _invoke_graph(
        self,
        user_query: str,
        run_id: str,
        *,
        context: dict[str, Any] | None = None,
        services: Any | None = None,
    ) -> dict[str, Any]:
        import asyncio

        from evoagent.core.graph import (
            activate_runtime_services,
            app,
            clear_runtime_services,
        )

        initial_state = {
            "run_id": run_id,
            "user_query": user_query,
            "context": dict(context or {}),
            "plan": None,
            "steps": [],
            "current_step": 0,
            "worker_outputs": [],
            "agent_outputs": [],
            "final_answer": None,
            "error": None,
            "confidence": 0.0,
            "metadata": {"run_id": run_id},
        }
        def _invoke_in_thread() -> dict[str, Any]:
            if services is not None:
                activate_runtime_services(services)
            try:
                return app.invoke(initial_state, self._graph_config(run_id))
            finally:
                if services is not None:
                    clear_runtime_services()

        return await asyncio.to_thread(_invoke_in_thread)

    def _result_from_state(
        self, user_query: str, context: dict[str, Any], state: dict[str, Any]
    ) -> ExecutionResult:
        from evoagent.core.graph import execution_result_from_state

        result = execution_result_from_state(state)
        return RunRuntime().finalize_result(user_query, context, result)

    async def run(
        self,
        user_query: str,
        run_id: str | None = None,
        *,
        context: dict[str, Any] | None = None,
        services: Any | None = None,
    ) -> Run:
        resolved_run_id = run_id or str(uuid4())
        self._create_run(user_query, resolved_run_id)
        try:
            state = await self._invoke_graph(
                user_query,
                resolved_run_id,
                context=context,
                services=services,
            )
            self._persist_outputs(resolved_run_id, state)
            return self._set_run_status(
                resolved_run_id,
                DbRunStatus.succeeded,
                final_answer=state.get("final_answer"),
                plan=state.get("plan"),
                metadata=state.get("metadata", {}),
            )
        except Exception as exc:
            return self._set_run_status(
                resolved_run_id, DbRunStatus.failed, error=str(exc)
            )

    async def run_result(
        self,
        user_query: str,
        *,
        context: dict[str, Any] | None = None,
        services: Any | None = None,
        run_id: str | None = None,
    ) -> ExecutionResult:
        resolved_run_id = run_id or str(uuid4())
        self._create_run(user_query, resolved_run_id)
        context_payload = dict(context or {})
        try:
            state = await self._invoke_graph(
                user_query,
                resolved_run_id,
                context=context_payload,
                services=services,
            )
            self._persist_outputs(resolved_run_id, state)
            self._set_run_status(
                resolved_run_id,
                DbRunStatus.succeeded,
                final_answer=state.get("final_answer"),
                plan=state.get("plan"),
                metadata=state.get("metadata", {}),
            )
            return self._result_from_state(user_query, context_payload, state)
        except Exception as exc:
            self._set_run_status(resolved_run_id, DbRunStatus.failed, error=str(exc))
            raise

    async def resume(self, run_id: str) -> Run:
        run = self._get_run(run_id)
        if run is None:
            raise RuntimeError(f"Run not found: {run_id}")
        if run.status not in {DbRunStatus.failed, DbRunStatus.running}:
            return run
        self._set_run_status(run_id, DbRunStatus.running, error=None)
        try:
            state = await self._invoke_graph(run.user_query, run_id)
            self._persist_outputs(run_id, state)
            return self._set_run_status(
                run_id,
                DbRunStatus.succeeded,
                final_answer=state.get("final_answer"),
                plan=state.get("plan"),
                metadata=state.get("metadata", {}),
            )
        except Exception as exc:
            return self._set_run_status(run_id, DbRunStatus.failed, error=str(exc))
