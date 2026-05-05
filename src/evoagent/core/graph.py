from __future__ import annotations

import asyncio
import operator
import os
import random
import threading
import time
import uuid
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, StateGraph

from evoagent.core.config import settings
from evoagent.core.models import (
    AgentOutput,
    Difficulty,
    ExecutionResult,
    TaskDecision,
    TaskFeatures,
    TaskRequest,
)
from evoagent.core.services import EvoAgentServices
from evoagent.core.tool_evidence import successful_tool_events_from_outputs
from evoagent.core.weather import extract_weather_location, is_weather_query
from evoagent.llm.service import LLMService
from evoagent.llm.structured import Plan, ReflectionResult, StepSpec
from evoagent.llm.structured_service import StructuredLLMService

try:
    from langgraph.checkpoint.sqlite import SqliteSaver
except Exception:  # pragma: no cover
    SqliteSaver = None

try:
    from langgraph.checkpoint.memory import MemorySaver
except Exception:  # pragma: no cover
    MemorySaver = None

if os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true" and os.getenv(
    "LANGCHAIN_API_KEY"
):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
elif not os.getenv("LANGCHAIN_API_KEY"):
    os.environ.pop("LANGCHAIN_TRACING_V2", None)


class AgentState(TypedDict, total=False):
    run_id: str
    task_id: str
    user_query: str
    context: dict
    plan: dict | None
    steps: list[dict]
    current_step: int
    worker_outputs: list[dict]
    agent_outputs: list[dict]
    decision: dict | None
    final_answer: str | None
    error: str | None
    confidence: float
    quality_score: float
    latency_seconds: float
    completed: bool
    finalized: bool
    start_time: float
    trace: list[dict]
    semantic_hints: list[dict]
    matched_skill: dict | None
    match_score: float
    pending_agent_specs: list[dict]
    current_agent_spec: dict
    agent_result_batches: Annotated[list[dict], operator.add]
    dispatch_started_at: float
    retry_count: int
    retry_reason: str | None
    retry_history: list[dict]
    retry_requested: bool
    post_actions: dict
    file_output: dict | None
    metadata: dict


_fallback_services: EvoAgentServices | None = None
_bound_services: dict[str, EvoAgentServices] = {}
_thread_services = threading.local()


def activate_runtime_services(services: EvoAgentServices) -> None:
    _thread_services.services = services


def clear_runtime_services() -> None:
    if hasattr(_thread_services, "services"):
        delattr(_thread_services, "services")


def bind_runtime_services(run_id: str, services: EvoAgentServices) -> None:
    _bound_services[str(run_id)] = services


def unbind_runtime_services(run_id: str) -> None:
    _bound_services.pop(str(run_id), None)


def _services_from_config(config: Any | None = None) -> EvoAgentServices:
    services = getattr(_thread_services, "services", None)
    if services is not None:
        return services

    configurable = {}
    if isinstance(config, dict):
        configurable = config.get("configurable", {}) or {}
    else:
        configurable = getattr(config, "configurable", {}) or {}
    services = configurable.get("services")
    if services is not None:
        return services
    thread_id = configurable.get("thread_id")
    if thread_id is not None and str(thread_id) in _bound_services:
        return _bound_services[str(thread_id)]

    global _fallback_services
    if _fallback_services is None:
        _fallback_services = EvoAgentServices()
    return _fallback_services


def _now_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000


def _append_trace(state: AgentState, step: str, elapsed_ms: float, **extra: Any) -> None:
    trace = list(state.get("trace", []))
    payload: dict[str, Any] = {"step": step, "elapsed_ms": elapsed_ms}
    payload.update(extra)
    trace.append(payload)
    state["trace"] = trace


def _append_graph_step(
    state: AgentState,
    *,
    role: str,
    description: str,
    status: str = "succeeded",
    output: dict[str, Any] | None = None,
    error: str | None = None,
    duration_ms: float = 0.0,
) -> None:
    steps = list(state.get("steps", []))
    sequence = len(steps) + 1
    steps.append(
        {
            "sequence": sequence,
            "role": role,
            "description": description,
            "status": status,
            "output": output or {},
            "error": error,
            "duration_ms": duration_ms,
        }
    )
    state["steps"] = steps


def _task_from_state(state: AgentState) -> TaskRequest:
    return TaskRequest(
        task_id=str(state.get("task_id") or state.get("run_id") or uuid.uuid4()),
        query=str(state.get("user_query", "")),
        context=dict(state.get("context", {})),
    )


def _decision_from_state(state: AgentState) -> TaskDecision:
    raw = state.get("decision")
    if isinstance(raw, dict):
        return TaskDecision.model_validate(raw)
    return TaskDecision(
        difficulty=Difficulty.MEDIUM,
        score=0.5,
        reasoning="missing_decision",
        features=TaskFeatures(),
        confidence=0.5,
    )


def _agent_outputs_from_state(state: AgentState) -> list[AgentOutput]:
    outputs: list[AgentOutput] = []
    for raw in state.get("agent_outputs", []):
        if isinstance(raw, dict):
            outputs.append(AgentOutput.model_validate(raw))
    return outputs


def _run_coro(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise RuntimeError("EvoAgent graph nodes must run in a worker thread or sync context")


def _agent_output_to_worker(output: AgentOutput) -> dict[str, Any]:
    return {
        "role": output.agent_name,
        "output": output.result,
        "artifacts": [],
        "confidence": output.confidence,
        "metadata": output.metadata,
    }


_AGENT_NODE_BY_NAME = {
    "search-agent": "search_agent",
    "reasoning-agent": "reasoning_agent",
    "code-agent": "code_agent",
    "file-agent": "file_agent",
    "integration-agent": "integration_agent",
    "skill-agent": "skill_agent",
    "tool-agent": "tool_agent",
}


def _agent_for_name(name: str, llm: LLMService):
    from evoagent.agents.specialists import (
        CodeAgent,
        FileAgent,
        IntegrationAgent,
        ReasoningAgent,
        SearchAgent,
        SkillAgent,
        ToolAgent,
    )

    agent_map = {
        "search-agent": SearchAgent,
        "reasoning-agent": ReasoningAgent,
        "code-agent": CodeAgent,
        "file-agent": FileAgent,
        "integration-agent": IntegrationAgent,
        "skill-agent": SkillAgent,
        "tool-agent": ToolAgent,
    }
    agent_type = agent_map.get(name)
    if agent_type is None:
        raise ValueError(f"unknown sub-agent: {name}")
    return agent_type(llm)


def _spec_for_agent(state: AgentState, agent_name: str) -> dict[str, Any] | None:
    for raw in state.get("pending_agent_specs", []):
        if not isinstance(raw, dict):
            continue
        if raw.get("agent_name") == agent_name:
            return raw
    return None


def _orchestrator_execute_overridden(orchestrator: Any) -> bool:
    from evoagent.core.orchestrator import SubAgentOrchestrator

    execute_attr = getattr(orchestrator, "execute", None)
    execute_func = getattr(execute_attr, "__func__", execute_attr)
    return execute_func is not SubAgentOrchestrator.execute


def _worker_output_to_agent(output: dict[str, Any], idx: int) -> dict[str, Any]:
    role = str(output.get("role") or output.get("agent_name") or f"worker-{idx + 1}")
    result = str(output.get("output") or output.get("result") or "")
    try:
        confidence = float(output.get("confidence", 0.5))
    except Exception:
        confidence = 0.5
    metadata = output.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    if output.get("error"):
        metadata = {**metadata, "failed": True, "reflection_error": output.get("error")}
    return AgentOutput(
        agent_name=role,
        objective=str(output.get("objective") or "reflection_revision"),
        result=result,
        confidence=confidence,
        metadata=metadata,
    ).model_dump(mode="json")


def _retry_reason(
    final_answer: str,
    quality_score: float,
    outputs: list[AgentOutput],
    post_actions: dict[str, Any],
    file_output: Any,
) -> str | None:
    if file_output is not None:
        return None

    if any(bool(output.metadata.get("timed_out", False)) for output in outputs):
        return "subagent_timeout"
    if outputs and all(bool(output.metadata.get("failed", False)) for output in outputs):
        return "all_subagents_failed"
    if any(bool(output.metadata.get("failed", False)) for output in outputs):
        return "partial_subagent_failure"

    fallback_markers = (
        "[Fallback-LLM]",
        "[subagent-error]",
        "[subagent-timeout]",
        "确定性兜底",
        "无法直接完成",
        "无法完成该任务",
        "无可用降级方案",
    )
    if any(marker in final_answer for marker in fallback_markers):
        return "fallback_answer"

    if quality_score < settings.graph_retry_quality_threshold:
        failure_class = str(post_actions.get("failure_class", "") or "")
        return failure_class if failure_class and failure_class != "none" else "low_quality"

    return None


def _correctness_score(
    avg_confidence: float, outputs: list[AgentOutput], final_answer: str
) -> float:
    fallback_markers = (
        "[Fallback-LLM]",
        "[subagent-error]",
        "[subagent-timeout]",
        "无法直接完成",
        "无法完成该任务",
        "无可用降级方案",
    )
    has_failure_marker = any(marker in final_answer for marker in fallback_markers)
    has_tool_evidence = bool(successful_tool_events_from_outputs(outputs))
    if has_tool_evidence and not has_failure_marker:
        return max(avg_confidence, 0.86)
    return avg_confidence


def _plan_steps(plan: dict | None) -> list[dict[str, Any]]:
    if not plan:
        return []
    raw_steps = plan.get("steps", [])
    if not isinstance(raw_steps, list):
        return []
    return [step for step in raw_steps if isinstance(step, dict)]


def _default_plan(user_query: str) -> Plan:
    return Plan(
        reasoning="fallback plan",
        steps=[
            StepSpec(
                sequence=1,
                role="researcher",
                description=user_query,
                depends_on=[],
                expected_output="Concise answer or findings.",
            )
        ],
    )


def planner_node(state: AgentState) -> AgentState:
    user_query = state.get("user_query", "")
    prompt = (
        "Create an execution plan for the user query. Return JSON matching this schema: "
        '{"reasoning": "...", "steps": [{"sequence": 1, "role": "researcher", '
        '"description": "...", "depends_on": [], "expected_output": "..."}]}.\n'
        f"User query: {user_query}"
    )
    service = StructuredLLMService()
    try:
        plan = service.generate_structured_sync(
            prompt,
            Plan,
            system="You are a planner for a multi-agent runtime.",
        )
        if not plan.steps:
            plan = _default_plan(user_query)
    except Exception:
        plan = _default_plan(user_query)
    payload = plan.model_dump(mode="json")
    return {
        **state,
        "plan": payload,
        "steps": [
            {
                "sequence": step["sequence"],
                "role": step["role"],
                "description": step["description"],
                "status": "pending",
            }
            for step in payload["steps"]
        ],
        "current_step": int(state.get("current_step", 0) or 0),
        "worker_outputs": list(state.get("worker_outputs", [])),
        "metadata": dict(state.get("metadata", {})),
    }


def supervisor_node(state: AgentState) -> AgentState:
    steps = _plan_steps(state.get("plan"))
    current_step = int(state.get("current_step", 0) or 0)
    metadata = dict(state.get("metadata", {}))
    if current_step < len(steps):
        spec = steps[current_step]
        metadata["next_worker"] = spec.get("role", "researcher")
        metadata["current_step_spec"] = spec
    return {**state, "metadata": metadata}


def worker_node(state: AgentState) -> AgentState:
    from evoagent.agents.workers import (
        coder_worker,
        memory_worker,
        researcher_worker,
        reviewer_worker,
        tool_worker,
    )

    role = str(state.get("metadata", {}).get("next_worker", "researcher"))
    worker_map = {
        "researcher": researcher_worker,
        "coder": coder_worker,
        "tool": tool_worker,
        "memory": memory_worker,
        "reviewer": reviewer_worker,
        "reporter": reviewer_worker,
    }
    worker = worker_map.get(role, researcher_worker)
    outputs = list(state.get("worker_outputs", []))
    steps = list(state.get("steps", []))
    current_step = int(state.get("current_step", 0) or 0)
    try:
        output = worker(state, LLMService())
    except Exception as exc:
        output = {
            "role": role,
            "output": "",
            "artifacts": [],
            "confidence": 0.0,
            "error": str(exc),
        }
    outputs.append(output)
    if current_step < len(steps):
        steps[current_step] = {
            **steps[current_step],
            "status": "failed" if output.get("error") else "succeeded",
            "output": output,
            "error": output.get("error"),
        }
    return {
        **state,
        "steps": steps,
        "worker_outputs": outputs,
        "current_step": current_step + 1,
    }


def _avg_worker_confidence(outputs: list[dict]) -> float:
    if not outputs:
        return 0.0
    values = []
    for output in outputs:
        try:
            values.append(float(output.get("confidence", 0.0)))
        except Exception:
            values.append(0.0)
    return sum(values) / max(len(values), 1)


def _reflection_needed(state: AgentState) -> bool:
    if not settings.reflection_enabled:
        return False
    metadata = state.get("metadata", {})
    if metadata.get("reflection_done"):
        return False
    plan_steps = _plan_steps(state.get("plan"))
    steps = state.get("steps", [])
    confidence = float(state.get("confidence", 0.0) or 0.0)
    failed = any(step.get("status") == "failed" for step in steps)
    avg_confidence = _avg_worker_confidence(list(state.get("worker_outputs", [])))
    complex_low = (
        len(plan_steps) > settings.reflection_complex_threshold
        and avg_confidence < 0.75
    )
    return confidence < settings.reflection_min_confidence or failed or complex_low


def reflection_node(state: AgentState) -> AgentState:
    metadata = dict(state.get("metadata", {}))
    metadata["reflection_done"] = True
    metadata["reflection_triggered"] = True
    outputs = list(state.get("worker_outputs", []))
    prompt = (
        "Review these worker outputs and return JSON with needs_revision, issues, "
        f"revised_outputs, confidence.\nWorker outputs: {outputs}"
    )
    try:
        result = StructuredLLMService().generate_structured_sync(
            prompt,
            ReflectionResult,
            system="You are a critical reviewer. Examine the following agent outputs and identify issues.",
        )
    except Exception:
        result = ReflectionResult(
            needs_revision=False,
            issues=[],
            revised_outputs=None,
            confidence=state.get("confidence", 0.0) or 0.0,
        )
    if result.needs_revision and result.revised_outputs is not None:
        outputs = result.revised_outputs
    updated: AgentState = {
        **state,
        "worker_outputs": outputs,
        "confidence": float(result.confidence),
        "metadata": metadata,
    }
    if state.get("agent_outputs") and result.revised_outputs is not None:
        updated["agent_outputs"] = [
            _worker_output_to_agent(output, idx) for idx, output in enumerate(outputs)
        ]
    return updated


def reporter_node(state: AgentState) -> AgentState:
    outputs = list(state.get("worker_outputs", []))
    answer_parts = [
        str(output.get("output", "")).strip()
        for output in outputs
        if str(output.get("output", "")).strip()
    ]
    final_answer = (
        "\n\n".join(answer_parts) if answer_parts else "No worker output was produced."
    )
    confidence = (
        _avg_worker_confidence(outputs)
        if outputs
        else float(state.get("confidence", 0.0) or 0.0)
    )
    return {**state, "final_answer": final_answer, "confidence": confidence}


def should_continue(state: AgentState) -> str:
    if int(state.get("current_step", 0) or 0) < len(_plan_steps(state.get("plan"))):
        return "supervisor"
    if _reflection_needed(state):
        return "reflection"
    return "reporter"


def prepare_runtime_node(state: AgentState) -> AgentState:
    services = _services_from_config()
    start = time.perf_counter()
    query = str(state.get("user_query", ""))
    input_had_context = "context" in state
    input_confidence = state.get("confidence")
    context_payload = dict(state.get("context", {}) or {})
    metadata = dict(state.get("metadata", {}) or {})
    run_id = str(state.get("run_id") or metadata.get("run_id") or uuid.uuid4())
    task_id = str(state.get("task_id") or run_id)

    session_id = str(context_payload.get("session_id", "default")).strip() or "default"
    trace: list[dict[str, Any]] = []
    conversation_turns: list[dict[str, str]] = []
    if settings.conversation_context_enabled:
        t_conv = time.perf_counter()
        conversation_turns = services.memory.conversation.recent_turns(
            session_id=session_id,
            limit=max(1, settings.conversation_history_window),
        )
        trace.append(
            {
                "step": "conversation_recall",
                "elapsed_ms": _now_ms(t_conv),
            }
        )
        context_payload["conversation_history"] = conversation_turns
    context_payload["session_id"] = session_id

    recent_for_prior = services.memory.episodic.search_recent(limit=20)
    if "episode_count" not in context_payload:
        context_payload["episode_count"] = len(recent_for_prior)
    if "historical_success_rate" not in context_payload:
        if recent_for_prior:
            success_count = sum(1 for rec in recent_for_prior if rec.success)
            context_payload["historical_success_rate"] = success_count / max(
                len(recent_for_prior), 1
            )
        else:
            context_payload["historical_success_rate"] = 0.0

    services.observer.on_task_start(task_id, query)
    prepared: AgentState = {
        **state,
        "run_id": run_id,
        "task_id": task_id,
        "context": context_payload,
        "start_time": start,
        "trace": trace,
        "steps": [],
        "worker_outputs": [],
        "agent_outputs": [],
        "semantic_hints": [],
        "matched_skill": None,
        "match_score": 0.0,
        "pending_agent_specs": [],
        "agent_result_batches": [],
        "retry_count": int(context_payload.get("retry_count", 0) or 0),
        "retry_reason": None,
        "retry_history": [],
        "retry_requested": False,
        "post_actions": {},
        "file_output": None,
        "completed": False,
        "finalized": False,
        "metadata": {
            **metadata,
            "run_id": run_id,
            "session_id": session_id,
            "conversation_turns_used": len(conversation_turns),
            "graph_input_had_context": input_had_context,
            "graph_input_confidence": input_confidence,
        },
        "plan": {
            "reasoning": "langgraph_evoagent_runtime",
            "steps": [
                {
                    "sequence": 1,
                    "role": "runtime",
                    "description": "Prepare task context and historical priors.",
                    "expected_output": "Task context",
                },
                {
                    "sequence": 2,
                    "role": "runtime",
                    "description": "Evaluate local time fast path.",
                    "expected_output": "Time answer or pass-through",
                },
                {
                    "sequence": 3,
                    "role": "runtime",
                    "description": "Evaluate weather fast path.",
                    "expected_output": "Weather answer or pass-through",
                },
                {
                    "sequence": 4,
                    "role": "skill",
                    "description": "Match evolved skills and branch when one is applicable.",
                    "expected_output": "Skill match decision",
                },
                {
                    "sequence": 5,
                    "role": "skill",
                    "description": "Execute the matched skill, otherwise continue to agent runtime.",
                    "expected_output": "Skill result or pass-through",
                },
                {
                    "sequence": 6,
                    "role": "memory",
                    "description": "Recall semantic hints for the task.",
                    "expected_output": "Relevant memory hints",
                },
                {
                    "sequence": 7,
                    "role": "router",
                    "description": "Decide task difficulty and required capabilities.",
                    "expected_output": "Task decision",
                },
                {
                    "sequence": 8,
                    "role": "agent",
                    "description": "Build a graph-visible sub-agent execution plan.",
                    "expected_output": "Selected sub-agent specs",
                },
                {
                    "sequence": 9,
                    "role": "agent",
                    "description": "Run selected sub-agent nodes in parallel and join outputs.",
                    "expected_output": "Sub-agent outputs",
                },
                {
                    "sequence": 10,
                    "role": "reflection",
                    "description": "Optionally revise low-confidence sub-agent outputs before composing.",
                    "expected_output": "Reviewed sub-agent outputs",
                },
                {
                    "sequence": 11,
                    "role": "aggregator",
                    "description": "Compose the final answer and requested file output.",
                    "expected_output": "Final answer",
                },
                {
                    "sequence": 12,
                    "role": "feedback",
                    "description": "Score result and run feedback actions.",
                    "expected_output": "Post-task actions",
                },
                {
                    "sequence": 13,
                    "role": "runtime",
                    "description": "Retry low-quality or failed agent runs when budget remains.",
                    "expected_output": "Retry decision",
                },
                {
                    "sequence": 14,
                    "role": "runtime",
                    "description": "Persist conversation state and finish runtime metadata.",
                    "expected_output": "ExecutionResult metadata",
                },
            ],
        },
    }
    _append_graph_step(
        prepared,
        role="runtime",
        description="prepare_context",
        output={
            "session_id": session_id,
            "conversation_turns_used": len(conversation_turns),
            "episode_count": context_payload.get("episode_count", 0),
        },
        duration_ms=_now_ms(start),
    )
    return prepared


def time_fastpath_node(state: AgentState) -> AgentState:
    if state.get("completed"):
        return state

    services = _services_from_config()
    query = str(state.get("user_query", ""))
    task = _task_from_state(state)
    started = time.perf_counter()
    file_output_requested = services._file_output_requested(query)

    if services._is_time_query(query) and not file_output_requested:
        decision = TaskDecision(
            difficulty=Difficulty.SIMPLE,
            score=0.05,
            reasoning="builtin_time_fastpath",
            features=TaskFeatures(
                estimated_steps=1,
                needs_external_tools=False,
                has_historical_pattern=False,
                historical_pattern_score=0.0,
                confidence=0.98,
            ),
            confidence=0.98,
            fallback_plan="若本地时钟异常，回退通用路由。",
        )
        services.observer.on_task_decision(
            task.task_id, decision.difficulty.value, decision.confidence
        )
        final_answer = services._format_local_time_answer()
        _append_trace(state, "utility_fastpath", _now_ms(started))
        state.update(
            {
                "completed": True,
                "decision": decision.model_dump(mode="json"),
                "final_answer": final_answer,
                "quality_score": 0.96,
                "confidence": decision.confidence,
                "agent_outputs": [],
                "worker_outputs": [],
            }
        )
        metadata = dict(state.get("metadata", {}))
        metadata["mode"] = "builtin_time_fastpath"
        state["metadata"] = metadata
        _append_graph_step(
            state,
            role="runtime",
            description="builtin_time_fastpath",
            output={"final_answer": final_answer},
            duration_ms=_now_ms(started),
        )
        return state

    _append_graph_step(
        state,
        role="runtime",
        description="time_fastpath_pass_through",
        output={"file_output_requested": file_output_requested},
        duration_ms=_now_ms(started),
    )
    return state


def weather_fastpath_node(state: AgentState) -> AgentState:
    if state.get("completed"):
        return state

    services = _services_from_config()
    query = str(state.get("user_query", ""))
    task = _task_from_state(state)
    started = time.perf_counter()
    file_output_requested = services._file_output_requested(query)

    if is_weather_query(query) and not file_output_requested:
        weather_location = extract_weather_location(query)
        if weather_location:
            decision = TaskDecision(
                difficulty=Difficulty.SIMPLE,
                score=0.18,
                reasoning="builtin_weather_fastpath",
                features=TaskFeatures(
                    estimated_steps=1,
                    needs_external_tools=True,
                    has_historical_pattern=False,
                    historical_pattern_score=0.0,
                    confidence=0.88,
                ),
                confidence=0.88,
                fallback_plan="若天气服务不可用，回退通用工具编排。",
            )
            services.observer.on_task_decision(
                task.task_id, decision.difficulty.value, decision.confidence
            )
            t_weather = time.perf_counter()
            weather_answer = services.fetch_weather_report(weather_location)
            _append_trace(state, "weather_fastpath", _now_ms(t_weather))
            if not weather_answer.startswith("error:"):
                state.update(
                    {
                        "completed": True,
                        "decision": decision.model_dump(mode="json"),
                        "final_answer": weather_answer,
                        "quality_score": 0.9,
                        "confidence": decision.confidence,
                        "agent_outputs": [],
                        "worker_outputs": [],
                    }
                )
                metadata = dict(state.get("metadata", {}))
                metadata.update(
                    {
                        "mode": "builtin_weather_fastpath",
                        "weather_location": weather_location,
                    }
                )
                state["metadata"] = metadata
                _append_graph_step(
                    state,
                    role="runtime",
                    description="builtin_weather_fastpath",
                    output={
                        "weather_location": weather_location,
                        "final_answer": weather_answer,
                    },
                    duration_ms=_now_ms(started),
                )
                return state
            _append_trace(
                state,
                "weather_fastpath_error",
                0.0,
                error=weather_answer[:180],
            )

    _append_graph_step(
        state,
        role="runtime",
        description="weather_fastpath_pass_through",
        output={"file_output_requested": file_output_requested},
        duration_ms=_now_ms(started),
    )
    return state


def fastpath_node(state: AgentState) -> AgentState:
    state = time_fastpath_node(state)
    return weather_fastpath_node(state)


def skill_match_node(state: AgentState) -> AgentState:
    if state.get("completed"):
        return state

    services = _services_from_config()
    query = str(state.get("user_query", ""))
    started = time.perf_counter()

    matched_skill, match_score = services.skill_artifacts.find_relevant(query)
    state["matched_skill"] = matched_skill
    state["match_score"] = match_score
    metadata = dict(state.get("metadata", {}))
    metadata.update({"matched_skill": matched_skill, "match_score": match_score})
    state["metadata"] = metadata

    if matched_skill is None or match_score < 0.34:
        _append_graph_step(
            state,
            role="skill",
            description="skill_match_pass_through",
            output={"match_score": match_score},
            duration_ms=_now_ms(started),
        )
        return state

    _append_graph_step(
        state,
        role="skill",
        description="skill_match",
        output={"matched_skill": matched_skill, "match_score": match_score},
        duration_ms=_now_ms(started),
    )
    return state


def skill_execute_node(state: AgentState) -> AgentState:
    if state.get("completed"):
        return state

    matched_skill = state.get("matched_skill")
    match_score = float(state.get("match_score", 0.0) or 0.0)
    if not isinstance(matched_skill, dict) or match_score < 0.34:
        return state

    services = _services_from_config()
    query = str(state.get("user_query", ""))
    task = _task_from_state(state)
    started = time.perf_counter()

    try:
        services.observer.emit(
            "skill_runtime_start",
            task_id=task.task_id,
            skill_name=str(matched_skill.get("name", "unknown")),
            skill_version=str(matched_skill.get("version", "0.0.0")),
            skill_status=str(matched_skill.get("status", "unknown")),
        )
        t_skill = time.perf_counter()
        skill_result = services.skill_artifacts.execute(matched_skill, query)
        _append_trace(state, "skill_runtime", _now_ms(t_skill))

        decision = services.router.decide(task)
        services.observer.on_task_decision(
            task.task_id, decision.difficulty.value, decision.confidence
        )
        skill_name = str(matched_skill.get("name", "unknown"))
        skill_version = str(matched_skill.get("version", "0.0.0"))
        skill_status = str(matched_skill.get("status", "active"))
        calls, success_rate = services.skill_registry.update_version_score(
            skill_name, skill_version, True
        )
        rollout_meta: dict[str, object] = {
            "status": skill_status,
            "calls": calls,
            "success_rate": success_rate,
            "decision": "active_kept"
            if skill_status == "active"
            else "candidate_keep",
        }
        if skill_status == "candidate" and calls >= settings.skill_candidate_min_calls:
            if success_rate >= settings.skill_candidate_promote_success_rate:
                services.skill_registry.activate_version(skill_name, skill_version)
                services.skill_artifacts.activate_version(skill_name, skill_version)
                rollout_meta["decision"] = "candidate_promoted"
            elif success_rate < settings.skill_candidate_archive_success_rate:
                services.skill_registry.set_version_status(
                    skill_name, skill_version, "archived"
                )
                services.skill_artifacts.set_version_status(
                    skill_name, skill_version, "archived"
                )
                rollout_meta["decision"] = "candidate_archived"

        ab_meta: dict[str, object] = {}
        if random.random() < settings.skill_ab_test_rate:
            t_ab = time.perf_counter()
            outputs = _run_coro(services.supervisor.execute(task, decision))
            baseline_answer = services.supervisor.aggregate(task, outputs)
            _append_trace(state, "ab_baseline", _now_ms(t_ab))
            baseline_is_fallback = baseline_answer.startswith("[Fallback-LLM]")
            skill_len = len(skill_result.strip())
            baseline_len = len(baseline_answer.strip())
            if baseline_is_fallback:
                winner = "skill"
            elif baseline_len > int(skill_len * 1.3):
                winner = "baseline"
            elif skill_len > int(baseline_len * 1.3):
                winner = "skill"
            else:
                winner = "tie"
            services.skill_artifacts.record_ab_result(skill_name, skill_version, winner)
            ab_meta = {
                "ab_test": True,
                "winner": winner,
                "baseline_answer_preview": baseline_answer[:240],
            }

        t_forget = time.perf_counter()
        forgotten = services.memory.run_semantic_forgetting()
        _append_trace(state, "semantic_forgetting", _now_ms(t_forget))

        state.update(
            {
                "completed": True,
                "decision": decision.model_dump(mode="json"),
                "final_answer": skill_result,
                "quality_score": 0.88,
                "confidence": decision.confidence,
                "agent_outputs": [],
                "worker_outputs": [],
                "post_actions": {"semantic_forgetting": f"removed={forgotten}"},
            }
        )
        metadata = dict(state.get("metadata", {}))
        metadata.update(
            {
                "mode": "auto_evolved_skill",
                "matched_skill": matched_skill,
                "match_score": match_score,
                "skill_rollout": rollout_meta,
                **ab_meta,
            }
        )
        state["metadata"] = metadata
        _append_graph_step(
            state,
            role="skill",
            description="skill_runtime",
            output={
                "matched_skill": matched_skill,
                "match_score": match_score,
                "preview": skill_result[:300],
            },
            duration_ms=_now_ms(started),
        )
        return state
    except Exception as exc:
        if matched_skill is not None:
            skill_name = str(matched_skill.get("name", "unknown"))
            skill_version = str(matched_skill.get("version", "0.0.0"))
            skill_status = str(matched_skill.get("status", "active"))
            calls, success_rate = services.skill_registry.update_version_score(
                skill_name, skill_version, False
            )
            if skill_status == "candidate" and calls >= settings.skill_candidate_min_calls:
                if success_rate < settings.skill_candidate_archive_success_rate:
                    services.skill_registry.set_version_status(
                        skill_name, skill_version, "archived"
                    )
                    services.skill_artifacts.set_version_status(
                        skill_name, skill_version, "archived"
                    )
        _append_trace(
            state,
            "skill_runtime_error",
            0.0,
            error=str(exc)[:180],
        )
        services.observer.emit(
            "skill_runtime_error",
            task_id=task.task_id,
            error=str(exc)[:220],
        )
        _append_graph_step(
            state,
            role="skill",
            description="skill_runtime_error",
            status="failed",
            output={"matched_skill": matched_skill, "match_score": match_score},
            error=str(exc)[:240],
            duration_ms=_now_ms(started),
        )
        return state


def skill_runtime_node(state: AgentState) -> AgentState:
    state = skill_match_node(state)
    if skill_execution_ready(state) == "skill_execute":
        state = skill_execute_node(state)
    return state


def semantic_recall_node(state: AgentState) -> AgentState:
    if state.get("completed"):
        return state

    services = _services_from_config()
    query = str(state.get("user_query", ""))
    started = time.perf_counter()

    semantic_hints = services.memory.recall_semantic(query, top_k=3)
    _append_trace(state, "semantic_recall", _now_ms(started))
    state["semantic_hints"] = semantic_hints
    _append_graph_step(
        state,
        role="memory",
        description="semantic_recall",
        output={"hits": len(semantic_hints)},
        duration_ms=_now_ms(started),
    )
    return state


def route_task_node(state: AgentState) -> AgentState:
    if state.get("completed"):
        return state

    services = _services_from_config()
    task = _task_from_state(state)
    started = time.perf_counter()

    services.memory.add_step("route:start")
    decision = services.router.decide(task)
    services.observer.on_task_decision(
        task.task_id, decision.difficulty.value, decision.confidence
    )
    _append_trace(state, "routing", _now_ms(started))
    services.memory.add_step(f"route:decision={decision.difficulty}")
    state["decision"] = decision.model_dump(mode="json")
    _append_graph_step(
        state,
        role="router",
        description="route_task",
        output={
            "difficulty": decision.difficulty.value,
            "score": decision.score,
            "confidence": decision.confidence,
            "capabilities": [
                capability.value
                for capability in decision.features.required_capabilities
            ],
        },
        duration_ms=_now_ms(started),
    )
    return state


def plan_agents_node(state: AgentState) -> AgentState:
    if state.get("completed"):
        return state

    services = _services_from_config()
    task = _task_from_state(state)
    decision = _decision_from_state(state)
    started = time.perf_counter()

    if _orchestrator_execute_overridden(services.orchestrator):
        outputs = _run_coro(services.supervisor.execute(task, decision))
        state.update(
            {
                "pending_agent_specs": [],
                "agent_result_batches": [
                    {
                        "idx": idx,
                        "retry_count": int(state.get("retry_count", 0) or 0),
                        "duration_ms": _now_ms(started),
                        "output": output.model_dump(mode="json"),
                    }
                    for idx, output in enumerate(outputs)
                ],
                "dispatch_started_at": started,
            }
        )
        _append_graph_step(
            state,
            role="agent",
            description="plan_sub_agents_compat",
            output={
                "agent_count": len(outputs),
                "agents": [output.agent_name for output in outputs],
            },
            duration_ms=_now_ms(started),
        )
        return state

    plan = services.orchestrator._spawn_plan(task, decision)
    specs = [
        {
            "idx": idx,
            "agent_name": agent.name,
            "node": _AGENT_NODE_BY_NAME.get(agent.name, ""),
            "objective": objective,
        }
        for idx, (agent, objective) in enumerate(plan)
    ]
    state.update(
        {
            "pending_agent_specs": specs,
            "agent_result_batches": [],
            "dispatch_started_at": started,
            "agent_outputs": [],
            "worker_outputs": [],
            "current_step": 0,
        }
    )
    _append_graph_step(
        state,
        role="agent",
        description="plan_sub_agents",
        output={
            "agent_count": len(specs),
            "agents": [spec["agent_name"] for spec in specs],
        },
        duration_ms=_now_ms(started),
    )
    return state


def select_agent_nodes(state: AgentState) -> list[str]:
    nodes = [
        str(spec.get("node"))
        for spec in state.get("pending_agent_specs", [])
        if isinstance(spec, dict) and spec.get("node")
    ]
    return nodes or ["join_agents"]


def _run_single_planned_agent(state: AgentState, agent_name: str) -> AgentOutput | None:
    spec = _spec_for_agent(state, agent_name)
    if spec is None:
        return None

    services = _services_from_config()
    task = _task_from_state(state)
    objective = str(spec.get("objective") or "")
    agent = _agent_for_name(agent_name, services.llm)
    retries = max(0, settings.subagent_retry_attempts)
    backoff = max(0.1, settings.subagent_retry_backoff_seconds)
    timeout = max(0.1, float(settings.orchestrator_task_timeout_seconds))

    async def _run() -> AgentOutput:
        start = time.perf_counter()
        if services.observer is not None:
            services.observer.on_subagent_start(
                task_id=task.task_id, agent_name=agent.name, objective=objective
            )

        last_error: str | None = None
        timed_out = False
        for attempt in range(retries + 1):
            try:
                output = await asyncio.wait_for(agent.run(task, objective), timeout=timeout)
                elapsed_ms = (time.perf_counter() - start) * 1000
                tool_calls = (
                    len(output.metadata.get("tool_events", []))
                    if isinstance(output.metadata, dict)
                    else 0
                )
                if services.observer is not None:
                    services.observer.on_subagent_end(
                        task_id=task.task_id,
                        agent_name=agent.name,
                        elapsed_ms=elapsed_ms,
                        tool_calls=tool_calls,
                        success=not bool(output.metadata.get("failed", False)),
                        timed_out=False,
                    )
                return output
            except asyncio.TimeoutError:
                timed_out = True
                last_error = "timed out"
            except Exception as exc:
                last_error = str(exc)

            if services.observer is not None and attempt < retries:
                services.observer.on_subagent_retry(
                    task_id=task.task_id,
                    agent_name=agent.name,
                    attempt=attempt + 1,
                    error=last_error or "unknown error",
                )
            if attempt < retries:
                await asyncio.sleep(backoff * (2**attempt) + random.uniform(0, 0.2))

        elapsed_ms = (time.perf_counter() - start) * 1000
        metadata: dict[str, Any] = {"retries": retries, "failed": True}
        if timed_out:
            metadata["timed_out"] = True
        result = (
            "[subagent-timeout] 任务级超时触发，使用部分结果继续聚合。"
            if timed_out
            else f"[subagent-error] {last_error or 'unknown error'}"
        )
        if services.observer is not None:
            services.observer.on_subagent_end(
                task_id=task.task_id,
                agent_name=agent.name,
                elapsed_ms=elapsed_ms,
                tool_calls=0,
                success=False,
                timed_out=timed_out,
            )
        return AgentOutput(
            agent_name=agent.name,
            objective=objective,
            result=result,
            confidence=0.15 if timed_out else 0.2,
            metadata=metadata,
        )

    return _run_coro(_run())


def _agent_node_result(state: AgentState, agent_name: str) -> AgentState:
    started = time.perf_counter()
    output = _run_single_planned_agent(state, agent_name)
    if output is None:
        return {"agent_result_batches": []}
    spec = _spec_for_agent(state, agent_name) or {}
    return {
        "agent_result_batches": [
            {
                "idx": int(spec.get("idx", 0)),
                "retry_count": int(state.get("retry_count", 0) or 0),
                "duration_ms": _now_ms(started),
                "output": output.model_dump(mode="json"),
            }
        ]
    }


def search_agent_node(state: AgentState) -> AgentState:
    return _agent_node_result(state, "search-agent")


def reasoning_agent_node(state: AgentState) -> AgentState:
    return _agent_node_result(state, "reasoning-agent")


def code_agent_node(state: AgentState) -> AgentState:
    return _agent_node_result(state, "code-agent")


def file_agent_node(state: AgentState) -> AgentState:
    return _agent_node_result(state, "file-agent")


def integration_agent_node(state: AgentState) -> AgentState:
    return _agent_node_result(state, "integration-agent")


def skill_agent_node(state: AgentState) -> AgentState:
    return _agent_node_result(state, "skill-agent")


def tool_agent_node(state: AgentState) -> AgentState:
    return _agent_node_result(state, "tool-agent")


def join_agent_outputs_node(state: AgentState) -> AgentState:
    if state.get("completed"):
        return state

    started = float(state.get("dispatch_started_at", time.perf_counter()))
    batches = [
        batch
        for batch in state.get("agent_result_batches", [])
        if (
            isinstance(batch, dict)
            and isinstance(batch.get("output"), dict)
            and int(batch.get("retry_count", 0) or 0)
            == int(state.get("retry_count", 0) or 0)
        )
    ]
    batches.sort(key=lambda item: int(item.get("idx", 0)))
    outputs = [AgentOutput.model_validate(batch["output"]) for batch in batches]
    _append_trace(state, "sub_agents", _now_ms(started))
    for batch, output in zip(batches, outputs):
        _append_graph_step(
            state,
            role="agent",
            description=f"run_{output.agent_name}",
            output={
                "confidence": output.confidence,
                "preview": output.result[:300],
            },
            duration_ms=float(batch.get("duration_ms", 0.0) or 0.0),
        )
    state.update(
        {
            "agent_outputs": [output.model_dump(mode="json") for output in outputs],
            "worker_outputs": [_agent_output_to_worker(output) for output in outputs],
            "current_step": len(outputs),
        }
    )
    _append_graph_step(
        state,
        role="agent",
        description="join_agent_outputs",
        output={"agent_count": len(outputs)},
        duration_ms=_now_ms(started),
    )
    return state


def dispatch_agents_node(state: AgentState) -> AgentState:
    state = plan_agents_node(state)
    for spec in state.get("pending_agent_specs", []):
        if not isinstance(spec, dict):
            continue
        node = str(spec.get("node", ""))
        agent_name = str(spec.get("agent_name", ""))
        if node and agent_name:
            result_state = _agent_node_result(state, agent_name)
            state["agent_result_batches"] = list(state.get("agent_result_batches", [])) + list(
                result_state.get("agent_result_batches", [])
            )
    return join_agent_outputs_node(state)


def compose_answer_node(state: AgentState) -> AgentState:
    if state.get("completed"):
        return state

    services = _services_from_config()
    query = str(state.get("user_query", ""))
    task = _task_from_state(state)
    decision = _decision_from_state(state)
    outputs = _agent_outputs_from_state(state)
    started = time.perf_counter()

    all_outputs_failed = bool(outputs) and all(
        bool(o.metadata.get("failed", False)) for o in outputs
    )
    any_output_timed_out = any(
        bool(o.metadata.get("timed_out", False)) for o in outputs
    )
    if all_outputs_failed or any_output_timed_out:
        final_answer = services.supervisor.deterministic_aggregate(task, outputs)
        _append_trace(state, "aggregate_skipped_timeout_or_failed", 0.0)
    elif (
        decision.difficulty == Difficulty.SIMPLE
        and len(outputs) == 1
        and not str(outputs[0].result).startswith("[Fallback-LLM]")
        and not bool(outputs[0].metadata.get("failed", False))
    ):
        final_answer = outputs[0].result
        _append_trace(state, "aggregate_skipped_simple_single", 0.0)
    else:
        final_answer = services.supervisor.aggregate(task, outputs)
        _append_trace(state, "aggregate", _now_ms(started))

    file_output = services._maybe_write_requested_file(query, final_answer)
    if file_output is not None:
        _append_trace(state, "file_output", _now_ms(started))
        final_answer = f"{final_answer}\n\n已写入文件：{file_output['path']}"

    state["final_answer"] = final_answer
    state["file_output"] = file_output
    _append_graph_step(
        state,
        role="aggregator",
        description="compose_answer",
        output={
            "final_answer_preview": final_answer[:300],
            "agent_count": len(outputs),
            "all_outputs_failed": all_outputs_failed,
            "timed_out": any_output_timed_out,
            "file_output": file_output,
        },
        duration_ms=_now_ms(started),
    )
    return state


def aggregate_outputs_node(state: AgentState) -> AgentState:
    return compose_answer_node(state)


def postprocess_output_node(state: AgentState) -> AgentState:
    if state.get("completed"):
        return state
    if state.get("file_output") is not None:
        return state

    services = _services_from_config()
    query = str(state.get("user_query", ""))
    final_answer = str(state.get("final_answer") or "")
    started = time.perf_counter()

    file_output = services._maybe_write_requested_file(query, final_answer)
    if file_output is not None:
        _append_trace(state, "file_output", _now_ms(started))
        final_answer = f"{final_answer}\n\n已写入文件：{file_output['path']}"
        state["final_answer"] = final_answer
    state["file_output"] = file_output
    _append_graph_step(
        state,
        role="runtime",
        description="postprocess_output",
        output={"file_output": file_output},
        duration_ms=_now_ms(started),
    )
    return state


def feedback_node(state: AgentState) -> AgentState:
    if state.get("completed"):
        return state

    services = _services_from_config()
    query = str(state.get("user_query", ""))
    task = _task_from_state(state)
    decision = _decision_from_state(state)
    outputs = _agent_outputs_from_state(state)
    final_answer = str(state.get("final_answer") or "")
    semantic_hints = list(state.get("semantic_hints", []))
    file_output = state.get("file_output")
    started = time.perf_counter()

    avg_confidence = sum(o.confidence for o in outputs) / max(len(outputs), 1)
    correctness = _correctness_score(avg_confidence, outputs, final_answer)
    quality_score = services.feedback.score_result(
        correctness=correctness,
        efficiency=max(0.3, 1.0 - decision.score),
        user_satisfaction=0.8,
    )

    post_actions = services.feedback.run_post_task(
        task_id=task.task_id,
        task_text=query,
        final_answer=final_answer,
        quality_score=quality_score,
        path_length=len(outputs),
        confidence=avg_confidence,
        decision=decision,
        outputs=outputs,
        semantic_hints=semantic_hints,
        trace=list(state.get("trace", [])),
    )
    _append_trace(state, "feedback", _now_ms(started))

    t_forget = time.perf_counter()
    forgotten = services.memory.run_semantic_forgetting()
    _append_trace(state, "semantic_forgetting", _now_ms(t_forget))
    post_actions["semantic_forgetting"] = f"removed={forgotten}"
    if file_output is not None:
        post_actions["file_output"] = file_output

    retry_count = int(state.get("retry_count", 0) or 0)
    retry_reason = None
    if settings.graph_retry_enabled and retry_count < max(0, settings.graph_max_retries):
        retry_reason = _retry_reason(
            final_answer=final_answer,
            quality_score=quality_score,
            outputs=outputs,
            post_actions=post_actions,
            file_output=file_output,
        )
    retry_requested = retry_reason is not None
    if retry_requested:
        post_actions["graph_retry"] = {
            "decision": "retry",
            "reason": retry_reason,
            "retry_count": retry_count,
            "max_retries": settings.graph_max_retries,
        }

    timed_out = any(bool(o.metadata.get("timed_out", False)) for o in outputs)
    task_profile = (
        "tooling"
        if decision.features.needs_external_tools
        else ("complex" if decision.features.estimated_steps >= 5 else "general")
    )
    state.update(
        {
            "completed": True,
            "quality_score": quality_score,
            "confidence": avg_confidence,
            "post_actions": post_actions,
            "retry_requested": retry_requested,
            "retry_reason": retry_reason,
        }
    )
    metadata = dict(state.get("metadata", {}))
    metadata.update(
        {
            "mode": "langgraph_evoagent_runtime",
            "task_profile": task_profile,
            "timed_out": timed_out,
            "retry_count": retry_count,
            "retry_requested": retry_requested,
            "retry_reason": retry_reason,
        }
    )
    state["metadata"] = metadata
    _append_graph_step(
        state,
        role="feedback",
        description="score_feedback_memory",
        output={
            "difficulty": decision.difficulty.value,
            "quality_score": quality_score,
            "agent_count": len(outputs),
            "semantic_forgetting_removed": forgotten,
        },
        duration_ms=_now_ms(started),
    )
    return state


def retry_prepare_node(state: AgentState) -> AgentState:
    if not state.get("retry_requested"):
        return state

    reason = str(state.get("retry_reason") or "unknown")
    previous_answer = str(state.get("final_answer") or "")
    retry_count = int(state.get("retry_count", 0) or 0) + 1
    retry_history = list(state.get("retry_history", []))
    retry_history.append(
        {
            "retry": retry_count,
            "reason": reason,
            "quality_score": float(state.get("quality_score", 0.0) or 0.0),
            "answer_preview": previous_answer[:240],
        }
    )

    context = dict(state.get("context", {}) or {})
    context.update(
        {
            "retry_count": retry_count,
            "last_failure_reason": reason,
            "previous_answer_preview": previous_answer[:240],
        }
    )
    metadata = dict(state.get("metadata", {}))
    metadata.update(
        {
            "retry_count": retry_count,
            "retry_reason": reason,
            "retry_history": retry_history,
        }
    )
    _append_trace(state, "graph_retry", 0.0, reason=reason, retry_count=retry_count)
    _append_graph_step(
        state,
        role="runtime",
        description="prepare_retry",
        output={"reason": reason, "retry_count": retry_count},
    )

    state.update(
        {
            "context": context,
            "metadata": metadata,
            "completed": False,
            "decision": None,
            "agent_outputs": [],
            "worker_outputs": [],
            "pending_agent_specs": [],
            "agent_result_batches": [],
            "current_step": 0,
            "final_answer": None,
            "quality_score": 0.0,
            "confidence": 0.0,
            "file_output": None,
            "post_actions": {},
            "retry_count": retry_count,
            "retry_history": retry_history,
            "retry_requested": False,
            "retry_reason": reason,
        }
    )
    return state


def agent_execution_node(state: AgentState) -> AgentState:
    """Backward-compatible wrapper for the split runtime execution nodes."""

    for node in (
        semantic_recall_node,
        route_task_node,
        dispatch_agents_node,
        compose_answer_node,
        feedback_node,
    ):
        state = node(state)
    return state


def finalize_runtime_node(state: AgentState) -> AgentState:
    if state.get("finalized"):
        return state

    services = _services_from_config()
    query = str(state.get("user_query", ""))
    task = _task_from_state(state)
    decision = _decision_from_state(state)
    final_answer = str(state.get("final_answer") or "")
    started = float(state.get("start_time", time.perf_counter()))
    latency = time.perf_counter() - started
    metadata = dict(state.get("metadata", {}))
    trace = list(state.get("trace", []))
    post_actions = dict(state.get("post_actions", {}))
    semantic_hints = list(state.get("semantic_hints", []))
    file_output = state.get("file_output")
    timed_out = bool(metadata.get("timed_out", False))
    task_profile = str(metadata.get("task_profile") or metadata.get("mode") or "runtime")

    services.observer.on_task_end(
        task_id=task.task_id,
        elapsed_ms=latency * 1000,
        difficulty=decision.difficulty.value,
        task_profile=task_profile,
        timed_out=timed_out,
    )

    session_id = str(metadata.get("session_id", "default"))
    if settings.conversation_context_enabled and final_answer:
        services.memory.conversation.add_turn(
            session_id=session_id,
            task_id=task.task_id,
            role="user",
            content=query,
        )
        services.memory.conversation.add_turn(
            session_id=session_id,
            task_id=task.task_id,
            role="assistant",
            content=final_answer,
        )
    services.memory.working.reset()

    result_metadata = {
        **metadata,
        "post_actions": post_actions,
        "semantic_hints": semantic_hints,
        "file_output": file_output,
        "session_id": session_id,
        "conversation_turns_used": metadata.get("conversation_turns_used", 0),
        "trace": trace,
    }
    state.update(
        {
            "finalized": True,
            "latency_seconds": latency,
            "metadata": result_metadata,
        }
    )
    _append_graph_step(
        state,
        role="runtime",
        description="finalize_runtime",
        output={
            "mode": result_metadata.get("mode"),
            "quality_score": state.get("quality_score", 0.0),
            "latency_seconds": latency,
        },
        duration_ms=latency * 1000,
    )
    return state


def fastpath_or_skill(state: AgentState) -> str:
    return "finalize" if state.get("completed") else "skill"


def fastpath_or_weather(state: AgentState) -> str:
    return "finalize" if state.get("completed") else "weather_fastpath"


def fastpath_or_skill_match(state: AgentState) -> str:
    return "finalize" if state.get("completed") else "skill_match"


def skill_or_agent(state: AgentState) -> str:
    return "finalize" if state.get("completed") else "agent"


def skill_execution_ready(state: AgentState) -> str:
    if state.get("completed"):
        return "finalize"
    matched_skill = state.get("matched_skill")
    try:
        match_score = float(state.get("match_score", 0.0) or 0.0)
    except Exception:
        match_score = 0.0
    if isinstance(matched_skill, dict) and match_score >= 0.34:
        return "skill_execute"
    return "semantic_recall"


def skill_execution_or_recall(state: AgentState) -> str:
    return "finalize" if state.get("completed") else "semantic_recall"


def skill_or_execution(state: AgentState) -> str:
    return "finalize" if state.get("completed") else "semantic_recall"


def _reflection_route(state: AgentState, default: str) -> str:
    if not settings.reflection_enabled:
        return default
    metadata = dict(state.get("metadata", {}))
    if metadata.get("graph_input_had_context", True):
        return default
    if metadata.get("reflection_done"):
        return default
    try:
        input_confidence = float(metadata.get("graph_input_confidence"))
    except Exception:
        input_confidence = float(state.get("confidence", 0.0) or 0.0)
    if input_confidence < settings.reflection_min_confidence:
        return "reflection"
    if _reflection_needed(state):
        return "reflection"
    return default


def dispatch_or_reflection(state: AgentState) -> str:
    return _reflection_route(state, "compose_answer")


def feedback_or_retry(state: AgentState) -> str:
    return "retry_prepare" if state.get("retry_requested") else "finalize"


def feedback_or_reflection(state: AgentState) -> str:
    return _reflection_route(state, "finalize")


def agent_or_reflection(state: AgentState) -> str:
    return feedback_or_reflection(state)


def execution_result_from_state(state: AgentState) -> ExecutionResult:
    decision = _decision_from_state(state)
    outputs = _agent_outputs_from_state(state)
    metadata = dict(state.get("metadata", {}))
    return ExecutionResult(
        task_id=str(state.get("task_id") or state.get("run_id") or uuid.uuid4()),
        decision=decision,
        outputs=outputs,
        final_answer=str(state.get("final_answer") or ""),
        quality_score=float(state.get("quality_score", 0.0) or 0.0),
        latency_seconds=float(state.get("latency_seconds", 0.0) or 0.0),
        metadata=metadata,
    )


def _checkpoint_path() -> str:
    db_path = settings.db_path
    root = db_path if db_path.suffix == "" else db_path.parent
    root.mkdir(parents=True, exist_ok=True)
    return str(root / "checkpoints.db")


_checkpoint_context = None


def _make_checkpointer():
    global _checkpoint_context
    if SqliteSaver is not None:
        try:
            _checkpoint_context = SqliteSaver.from_conn_string(_checkpoint_path())
            return _checkpoint_context.__enter__()
        except Exception:
            _checkpoint_context = None
    if MemorySaver is None:
        return None
    return MemorySaver()


def build_legacy_graph():
    graph = StateGraph(AgentState)
    graph.add_node("planner", planner_node)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("worker", worker_node)
    graph.add_node("reflection", reflection_node)
    graph.add_node("reporter", reporter_node)
    graph.set_entry_point("planner")
    graph.add_edge("planner", "supervisor")
    graph.add_edge("supervisor", "worker")
    graph.add_conditional_edges(
        "worker",
        should_continue,
        {
            "supervisor": "supervisor",
            "reflection": "reflection",
            "reporter": "reporter",
        },
    )
    graph.add_edge("reflection", "reporter")
    graph.add_edge("reporter", END)
    return graph


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("prepare", prepare_runtime_node)
    graph.add_node("time_fastpath", time_fastpath_node)
    graph.add_node("weather_fastpath", weather_fastpath_node)
    graph.add_node("skill_match", skill_match_node)
    graph.add_node("skill_execute", skill_execute_node)
    graph.add_node("semantic_recall", semantic_recall_node)
    graph.add_node("route", route_task_node)
    graph.add_node("plan_agents", plan_agents_node)
    graph.add_node("search_agent", search_agent_node)
    graph.add_node("reasoning_agent", reasoning_agent_node)
    graph.add_node("code_agent", code_agent_node)
    graph.add_node("file_agent", file_agent_node)
    graph.add_node("integration_agent", integration_agent_node)
    graph.add_node("skill_agent", skill_agent_node)
    graph.add_node("tool_agent", tool_agent_node)
    graph.add_node("join_agents", join_agent_outputs_node)
    graph.add_node("compose_answer", compose_answer_node)
    graph.add_node("feedback", feedback_node)
    graph.add_node("retry_prepare", retry_prepare_node)
    graph.add_node("reflection", reflection_node)
    graph.add_node("finalize", finalize_runtime_node)
    graph.set_entry_point("prepare")
    graph.add_edge("prepare", "time_fastpath")
    graph.add_conditional_edges(
        "time_fastpath",
        fastpath_or_weather,
        {
            "weather_fastpath": "weather_fastpath",
            "finalize": "finalize",
        },
    )
    graph.add_conditional_edges(
        "weather_fastpath",
        fastpath_or_skill_match,
        {
            "skill_match": "skill_match",
            "finalize": "finalize",
        },
    )
    graph.add_conditional_edges(
        "skill_match",
        skill_execution_ready,
        {
            "skill_execute": "skill_execute",
            "semantic_recall": "semantic_recall",
            "finalize": "finalize",
        },
    )
    graph.add_conditional_edges(
        "skill_execute",
        skill_execution_or_recall,
        {
            "semantic_recall": "semantic_recall",
            "finalize": "finalize",
        },
    )
    graph.add_edge("semantic_recall", "route")
    graph.add_edge("route", "plan_agents")
    graph.add_conditional_edges(
        "plan_agents",
        select_agent_nodes,
        {
            "search_agent": "search_agent",
            "reasoning_agent": "reasoning_agent",
            "code_agent": "code_agent",
            "file_agent": "file_agent",
            "integration_agent": "integration_agent",
            "skill_agent": "skill_agent",
            "tool_agent": "tool_agent",
            "join_agents": "join_agents",
        },
    )
    for node_name in (
        "search_agent",
        "reasoning_agent",
        "code_agent",
        "file_agent",
        "integration_agent",
        "skill_agent",
        "tool_agent",
    ):
        graph.add_edge(node_name, "join_agents")
    graph.add_conditional_edges(
        "join_agents",
        dispatch_or_reflection,
        {
            "reflection": "reflection",
            "compose_answer": "compose_answer",
        },
    )
    graph.add_edge("reflection", "compose_answer")
    graph.add_edge("compose_answer", "feedback")
    graph.add_conditional_edges(
        "feedback",
        feedback_or_retry,
        {
            "retry_prepare": "retry_prepare",
            "finalize": "finalize",
        },
    )
    graph.add_edge("retry_prepare", "semantic_recall")
    graph.add_edge("finalize", END)
    return graph


checkpointer = _make_checkpointer()
graph = build_graph()
app = graph.compile(checkpointer=checkpointer)
