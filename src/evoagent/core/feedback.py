from __future__ import annotations

from typing import Any

from evoagent.core.models import AgentOutput, TaskDecision
from evoagent.core.router import DifficultyAssessor
from evoagent.core.config import settings
from evoagent.core.tools import get_tool_stats
from evoagent.memory.layers import MemoryManager
from evoagent.skills.evolution import SkillEvolutionEngine


class FeedbackLoop:
    def __init__(
        self,
        router: DifficultyAssessor,
        memory: MemoryManager,
        evolution: SkillEvolutionEngine,
    ) -> None:
        self.router = router
        self.memory = memory
        self.evolution = evolution

    def score_result(
        self, correctness: float, efficiency: float, user_satisfaction: float
    ) -> float:
        return max(
            0.0,
            min(
                (0.5 * correctness + 0.25 * efficiency + 0.25 * user_satisfaction), 1.0
            ),
        )

    @staticmethod
    def _collect_tool_events(outputs: list[AgentOutput]) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for out in outputs:
            payload = (
                out.metadata.get("tool_events", [])
                if isinstance(out.metadata, dict)
                else []
            )
            if isinstance(payload, list):
                for item in payload:
                    if isinstance(item, dict):
                        events.append(item)
        return events

    def _classify_failure(
        self,
        final_answer: str,
        decision: TaskDecision | None,
        semantic_hints: list[dict[str, Any]],
        tool_events: list[dict[str, Any]],
        trace: list[dict[str, Any]],
    ) -> str:
        if any(str(step.get("step", "")) == "skill_runtime_error" for step in trace):
            return "skill_bug"

        if any(not bool(evt.get("success", False)) for evt in tool_events):
            return "tool_failure"

        if decision is not None:
            low_conf = decision.confidence < settings.router_low_confidence_threshold
            under_allocated = (
                decision.features.needs_external_tools
                and decision.difficulty.value == "simple"
            )
            if decision.escalated_due_to_low_confidence or low_conf or under_allocated:
                return "routing_error"

        if not semantic_hints:
            return "memory_miss"

        if (
            "[Fallback-LLM]" in final_answer
            or "无法直接完成" in final_answer
            or "无可用降级方案" in final_answer
        ):
            return "llm_reasoning_failure"

        return "llm_reasoning_failure"

    @staticmethod
    def _is_low_value_fallback(
        final_answer: str,
        quality_score: float,
        semantic_hints: list[dict[str, Any]],
        tool_events: list[dict[str, Any]],
    ) -> bool:
        markers = (
            "[Fallback-LLM]",
            "无法直接完成",
            "无法完成该任务",
            "无可用降级方案",
            "确定性兜底",
        )
        if not any(marker in final_answer for marker in markers):
            return False

        short_answer = len(final_answer.strip()) <= max(
            60, settings.skill_low_value_fallback_max_chars
        )
        low_signal = not semantic_hints and not tool_events
        low_quality = quality_score < settings.skill_low_value_fallback_min_quality
        return short_answer and (low_signal or low_quality)

    def run_post_task(
        self,
        task_id: str,
        task_text: str,
        final_answer: str,
        quality_score: float,
        path_length: int,
        confidence: float,
        decision: TaskDecision | None = None,
        outputs: list[AgentOutput] | None = None,
        semantic_hints: list[dict[str, Any]] | None = None,
        trace: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        outputs = outputs or []
        semantic_hints = semantic_hints or []
        trace = trace or []

        failure_markers = [
            "无法直接完成",
            "无法完成该任务",
            "无可用降级方案",
            "[Fallback-LLM]",
            "Timeout",
            "超时",
            "status",
            "failed",
            "task is fundamentally infeasible",
        ]
        has_failure_marker = any(marker in final_answer for marker in failure_markers)
        tool_events = self._collect_tool_events(outputs)
        has_subagent_failure = any(
            bool(o.metadata.get("failed", False)) for o in outputs
        )
        success = (
            quality_score >= 0.72
            and not has_failure_marker
            and not has_subagent_failure
        )
        failure_class = (
            "none"
            if success
            else self._classify_failure(
                final_answer=final_answer,
                decision=decision,
                semantic_hints=semantic_hints,
                tool_events=tool_events,
                trace=trace,
            )
        )

        summary = f"Task={task_text[:80]} | score={quality_score:.2f} | answer={final_answer[:120]}"
        self.memory.episodic.add(
            task_id=task_id, summary=summary, success=success, score=quality_score
        )
        self.memory.distill_episode_to_semantic(
            task_id=task_id,
            summary=summary,
            score=quality_score,
            success=success,
            failure_reason=None if success else failure_class,
        )

        actions: dict[str, Any] = {
            "memory": "updated",
            "failure_class": failure_class,
            "tool_observability": {
                "events": tool_events,
                "per_tool": get_tool_stats(),
            },
        }

        provider_failure_markers = (
            "[Fallback-LLM]",
            "[subagent-timeout]",
            "任务级超时",
            "all model candidates",
            "APITimeoutError",
            "Connection timed out",
            "litellm.Timeout",
        )
        provider_failure = any(
            marker in final_answer for marker in provider_failure_markers
        )

        if provider_failure:
            actions["skill_evolution"] = "skipped_provider_failure"
        elif self._is_low_value_fallback(
            final_answer=final_answer,
            quality_score=quality_score,
            semantic_hints=semantic_hints,
            tool_events=tool_events,
        ):
            actions["skill_evolution"] = "skipped_low_value_fallback"
        elif (
            self.evolution.identify_gap(quality_score, path_length, confidence)
            or has_failure_marker
        ):
            failure_reason = (
                "explicit_failure_marker"
                if has_failure_marker
                else (
                    failure_class
                    if failure_class != "none"
                    else "low_score_or_high_complexity"
                )
            )
            draft = self.evolution.generate_skill(
                task_text, failure_reason=failure_reason
            )
            ok, msg = self.evolution.validate_and_register(draft)
            actions["skill_evolution"] = "registered" if ok else f"failed:{msg}"
            if ok:
                artifacts = getattr(self.evolution, "artifacts", None)
                if artifacts is not None and hasattr(
                    artifacts, "register_active_skill_tools"
                ):
                    tool_names = artifacts.register_active_skill_tools()
                    actions["skill_tools_registered"] = tool_names
        else:
            actions["skill_evolution"] = "not_triggered"

        if success:
            self.router.update_weights(-0.05)
            actions["router"] = "weights_updated_success"
        elif failure_class == "routing_error":
            self.router.update_weights(0.15)
            actions["router"] = "weights_updated_routing_failure"
        else:
            actions["router"] = "skipped_non_routing_failure"

        archived = self.evolution.registry.decay_and_archive(min_calls=3, threshold=0.2)
        actions["skill_archive"] = f"archived={archived}"

        return actions
