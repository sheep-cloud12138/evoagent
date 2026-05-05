from __future__ import annotations

import asyncio
import random
import re
import time

from evoagent.agents.base import BaseSubAgent
from evoagent.agents.specialists import (
    CodeAgent,
    FileAgent,
    IntegrationAgent,
    ReasoningAgent,
    SearchAgent,
    SkillAgent,
)
from evoagent.core.config import settings
from evoagent.core.llm import LLMClient
from evoagent.core.models import (
    AgentOutput,
    Difficulty,
    TaskCapability,
    TaskDecision,
    TaskRequest,
)
from evoagent.core.observability import RealtimeExecutionObserver
from evoagent.core.tool_evidence import (
    answer_conflicts_with_successful_tools,
    format_tool_evidence,
    successful_tool_events_from_outputs,
)


class SubAgentOrchestrator:
    def __init__(
        self,
        llm: LLMClient,
        max_parallel: int = 4,
        observer: RealtimeExecutionObserver | None = None,
    ) -> None:
        self.llm = llm
        self.max_parallel = max_parallel
        self.observer = observer

    @staticmethod
    def _history_reference_markers() -> tuple[str, ...]:
        return (
            "刚才",
            "之前",
            "上次",
            "前面",
            "还记得",
            "你记得",
            "我叫什么",
            "我刚才",
            "earlier",
            "previous",
            "last time",
            "remember",
        )

    @staticmethod
    def _context_dependency_markers() -> tuple[str, ...]:
        return (
            "这个",
            "那个",
            "它",
            "上述",
            "上面",
            "继续",
            "接着",
            "再",
            "改成",
            "基于上面",
            "在此基础上",
            "this",
            "that",
            "it",
            "above",
            "continue",
            "again",
            "based on",
        )

    @classmethod
    def _is_history_reference_query(cls, query: str) -> bool:
        lowered = query.lower()
        return any(marker in lowered for marker in cls._history_reference_markers())

    @classmethod
    def _is_context_dependent_query(cls, query: str) -> bool:
        lowered = query.lower()
        if cls._is_history_reference_query(query):
            return True
        return any(marker in lowered for marker in cls._context_dependency_markers())

    @staticmethod
    def _code_generation_markers() -> tuple[str, ...]:
        return (
            "写一个",
            "代码",
            "程序",
            "脚本",
            "爬虫",
            "函数",
            "java",
            "python",
            "javascript",
            "typescript",
            "write code",
            "script",
            "crawler",
            "function",
        )

    @classmethod
    def _is_code_generation_query(cls, query: str) -> bool:
        lowered = query.lower()
        if any(marker in lowered for marker in cls._code_generation_markers()):
            return True
        implementation_markers = ("实现", "implement")
        algorithm_markers = (
            "排序",
            "算法",
            "比较器",
            "方法",
            "class",
            "method",
            "algorithm",
            "comparator",
        )
        return any(marker in lowered for marker in implementation_markers) and any(
            marker in lowered for marker in algorithm_markers
        )

    @staticmethod
    def _meaningful_tokens(text: str) -> set[str]:
        lowered = text.lower()
        en_tokens = {
            tok
            for tok in re.findall(r"[a-z0-9_]+", lowered)
            if tok
            not in {
                "the",
                "a",
                "an",
                "to",
                "for",
                "and",
                "or",
                "is",
                "are",
                "of",
                "in",
                "on",
            }
        }

        zh_stop = {
            "我",
            "你",
            "他",
            "她",
            "它",
            "的",
            "了",
            "吗",
            "呢",
            "是",
            "在",
            "有",
            "和",
            "与",
            "及",
            "把",
            "请",
            "给",
            "个",
            "这",
            "那",
            "就",
            "还",
        }
        zh_tokens = {
            ch for ch in lowered if "\u4e00" <= ch <= "\u9fff" and ch not in zh_stop
        }
        return en_tokens | zh_tokens

    @classmethod
    def _format_conversation_history(cls, task: TaskRequest) -> str:
        raw = (
            task.context.get("conversation_history", [])
            if isinstance(task.context, dict)
            else []
        )
        if not isinstance(raw, list) or not raw:
            return ""

        windowed = raw[-max(1, settings.conversation_history_window) :]
        if not cls._is_context_dependent_query(task.query):
            return ""
        if cls._is_history_reference_query(task.query):
            selected = windowed
        else:
            query_tokens = cls._meaningful_tokens(task.query)
            selected: list[dict[str, object]] = []
            for item in windowed:
                if not isinstance(item, dict):
                    continue
                content = str(item.get("content", "")).strip()
                if not content:
                    continue
                if query_tokens and (query_tokens & cls._meaningful_tokens(content)):
                    selected.append(item)

        lines: list[str] = []
        for item in selected:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", "user"))
            content = str(item.get("content", "")).strip().replace("\n", " ")
            if content:
                lines.append(f"[{role}] {content[:220]}")
        return "\n".join(lines)

    def _spawn_plan(
        self, task: TaskRequest, decision: TaskDecision
    ) -> list[tuple[BaseSubAgent, str]]:
        capabilities = set(decision.features.required_capabilities)
        if self._is_code_generation_query(task.query):
            capabilities.add(TaskCapability.CODING)
        if decision.features.needs_external_tools and not capabilities:
            capabilities.add(TaskCapability.WEB)
        if not capabilities:
            capabilities.add(TaskCapability.PLANNING)

        plan: list[tuple[BaseSubAgent, str]] = []
        added: set[str] = set()

        def add(agent: BaseSubAgent, objective: str) -> None:
            if agent.name in added:
                return
            plan.append((agent, objective))
            added.add(agent.name)

        if capabilities & {TaskCapability.RETRIEVAL, TaskCapability.WEB}:
            add(SearchAgent(self.llm), "检索语义记忆、网页信息与相似经验，提取可信依据")

        if TaskCapability.FILESYSTEM in capabilities:
            add(
                FileAgent(self.llm),
                "执行受限的本地文件读写或目录操作，并返回真实执行结果",
            )

        if TaskCapability.INTEGRATION in capabilities:
            add(
                IntegrationAgent(self.llm),
                "调用外部集成、MCP 或网络工具，验证依赖与接口路径",
            )

        if TaskCapability.SKILL in capabilities:
            add(SkillAgent(self.llm), "匹配并调用可复用 Skill，评估是否能复用已有能力")

        if TaskCapability.CODING in capabilities:
            add(CodeAgent(self.llm), "给出可落地实现、关键代码、错误处理与测试策略")

        needs_synthesis = (
            TaskCapability.PLANNING in capabilities
            or not plan
            or decision.difficulty == Difficulty.COMPLEX
        )
        if needs_synthesis and TaskCapability.CODING not in capabilities:
            add(
                ReasoningAgent(self.llm),
                "综合上下文与工具结果，形成可执行结论、步骤、假设与风险",
            )

        return plan

    async def execute(
        self, task: TaskRequest, decision: TaskDecision
    ) -> list[AgentOutput]:
        plan = self._spawn_plan(task, decision)
        semaphore = asyncio.Semaphore(self.max_parallel)
        retries = max(0, settings.subagent_retry_attempts)
        backoff = max(0.1, settings.subagent_retry_backoff_seconds)
        global_timeout = max(0.1, float(settings.orchestrator_task_timeout_seconds))

        async def _run(agent: BaseSubAgent, objective: str) -> AgentOutput:
            start = time.perf_counter()
            if self.observer is not None:
                self.observer.on_subagent_start(
                    task_id=task.task_id, agent_name=agent.name, objective=objective
                )

            async with semaphore:
                last_error: str | None = None
                for attempt in range(retries + 1):
                    try:
                        output = await agent.run(task, objective)
                        elapsed_ms = (time.perf_counter() - start) * 1000
                        tool_calls = (
                            len(output.metadata.get("tool_events", []))
                            if isinstance(output.metadata, dict)
                            else 0
                        )
                        if self.observer is not None:
                            self.observer.on_subagent_end(
                                task_id=task.task_id,
                                agent_name=agent.name,
                                elapsed_ms=elapsed_ms,
                                tool_calls=tool_calls,
                                success=not bool(output.metadata.get("failed", False)),
                                timed_out=False,
                            )
                        return output
                    except Exception as exc:
                        last_error = str(exc)
                        if self.observer is not None and attempt < retries:
                            self.observer.on_subagent_retry(
                                task_id=task.task_id,
                                agent_name=agent.name,
                                attempt=attempt + 1,
                                error=last_error,
                            )
                        if attempt < retries:
                            await asyncio.sleep(
                                backoff * (2**attempt) + random.uniform(0, 0.2)
                            )
                fallback = AgentOutput(
                    agent_name=agent.name,
                    objective=objective,
                    result=f"[subagent-error] {last_error or 'unknown error'}",
                    confidence=0.2,
                    metadata={"retries": retries, "failed": True},
                )
                elapsed_ms = (time.perf_counter() - start) * 1000
                if self.observer is not None:
                    self.observer.on_subagent_end(
                        task_id=task.task_id,
                        agent_name=agent.name,
                        elapsed_ms=elapsed_ms,
                        tool_calls=0,
                        success=False,
                        timed_out=False,
                    )
                return fallback

        scheduled: list[
            tuple[int, BaseSubAgent, str, asyncio.Task[AgentOutput], float]
        ] = []
        for idx, (agent, objective) in enumerate(plan):
            started_at = time.perf_counter()
            scheduled.append(
                (
                    idx,
                    agent,
                    objective,
                    asyncio.create_task(_run(agent, objective)),
                    started_at,
                )
            )

        done, pending = await asyncio.wait(
            [entry[3] for entry in scheduled],
            timeout=global_timeout,
            return_when=asyncio.ALL_COMPLETED,
        )

        outputs_by_idx: dict[int, AgentOutput] = {}
        task_to_entry = {entry[3]: entry for entry in scheduled}

        for done_task in done:
            idx, agent, objective, _, _ = task_to_entry[done_task]
            try:
                outputs_by_idx[idx] = done_task.result()
            except Exception as exc:
                outputs_by_idx[idx] = AgentOutput(
                    agent_name=agent.name,
                    objective=objective,
                    result=f"[subagent-error] {str(exc)[:220]}",
                    confidence=0.2,
                    metadata={"failed": True},
                )

        if pending:
            for pending_task in pending:
                idx, agent, objective, _, started_at = task_to_entry[pending_task]
                pending_task.cancel()
                elapsed_ms = (time.perf_counter() - started_at) * 1000
                timeout_output = AgentOutput(
                    agent_name=agent.name,
                    objective=objective,
                    result="[subagent-timeout] 任务级超时触发，使用部分结果继续聚合。",
                    confidence=0.15,
                    metadata={"failed": True, "timed_out": True},
                )
                outputs_by_idx[idx] = timeout_output
                if self.observer is not None:
                    self.observer.on_subagent_end(
                        task_id=task.task_id,
                        agent_name=agent.name,
                        elapsed_ms=elapsed_ms,
                        tool_calls=0,
                        success=False,
                        timed_out=True,
                    )

            await asyncio.gather(*pending, return_exceptions=True)

        return [outputs_by_idx[i] for i in range(len(plan)) if i in outputs_by_idx]

    def aggregate(self, task: TaskRequest, outputs: list[AgentOutput]) -> str:
        # Removed filesystem-only deterministic gate to avoid scenario-specific
        # branches; aggregation now uses one generic merge path for all tasks.

        tool_events = successful_tool_events_from_outputs(outputs)
        tool_evidence = format_tool_evidence(tool_events)
        tool_segment = (
            f"工具事实:\n{tool_evidence}\n\n"
            if tool_evidence
            else ""
        )
        merged = "\n\n".join(
            [f"[{o.agent_name}] objective={o.objective}\n{o.result}" for o in outputs]
        )
        history = self._format_conversation_history(task)
        history_segment = f"最近对话:\n{history}\n" if history else ""
        prompt = (
            "你是核心 Agent，负责冲突消解与最终整合。\n"
            f"用户任务: {task.query}\n"
            f"{history_segment}"
            f"{tool_segment}"
            f"子 Agent 输出:\n{merged}\n\n"
            "请输出最终答案，要求: 结论明确、步骤可执行、说明假设与风险。"
            "如果工具事实和子 Agent 自述冲突，必须以工具事实为准；"
            "不要声称无法访问已经由工具成功读取或调用的内容。"
        )
        answer = self.llm.generate(
            prompt,
            temperature=0.05,
            profile=LLMClient.Profile.REASONING,
            force_models=[
                settings.main_agent_model,
                settings.llm_reasoning_model,
                settings.llm_model,
            ],
        )
        if answer.startswith("[Fallback-LLM]") or answer_conflicts_with_successful_tools(
            answer, tool_events
        ):
            return self._deterministic_aggregate(task, outputs)
        return answer

    def maybe_reflect(
        self,
        task: TaskRequest,
        decision: TaskDecision,
        outputs: list[AgentOutput],
        draft_answer: str,
    ) -> tuple[str, bool, str]:
        _ = task, decision, outputs
        return draft_answer, False, "handled_by_langgraph"

    def _deterministic_aggregate(
        self, task: TaskRequest, outputs: list[AgentOutput]
    ) -> str:
        useful = [o for o in outputs if not o.result.startswith("[Fallback-LLM]")]
        picked = useful if useful else outputs
        head = picked[0].result[:700] if picked else ""
        tool_evidence = format_tool_evidence(
            successful_tool_events_from_outputs(outputs),
            max_event_chars=1200,
            max_total_chars=5000,
        )
        evidence_section = (
            f"\n工具事实摘录:\n{tool_evidence}\n"
            if tool_evidence
            else ""
        )
        return (
            "最终整合结果（确定性兜底）\n"
            f"任务: {task.query}\n"
            "结论: 已基于可用子 Agent 结果完成聚合。\n"
            f"{evidence_section}"
            "执行建议:\n"
            "1. 先按下述方案的短期步骤落地。\n"
            "2. 对关键指标建立监控并滚动优化。\n"
            "3. 为超时子任务开启异步重试。\n\n"
            f"核心内容摘录:\n{head}"
        )
