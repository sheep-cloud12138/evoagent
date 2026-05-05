from __future__ import annotations

import asyncio
import re

from evoagent.core.config import settings
from evoagent.core.llm import LLMClient
from evoagent.core.models import AgentOutput, TaskRequest
from evoagent.core.tool_evidence import (
    content_is_fallback,
    format_tool_evidence,
    successful_tool_events_from_metadata,
)
from evoagent.core.tools import resolve_tool_names


class BaseSubAgent:
    name = "base"
    model_profile = LLMClient.Profile.STANDARD
    enable_function_call = False
    tool_names: tuple[str, ...] = ()
    tool_categories: tuple[str, ...] = ()
    min_tool_success_rate: float | None = None

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    @staticmethod
    def _explicit_filesystem_markers() -> tuple[str, ...]:
        return (
            "当前文件夹",
            "当前目录",
            "写入文件",
            "保存到",
            "读取文件",
            "列出目录",
            "本地文件",
            "workspace",
            "working directory",
            "write file",
            "save file",
            "read file",
            "list directory",
        )

    def _tool_categories_for_task(self, task: TaskRequest) -> tuple[str, ...]:
        categories = self.tool_categories
        if "filesystem" not in categories:
            return categories

        lowered = task.query.lower()
        if any(marker in lowered for marker in self._explicit_filesystem_markers()):
            return categories
        return tuple(category for category in categories if category != "filesystem")

    def _select_tools(self, task: TaskRequest) -> tuple[str, ...]:
        threshold = (
            settings.tool_success_rate_threshold
            if self.min_tool_success_rate is None
            else max(0.0, min(self.min_tool_success_rate, 1.0))
        )
        return resolve_tool_names(
            preferred_names=self.tool_names,
            categories=self._tool_categories_for_task(task),
            min_success_rate=threshold,
        )

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
    def _filter_relevant_turns(
        cls, query: str, raw_turns: list[dict[str, object]]
    ) -> list[dict[str, object]]:
        if not raw_turns:
            return []

        windowed = raw_turns[-max(1, settings.conversation_history_window) :]
        if not cls._is_context_dependent_query(query):
            return []
        if cls._is_history_reference_query(query):
            return windowed

        query_tokens = cls._meaningful_tokens(query)
        if not query_tokens:
            return []

        selected: list[dict[str, object]] = []
        for item in windowed:
            content = str(item.get("content", "")).strip()
            if not content:
                continue
            turn_tokens = cls._meaningful_tokens(content)
            if query_tokens & turn_tokens:
                selected.append(item)

        return selected

    @staticmethod
    def _format_conversation_history(task: TaskRequest) -> str:
        raw = (
            task.context.get("conversation_history", [])
            if isinstance(task.context, dict)
            else []
        )
        if not isinstance(raw, list) or not raw:
            return ""

        max_chars = max(80, settings.conversation_max_turn_chars)
        chunks: list[str] = []
        filtered = BaseSubAgent._filter_relevant_turns(task.query, raw)
        for item in filtered:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", "user"))
            content = str(item.get("content", "")).strip().replace("\n", " ")
            if not content:
                continue
            chunks.append(f"[{role}] {content[:max_chars]}")
        if not chunks:
            return ""
        return "\n".join(chunks)

    async def run(self, task: TaskRequest, objective: str) -> AgentOutput:
        history_text = self._format_conversation_history(task)
        history_segment = f"最近对话:\n{history_text}\n" if history_text else ""
        prompt = (
            f"你是 {self.name} 子 Agent。\n"
            f"任务: {task.query}\n"
            f"{history_segment}"
            f"目标: {objective}\n"
            "请给出结构化且可执行的结果。"
        )
        metadata: dict[str, object] = {}
        if self.enable_function_call:
            selected_tools = self._select_tools(task)
            metadata["tools_selected"] = list(selected_tools)
            if selected_tools:
                prompt += (
                    "\n\n你可以调用工具。若任务涉及真实执行（如写文件、读取目录、访问网络），"
                    "必须先调用工具完成动作，再给出结论；未发生工具调用不得声称已完成执行。"
                )
                content, tool_events = await asyncio.to_thread(
                    self.llm.generate_with_tools_trace,
                    prompt,
                    selected_tools,
                    None,
                    self.model_profile,
                )
                metadata["tool_events"] = tool_events
                if content_is_fallback(content):
                    tool_evidence = format_tool_evidence(
                        successful_tool_events_from_metadata(metadata)
                    )
                    if tool_evidence:
                        content = (
                            "模型在工具执行后未能完成自然语言总结，"
                            "以下保留已成功执行的工具事实。\n\n"
                            f"{tool_evidence}"
                        )
                        metadata["llm_fallback_after_tools"] = True
                        metadata["tool_evidence_only"] = True
            else:
                content = await asyncio.to_thread(
                    self.llm.generate,
                    prompt,
                    None,
                    self.model_profile,
                )
        else:
            content = await asyncio.to_thread(
                self.llm.generate,
                prompt,
                None,
                self.model_profile,
            )
        if content_is_fallback(str(content)):
            metadata["failed"] = True
        confidence = 0.75 if metadata.get("tool_evidence_only") else 0.7
        return AgentOutput(
            agent_name=self.name,
            objective=objective,
            result=content,
            confidence=0.2 if bool(metadata.get("failed", False)) else confidence,
            metadata=metadata,
        )
