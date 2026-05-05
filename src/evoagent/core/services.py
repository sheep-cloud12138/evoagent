from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import re

from evoagent.core.config import settings
from evoagent.core.feedback import FeedbackLoop
from evoagent.core.llm import LLMClient
from evoagent.core.observability import RealtimeExecutionObserver
from evoagent.core.orchestrator import SubAgentOrchestrator
from evoagent.core.router import DifficultyAssessor
from evoagent.core.runtime import RunRuntime
from evoagent.core.supervisor import AgentSupervisor
from evoagent.core.tools import register_semantic_retriever
from evoagent.core.weather import fetch_weather_report as default_fetch_weather_report
from evoagent.core.workspace import WorkspaceSandbox
from evoagent.memory.layers import MemoryManager
from evoagent.skills.evolution import SkillEvolutionEngine
from evoagent.skills.registry import SkillRegistry
from evoagent.skills.runtime import SkillArtifactManager
from evoagent.sandbox import get_sandbox


class EvoAgentServices:
    """Shared service bundle used by the LangGraph runtime nodes."""

    def __init__(self) -> None:
        self.llm = LLMClient()
        self.router = DifficultyAssessor(self.llm)
        self.observer = RealtimeExecutionObserver(
            enabled=settings.observability_enabled,
            event_log_path=settings.observability_log_path,
            metrics_path=settings.observability_metrics_path,
            emit_stdout=settings.observability_stdout,
        )
        self.orchestrator = SubAgentOrchestrator(
            llm=self.llm,
            max_parallel=settings.max_parallel_subagents,
            observer=self.observer,
        )
        self.supervisor = AgentSupervisor(self.orchestrator)
        self.runtime = RunRuntime()
        self.workspace_sandbox = WorkspaceSandbox(settings.workspace_store_path)
        self.memory = MemoryManager(
            db_path=settings.db_path,
            semantic_store_path=settings.semantic_store_path,
        )
        register_semantic_retriever(self._semantic_retriever)
        self.skill_registry = SkillRegistry(settings.db_path)
        self.skill_artifacts = SkillArtifactManager(
            settings.skill_store_path, llm=self.llm
        )
        self.skill_artifacts.register_active_skill_tools()
        self.skill_engine = SkillEvolutionEngine(
            llm=self.llm,
            registry=self.skill_registry,
            sandbox=get_sandbox(),
            artifacts=self.skill_artifacts,
        )
        self.feedback = FeedbackLoop(
            router=self.router,
            memory=self.memory,
            evolution=self.skill_engine,
        )

    def _semantic_retriever(
        self, query: str, top_k: int, use_negative: bool
    ) -> list[dict]:
        if use_negative:
            return self.memory.recall_negative(query, top_k=top_k)
        return self.memory.recall_semantic(query, top_k=top_k)

    @staticmethod
    def _is_time_query(query: str) -> bool:
        compact = "".join(query.strip().lower().split())
        if not compact or len(compact) > 24:
            return False

        cn_markers = ("现在几点", "几点了", "当前时间", "现在时间")
        en_markers = ("whattimeisit", "currenttime", "time?")
        if compact in {"几点", "time", "whattimeisit"}:
            return True
        return any(marker in compact for marker in (*cn_markers, *en_markers))

    @staticmethod
    def _format_local_time_answer() -> str:
        now = datetime.now().astimezone()
        offset = now.utcoffset() or timedelta(0)
        total_seconds = int(offset.total_seconds())
        sign = "+" if total_seconds >= 0 else "-"
        abs_seconds = abs(total_seconds)
        hours = abs_seconds // 3600
        minutes = (abs_seconds % 3600) // 60
        offset_text = f"UTC{sign}{hours:02d}:{minutes:02d}"
        tz_name = now.tzname() or "Local"
        formatted = now.strftime("%Y-%m-%d %H:%M:%S")
        return f"当前时间：{formatted}（{tz_name}, {offset_text}）"

    @staticmethod
    def _file_output_requested(query: str) -> bool:
        lowered = query.lower()
        markers = (
            "写个文档",
            "写一个文档",
            "写文档",
            "生成文档",
            "保存文档",
            "写到当前文件夹",
            "写在当前文件夹",
            "保存到当前文件夹",
            "保存至当前文件夹",
            "当前文件夹",
            "当前目录",
            "write a document",
            "write document",
            "save to current folder",
            "save in current folder",
            "save file",
            "write file",
        )
        return any(marker in lowered for marker in markers)

    @staticmethod
    def _extract_requested_filename(query: str, final_answer: str) -> str:
        extensions = "md|txt|json|csv|html|py|js|ts|java|cpp|c|go|rs|sql|yaml|yml"
        patterns = (
            rf"`([^`\n]+\.(?:{extensions}))`",
            rf"(?:保存为|命名为|文件名为|file name is|save as)\s*[:：]?\s*([^\s`，。；;]+\.(?:{extensions}))",
        )
        for source in (final_answer, query):
            for pattern in patterns:
                match = re.search(pattern, source, flags=re.IGNORECASE)
                if match:
                    return EvoAgentServices._sanitize_output_filename(match.group(1))

        base = re.sub(r"\s+", "", query.strip())
        base = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", base)
        base = base.strip("_")[:36] or "evoagent_output"
        return EvoAgentServices._sanitize_output_filename(f"{base}.md")

    @staticmethod
    def _sanitize_output_filename(name: str) -> str:
        filename = Path(name.strip()).name
        filename = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", filename).strip(" ._")
        if not filename:
            filename = "evoagent_output.md"
        if "." not in filename:
            filename = f"{filename}.md"
        if filename.startswith("."):
            filename = f"output{filename}"
        return filename

    @staticmethod
    def _extract_document_content(final_answer: str) -> str:
        for pattern in (
            r"```(?:markdown|md)\s*\n(.*?)\n```",
            r"```\s*\n(.*?)\n```",
        ):
            match = re.search(pattern, final_answer, flags=re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip() + "\n"
        return final_answer.strip() + "\n"

    @classmethod
    def _maybe_write_requested_file(
        cls, query: str, final_answer: str
    ) -> dict[str, object] | None:
        if not cls._file_output_requested(query):
            return None

        filename = cls._extract_requested_filename(query, final_answer)
        target_dir = Path.cwd().resolve()
        target = (target_dir / filename).resolve()
        try:
            target.relative_to(target_dir)
        except ValueError:
            target = target_dir / "evoagent_output.md"

        if target.exists():
            stem = target.stem
            suffix = target.suffix or ".md"
            for idx in range(2, 1000):
                candidate = target_dir / f"{stem}_{idx}{suffix}"
                if not candidate.exists():
                    target = candidate
                    break

        content = cls._extract_document_content(final_answer)
        target.write_text(content, encoding="utf-8")
        return {
            "path": str(target),
            "filename": target.name,
            "bytes": len(content.encode("utf-8")),
        }

    def fetch_weather_report(self, location_text: str) -> str:
        return default_fetch_weather_report(location_text)
