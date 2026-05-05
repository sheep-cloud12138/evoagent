from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class RealtimeExecutionObserver:
    def __init__(
        self,
        enabled: bool,
        event_log_path: Path,
        metrics_path: Path,
        emit_stdout: bool = False,
    ) -> None:
        self.enabled = enabled
        self.emit_stdout = emit_stdout
        self.event_log_path = event_log_path
        self.metrics_path = metrics_path

        if not self.enabled:
            self.metrics: dict[str, Any] = self._default_metrics()
            return

        self.event_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.event_log_path.exists():
            self.event_log_path.write_text("", encoding="utf-8")
        if not self.metrics_path.exists():
            self.metrics_path.write_text(
                json.dumps(self._default_metrics(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        self.metrics = self._read_metrics()

    @staticmethod
    def _default_metrics() -> dict[str, Any]:
        return {
            "tasks": {
                "total": 0,
                "timed_out": 0,
                "by_difficulty": {},
                "task_profiles": {},
            },
            "agents": {},
        }

    def _read_metrics(self) -> dict[str, Any]:
        try:
            data = json.loads(self.metrics_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except Exception:
            pass
        return self._default_metrics()

    def _write_metrics(self) -> None:
        if not self.enabled:
            return
        self.metrics_path.write_text(
            json.dumps(self.metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def emit(self, event_type: str, **payload: Any) -> None:
        if not self.enabled:
            return
        event = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "event": event_type,
            **payload,
        }
        line = json.dumps(event, ensure_ascii=False)
        with self.event_log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
        if self.emit_stdout:
            print(line)

    def on_task_start(self, task_id: str, query: str) -> None:
        self.emit("task_start", task_id=task_id, query_preview=query[:160])

    def on_task_decision(
        self, task_id: str, difficulty: str, decision_confidence: float
    ) -> None:
        self.emit(
            "task_decision",
            task_id=task_id,
            difficulty=difficulty,
            decision_confidence=decision_confidence,
        )

    def on_task_end(
        self,
        task_id: str,
        elapsed_ms: float,
        difficulty: str,
        task_profile: str,
        timed_out: bool,
    ) -> None:
        self.emit(
            "task_end",
            task_id=task_id,
            elapsed_ms=round(elapsed_ms, 3),
            difficulty=difficulty,
            task_profile=task_profile,
            timed_out=timed_out,
        )

        tasks = self.metrics.setdefault("tasks", {})
        tasks["total"] = int(tasks.get("total", 0)) + 1
        if timed_out:
            tasks["timed_out"] = int(tasks.get("timed_out", 0)) + 1

        by_diff = tasks.setdefault("by_difficulty", {})
        diff_stat = by_diff.setdefault(
            difficulty, {"count": 0, "total_elapsed_ms": 0.0, "avg_elapsed_ms": 0.0}
        )
        diff_stat["count"] = int(diff_stat.get("count", 0)) + 1
        diff_stat["total_elapsed_ms"] = float(
            diff_stat.get("total_elapsed_ms", 0.0)
        ) + float(elapsed_ms)
        diff_stat["avg_elapsed_ms"] = diff_stat["total_elapsed_ms"] / max(
            diff_stat["count"], 1
        )

        by_profile = tasks.setdefault("task_profiles", {})
        profile_stat = by_profile.setdefault(
            task_profile, {"count": 0, "total_elapsed_ms": 0.0, "avg_elapsed_ms": 0.0}
        )
        profile_stat["count"] = int(profile_stat.get("count", 0)) + 1
        profile_stat["total_elapsed_ms"] = float(
            profile_stat.get("total_elapsed_ms", 0.0)
        ) + float(elapsed_ms)
        profile_stat["avg_elapsed_ms"] = profile_stat["total_elapsed_ms"] / max(
            profile_stat["count"], 1
        )

        self._write_metrics()

    def on_subagent_start(self, task_id: str, agent_name: str, objective: str) -> None:
        self.emit(
            "subagent_start",
            task_id=task_id,
            agent_name=agent_name,
            objective=objective,
        )

    def on_subagent_retry(
        self, task_id: str, agent_name: str, attempt: int, error: str
    ) -> None:
        self.emit(
            "subagent_retry",
            task_id=task_id,
            agent_name=agent_name,
            attempt=attempt,
            error=error[:220],
        )

    def on_subagent_end(
        self,
        task_id: str,
        agent_name: str,
        elapsed_ms: float,
        tool_calls: int,
        success: bool,
        timed_out: bool,
    ) -> None:
        self.emit(
            "subagent_end",
            task_id=task_id,
            agent_name=agent_name,
            elapsed_ms=round(elapsed_ms, 3),
            tool_calls=tool_calls,
            success=success,
            timed_out=timed_out,
        )

        agents = self.metrics.setdefault("agents", {})
        stat = agents.setdefault(
            agent_name,
            {
                "runs": 0,
                "failures": 0,
                "timeouts": 0,
                "total_elapsed_ms": 0.0,
                "avg_elapsed_ms": 0.0,
                "max_elapsed_ms": 0.0,
                "tool_calls_total": 0,
            },
        )
        stat["runs"] = int(stat.get("runs", 0)) + 1
        stat["total_elapsed_ms"] = float(stat.get("total_elapsed_ms", 0.0)) + float(
            elapsed_ms
        )
        stat["avg_elapsed_ms"] = stat["total_elapsed_ms"] / max(stat["runs"], 1)
        stat["max_elapsed_ms"] = max(
            float(stat.get("max_elapsed_ms", 0.0)), float(elapsed_ms)
        )
        stat["tool_calls_total"] = int(stat.get("tool_calls_total", 0)) + int(
            tool_calls
        )
        if not success:
            stat["failures"] = int(stat.get("failures", 0)) + 1
        if timed_out:
            stat["timeouts"] = int(stat.get("timeouts", 0)) + 1

        self._write_metrics()
