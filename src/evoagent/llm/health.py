from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from evoagent.core.config import settings


class ModelHealthStore:
    """Tracks transient model health and cooldown state."""

    def __init__(
        self,
        cooldown_seconds: int | None = None,
        cache_path: Path | None = None,
        persist: bool | None = None,
    ) -> None:
        self.cooldown_seconds = cooldown_seconds
        self.cache_path = cache_path or settings.llm_health_cache_path
        self.persist = settings.llm_health_cache_enabled if persist is None else persist
        self.cooldown_until: dict[str, float] = {}
        self.last_success: dict[str, float] = {}
        self.last_error: dict[str, str] = {}
        self.failure_count: dict[str, int] = {}
        self._load()

    def _cooldown_seconds(self) -> int:
        configured = (
            settings.llm_model_failure_cooldown_seconds
            if self.cooldown_seconds is None
            else self.cooldown_seconds
        )
        return max(30, int(configured))

    def _load(self) -> None:
        if not self.persist:
            return
        try:
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(payload, dict):
            return
        models = payload.get("models", {})
        if not isinstance(models, dict):
            return

        now = time.time()
        for model, item in models.items():
            if not isinstance(model, str) or not isinstance(item, dict):
                continue
            cooldown_until = self._float(item.get("cooldown_until"))
            last_success = self._float(item.get("last_success"))
            failures = self._int(item.get("failures"))
            if cooldown_until > now:
                self.cooldown_until[model] = cooldown_until
            if last_success > 0:
                self.last_success[model] = last_success
            if failures > 0:
                self.failure_count[model] = failures
            last_error = str(item.get("last_error", "")).strip()
            if last_error:
                self.last_error[model] = last_error[:240]

    @staticmethod
    def _float(value: Any) -> float:
        try:
            return float(value)
        except Exception:
            return 0.0

    @staticmethod
    def _int(value: Any) -> int:
        try:
            return int(value)
        except Exception:
            return 0

    def _write(self) -> None:
        if not self.persist:
            return
        names = (
            set(self.cooldown_until)
            | set(self.last_success)
            | set(self.last_error)
            | set(self.failure_count)
        )
        payload = {
            "updated_at": time.time(),
            "models": {
                name: {
                    "cooldown_until": float(self.cooldown_until.get(name, 0.0)),
                    "last_success": float(self.last_success.get(name, 0.0)),
                    "last_error": self.last_error.get(name, ""),
                    "failures": int(self.failure_count.get(name, 0)),
                }
                for name in sorted(names)
            },
        }
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception:
            return

    def cleanup(self) -> None:
        now = time.time()
        expired = [name for name, until in self.cooldown_until.items() if until <= now]
        for name in expired:
            self.cooldown_until.pop(name, None)
        if expired:
            self._write()

    def is_healthy(self, model: str) -> bool:
        self.cleanup()
        return self.cooldown_until.get(model, 0.0) <= time.time()

    def is_cooled_down(self, model: str) -> bool:
        return not self.is_healthy(model)

    def record_failure(self, model: str, error_text: str = "") -> None:
        self.cooldown_until[model] = time.time() + self._cooldown_seconds()
        self.last_error[model] = error_text[:240]
        self.failure_count[model] = self.failure_count.get(model, 0) + 1
        self._write()

    def record_success(self, model: str) -> None:
        self.last_success[model] = time.time()
        self.cooldown_until.pop(model, None)
        self.last_error.pop(model, None)
        self.failure_count[model] = 0
        self._write()

    def report(self, candidates: list[str]) -> list[dict[str, Any]]:
        self.cleanup()
        now = time.time()
        rows: list[dict[str, Any]] = []
        for model in candidates:
            cooldown_until = float(self.cooldown_until.get(model, 0.0))
            rows.append(
                {
                    "model": model,
                    "status": "cooled_down" if cooldown_until > now else "active",
                    "cooldown_remaining_seconds": max(
                        0, round(cooldown_until - now, 2)
                    ),
                    "last_success": float(self.last_success.get(model, 0.0)),
                    "failures": int(self.failure_count.get(model, 0)),
                    "last_error": self.last_error.get(model, ""),
                }
            )
        return rows
