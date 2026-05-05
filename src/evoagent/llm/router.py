from __future__ import annotations

from enum import Enum

from evoagent.core.config import settings
from evoagent.llm.health import ModelHealthStore


class ModelProfile(str, Enum):
    FAST = "fast"
    STANDARD = "standard"
    REASONING = "reasoning"


class ModelRouter:
    def __init__(self, health: ModelHealthStore | None = None) -> None:
        self.health = health or ModelHealthStore()

    def candidates_for(
        self,
        task_type: str | ModelProfile = ModelProfile.STANDARD,
        constraints: dict | None = None,
    ) -> list[str]:
        force_models = (constraints or {}).get("force_models")
        raw = (
            list(force_models)
            if force_models
            else self._model_candidates(str(task_type))
        )
        return self._limit_model_attempts(self._prepare_model_candidates(raw))

    @staticmethod
    def provider_family(model: str) -> str:
        normalized_model = ModelRouter.normalize_model(model)
        lowered = normalized_model.strip().lower()
        if lowered.startswith("deepseek/"):
            return "deepseek"
        if lowered.startswith("volcengine/") or lowered.startswith("ark/"):
            return "volcengine"
        if lowered.startswith("openai/qwen") or lowered.startswith("qwen/"):
            return "qwen"
        if lowered.startswith("openrouter/"):
            return "openrouter"
        if lowered.startswith("anthropic/"):
            return "anthropic"
        if lowered.startswith("gemini/"):
            return "gemini"
        if (
            lowered.startswith("openai/")
            or lowered.startswith("gpt")
            or lowered.startswith("o")
        ):
            return "openai"
        if "/" in lowered:
            return lowered.split("/", 1)[0]
        return "unknown"

    @staticmethod
    def normalize_model(model: str) -> str:
        m = model.strip()
        if "/" in m:
            if m.startswith("ark/"):
                return f"volcengine/{m.split('/', 1)[1]}"
            return m
        if m.startswith(("ep-", "ep_")):
            return f"volcengine/{m}"
        if m.startswith("deepseek"):
            return f"deepseek/{m}"
        if m.startswith("qwen"):
            return f"openai/{m}"
        if m.startswith("claude"):
            return f"anthropic/{m}"
        if m.startswith("gemini"):
            return f"gemini/{m}"
        if m.startswith("gpt") or m.startswith("o"):
            return m

        provider = settings.llm_provider.strip().lower()
        if provider == "deepseek":
            return f"deepseek/{m}"
        if provider in {"volcengine", "ark"}:
            return f"volcengine/{m}"
        if provider == "qwen":
            return f"openai/{m}"
        if provider == "anthropic":
            return f"anthropic/{m}"
        if provider == "google":
            return f"gemini/{m}"
        if provider == "openrouter":
            return f"openrouter/{m}"
        return m

    def _model_candidates(self, profile: str) -> list[str]:
        if profile == ModelProfile.FAST:
            primary = settings.llm_fast_model
        elif profile == ModelProfile.REASONING:
            primary = settings.llm_reasoning_model
        else:
            primary = settings.llm_standard_model or settings.llm_model

        backups = [
            m.strip() for m in settings.llm_backup_models.split(",") if m.strip()
        ]
        candidates = [primary, settings.llm_model, *backups]
        deduped: list[str] = []
        seen: set[str] = set()
        for model in candidates:
            if model and model not in seen:
                deduped.append(model)
                seen.add(model)
        return deduped

    def _prepare_model_candidates(self, candidates: list[str]) -> list[str]:
        rows: list[tuple[int, str, str]] = []
        seen: set[str] = set()
        for idx, model in enumerate(candidates):
            normalized = self.normalize_model(model)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            rows.append((idx, model, normalized))

        rows.sort(
            key=lambda item: (-self.health.last_success.get(item[2], 0.0), item[0])
        )
        if not rows:
            return []

        active = [item for item in rows if self.health.is_healthy(item[2])]
        if active:
            return [item[1] for item in active]

        rows.sort(
            key=lambda item: (self.health.cooldown_until.get(item[2], 0.0), item[0])
        )
        return [rows[0][1]]

    @staticmethod
    def _limit_model_attempts(candidates: list[str]) -> list[str]:
        max_attempts = max(1, int(settings.llm_max_candidate_attempts))
        return candidates[:max_attempts]
