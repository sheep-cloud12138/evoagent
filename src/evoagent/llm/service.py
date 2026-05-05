from __future__ import annotations

from typing import Any

from evoagent.core.config import settings
from evoagent.core.tools import get_tool_specs_for_llm
from evoagent.llm.adapter import ProviderAdapter
from evoagent.llm.health import ModelHealthStore
from evoagent.llm.router import ModelProfile, ModelRouter
from evoagent.llm.tools import ToolCallRunner


class LLMService:
    Profile = ModelProfile

    def __init__(
        self,
        router: ModelRouter | None = None,
        adapter: ProviderAdapter | None = None,
        tool_runner: ToolCallRunner | None = None,
        health: ModelHealthStore | None = None,
    ) -> None:
        self.health = health or ModelHealthStore()
        self.router = router or ModelRouter(self.health)
        self.adapter = adapter or ProviderAdapter()
        self.tool_runner = tool_runner or ToolCallRunner(self.adapter)

    @staticmethod
    def _is_fatal_auth_error(error_text: str) -> bool:
        lowered = error_text.lower()
        markers = (
            "invalid api key",
            "incorrect api key",
            "authentication",
            "unauthorized",
            "forbidden",
            "permission denied",
            "401",
        )
        return any(token in lowered for token in markers)

    @staticmethod
    def _limit_model_attempts(candidates: list[str]) -> list[str]:
        return ModelRouter._limit_model_attempts(candidates)

    @staticmethod
    def _model_provider_family(normalized_model: str) -> str:
        return ModelRouter.provider_family(normalized_model)

    def _normalize_model(self, model: str) -> str:
        return self.router.normalize_model(model)

    def _auth_for_model(self, model: str) -> dict[str, Any]:
        return self.adapter.auth_for_model(model)

    def _has_any_credential(self) -> bool:
        return self.adapter.has_any_credential()

    def _model_candidates(self, profile: str) -> list[str]:
        return self.router._model_candidates(profile)

    def _prepare_model_candidates(self, candidates: list[str]) -> list[str]:
        return self.router._prepare_model_candidates(candidates)

    def _is_model_cooled_down(self, normalized_model: str) -> bool:
        return self.health.is_cooled_down(normalized_model)

    def _mark_model_success(self, normalized_model: str) -> None:
        self.health.record_success(normalized_model)

    def _mark_model_failure(self, normalized_model: str, error_text: str = "") -> None:
        self.health.record_failure(normalized_model, error_text)

    def _timeout_for_profile(self, profile: str) -> int:
        base = settings.llm_timeout_seconds
        if profile == self.Profile.FAST:
            return max(8, int(base * 0.5))
        if profile == self.Profile.STANDARD:
            return max(20, int(base))
        return max(30, int(base * 1.5))

    def health_report(
        self, candidates: list[str] | None = None
    ) -> list[dict[str, Any]]:
        raw_candidates = candidates or self._model_candidates(self.Profile.STANDARD)
        normalized_candidates: list[str] = []
        seen: set[str] = set()
        for model in raw_candidates:
            normalized = self._normalize_model(model)
            if normalized and normalized not in seen:
                normalized_candidates.append(normalized)
                seen.add(normalized)
        return self.health.report(normalized_candidates)

    def generate(
        self,
        prompt: str,
        temperature: float | None = None,
        profile: str = Profile.STANDARD,
        force_models: list[str] | None = None,
    ) -> str:
        temp = settings.llm_temperature_default if temperature is None else temperature
        timeout = self._timeout_for_profile(profile)
        if not self.adapter.available() or not self._has_any_credential():
            return self._fallback(prompt)

        last_error: str | None = None
        ranked_candidates = self.router.candidates_for(
            profile, {"force_models": force_models} if force_models else None
        )
        if not ranked_candidates:
            return self._fallback(
                prompt, last_error="all model candidates are cooling down"
            )

        blocked_providers: set[str] = set()
        for model in ranked_candidates:
            normalized = self._normalize_model(model)
            provider_family = self._model_provider_family(normalized)
            if provider_family in blocked_providers:
                continue
            try:
                response = self.adapter.call(
                    model=normalized,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temp,
                    timeout=timeout,
                )
                self._mark_model_success(normalized)
                return response.choices[0].message.content.strip()
            except Exception as exc:  # pragma: no cover
                last_error = str(exc)
                self._mark_model_failure(normalized, last_error)
                if self._is_fatal_auth_error(last_error):
                    blocked_providers.add(provider_family)
                continue
        return self._fallback(prompt, last_error=last_error)

    def generate_with_tools(
        self,
        prompt: str,
        tool_names: tuple[str, ...],
        temperature: float | None = None,
        profile: str = Profile.STANDARD,
    ) -> str:
        answer, _ = self.generate_with_tools_trace(
            prompt, tool_names, temperature=temperature, profile=profile
        )
        return answer

    def generate_with_tools_trace(
        self,
        prompt: str,
        tool_names: tuple[str, ...],
        temperature: float | None = None,
        profile: str = Profile.STANDARD,
    ) -> tuple[str, list[dict[str, Any]]]:
        if not settings.llm_enable_function_call or not tool_names:
            return self.generate(prompt, temperature=temperature, profile=profile), []

        temp = settings.llm_temperature_default if temperature is None else temperature
        timeout = self._timeout_for_profile(profile)
        if not self.adapter.available() or not self._has_any_credential():
            return self._fallback(prompt), []

        tools = get_tool_specs_for_llm(tool_names)
        if not tools:
            return self.generate(prompt, temperature=temperature, profile=profile), []

        last_error: str | None = None
        tool_events: list[dict[str, Any]] = []
        ranked_candidates = self.router.candidates_for(profile)
        if not ranked_candidates:
            return self._fallback(
                prompt, last_error="all model candidates are cooling down"
            ), tool_events

        blocked_providers: set[str] = set()
        for model in ranked_candidates:
            normalized = self._normalize_model(model)
            provider_family = self._model_provider_family(normalized)
            if provider_family in blocked_providers:
                continue
            try:
                answer, tool_events = self.tool_runner.run(
                    normalized,
                    [{"role": "user", "content": prompt}],
                    tools,
                    max_rounds=settings.llm_tool_max_rounds,
                    temperature=temp,
                    timeout=timeout,
                )
                self._mark_model_success(normalized)
                return (answer if answer else self._fallback(prompt)), tool_events
            except Exception as exc:  # pragma: no cover
                last_error = str(exc)
                self._mark_model_failure(normalized, last_error)
                if self._is_fatal_auth_error(last_error):
                    blocked_providers.add(provider_family)
                continue
        return self._fallback(prompt, last_error=last_error), tool_events

    def _fallback(self, prompt: str, last_error: str | None = None) -> str:
        error_hint = f" | last_error={last_error[:120]}" if last_error else ""
        reason = (
            "当前 LLM API 暂不可用或候选模型请求失败，已返回本地降级响应。"
            if self._has_any_credential()
            else "当前未配置可用 API Key，已返回本地降级响应。"
        )
        return f"[Fallback-LLM] {reason}\nPrompt 摘要: {prompt[:220]}{error_hint}"
