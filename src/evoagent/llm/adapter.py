from __future__ import annotations

import sys
from typing import Any

from evoagent.core.config import settings

try:
    import litellm
    from litellm import completion

    if settings.llm_disable_provider_noise:
        if hasattr(litellm, "set_verbose"):
            litellm.set_verbose = False
        if hasattr(litellm, "suppress_debug_info"):
            litellm.suppress_debug_info = True
        if hasattr(litellm, "telemetry"):
            litellm.telemetry = False
except Exception:  # pragma: no cover
    litellm = None
    completion = None


class ProviderAdapter:
    def completion_callable(self):
        compat = sys.modules.get("evoagent.core.llm")
        compat_completion = (
            getattr(compat, "completion", None) if compat is not None else None
        )
        return compat_completion or completion

    def available(self) -> bool:
        return self.completion_callable() is not None

    def call(self, model: str, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        completion_fn = self.completion_callable()
        if completion_fn is None:
            raise RuntimeError("LiteLLM is not available")
        auth = self.auth_for_model(model)
        api_key = auth.get("api_key")
        if not api_key:
            raise RuntimeError(f"missing_api_key_for_model:{model}")
        payload = {
            "model": model,
            "messages": messages,
            "api_key": api_key,
            "num_retries": 0,
            **kwargs,
        }
        if auth.get("base_url"):
            payload["base_url"] = auth["base_url"]
        return completion_fn(**payload)

    @staticmethod
    def auth_for_model(model: str) -> dict[str, Any]:
        if model.startswith("deepseek/"):
            return {
                "api_key": settings.deepseek_api_key,
                "base_url": settings.deepseek_base_url,
            }
        if model.startswith("volcengine/"):
            return {
                "api_key": settings.volcengine_api_key
                or settings.deepseek_api_key
                or settings.qwen_api_key,
                "base_url": settings.volcengine_base_url,
            }
        if model.startswith("openai/qwen") or model.startswith("qwen/"):
            return {
                "api_key": settings.qwen_api_key,
                "base_url": settings.qwen_base_url,
            }
        if (
            model.startswith("openai/")
            and settings.llm_provider.strip().lower() == "qwen"
        ):
            return {
                "api_key": settings.qwen_api_key,
                "base_url": settings.qwen_base_url,
            }
        if model.startswith("openrouter/"):
            return {
                "api_key": settings.openrouter_api_key,
                "base_url": settings.openrouter_base_url,
            }
        if model.startswith("anthropic/"):
            return {"api_key": settings.anthropic_api_key}
        if model.startswith("gemini/"):
            return {"api_key": settings.google_api_key}
        return {"api_key": settings.openai_api_key}

    @staticmethod
    def has_any_credential() -> bool:
        return any(
            bool(k)
            for k in (
                settings.openai_api_key,
                settings.openrouter_api_key,
                settings.deepseek_api_key,
                settings.qwen_api_key,
                settings.volcengine_api_key,
                settings.anthropic_api_key,
                settings.google_api_key,
            )
        )
