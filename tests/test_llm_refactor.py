from __future__ import annotations

import time

from evoagent.core.llm import LLMClient
from evoagent.llm import LLMService
from evoagent.llm.health import ModelHealthStore
from evoagent.llm.router import ModelRouter


def test_llm_service_has_old_public_methods() -> None:
    assert LLMClient is LLMService
    service = LLMService()
    assert callable(service.generate)
    assert callable(service.generate_with_tools_trace)
    assert callable(service.generate_with_tools)
    assert service.Profile.STANDARD == "standard"


def test_model_health_store_cooldown_logic() -> None:
    store = ModelHealthStore(cooldown_seconds=30, persist=False)
    assert store.is_healthy("openai/test")

    store.record_failure("openai/test", "timeout")
    assert not store.is_healthy("openai/test")
    assert store.failure_count["openai/test"] == 1

    store.record_success("openai/test")
    assert store.is_healthy("openai/test")
    assert store.failure_count["openai/test"] == 0
    assert store.last_success["openai/test"] <= time.time()


def test_model_router_returns_healthy_models_only(monkeypatch) -> None:
    health = ModelHealthStore(cooldown_seconds=30, persist=False)
    router = ModelRouter(health)
    monkeypatch.setattr(router, "_model_candidates", lambda profile: ["openai/bad", "openai/good"])

    health.record_failure("openai/bad", "timeout")

    assert router.candidates_for("standard") == ["openai/good"]


def test_llm_service_uses_mock_litellm(monkeypatch) -> None:
    from evoagent.core import llm as llm_module
    from evoagent.core.config import settings

    class _Message:
        content = "ok"

    class _Choice:
        message = _Message()

    class _Response:
        choices = [_Choice()]

    calls: list[str] = []

    def fake_completion(**kwargs):
        calls.append(kwargs["model"])
        return _Response()

    monkeypatch.setattr(settings, "openai_api_key", "test-key")
    monkeypatch.setattr(llm_module, "completion", fake_completion)

    service = LLMService()
    assert service.generate("hello", force_models=["openai/test"]) == "ok"
    assert calls == ["openai/test"]
