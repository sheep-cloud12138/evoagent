import pytest

from evoagent.core.config import settings


@pytest.fixture(autouse=True)
def _disable_real_llm_credentials(monkeypatch):
    """Keep tests deterministic even when the local .env has real API keys."""
    for attr in (
        "openai_api_key",
        "openrouter_api_key",
        "deepseek_api_key",
        "qwen_api_key",
        "volcengine_api_key",
        "anthropic_api_key",
        "google_api_key",
    ):
        monkeypatch.setattr(settings, attr, None)
    monkeypatch.setattr(settings, "llm_backup_models", "")
    monkeypatch.setattr(settings, "llm_health_cache_enabled", False)
