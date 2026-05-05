from evoagent.core import llm as llm_module
from evoagent.core.config import settings
from evoagent.core.llm import LLMClient


class _Message:
    def __init__(self, content: str) -> None:
        self.content = content


class _Choice:
    def __init__(self, content: str) -> None:
        self.message = _Message(content)


class _Response:
    def __init__(self, content: str) -> None:
        self.choices = [_Choice(content)]


def test_failed_model_is_cooled_down(monkeypatch) -> None:
    client = LLMClient()
    monkeypatch.setattr(settings, "openai_api_key", "test-key")
    monkeypatch.setattr(settings, "llm_model_failure_cooldown_seconds", 600)

    calls: list[str] = []

    def fake_completion(**kwargs):
        model = kwargs["model"]
        calls.append(model)
        if model == "openai/bad-model":
            raise RuntimeError("timeout")
        return _Response("ok")

    monkeypatch.setattr(llm_module, "completion", fake_completion)

    out1 = client.generate(
        "hello",
        profile=LLMClient.Profile.STANDARD,
        force_models=["openai/bad-model", "openai/good-model"],
    )
    assert out1 == "ok"
    assert calls == ["openai/bad-model", "openai/good-model"]

    calls.clear()
    out2 = client.generate(
        "hello-again",
        profile=LLMClient.Profile.STANDARD,
        force_models=["openai/bad-model", "openai/good-model"],
    )
    assert out2 == "ok"
    assert calls == ["openai/good-model"]


def test_all_cooled_down_models_probe_shortest_cooldown(monkeypatch) -> None:
    client = LLMClient()
    monkeypatch.setattr(settings, "openai_api_key", "test-key")
    monkeypatch.setattr(settings, "llm_model_failure_cooldown_seconds", 600)

    calls: list[str] = []

    def fake_completion(**kwargs):
        calls.append(kwargs["model"])
        raise RuntimeError("timeout")

    monkeypatch.setattr(llm_module, "completion", fake_completion)

    out1 = client.generate("hello", force_models=["openai/bad-model"])
    assert out1.startswith("[Fallback-LLM]")
    assert calls == ["openai/bad-model"]

    calls.clear()
    out2 = client.generate("hello-again", force_models=["openai/bad-model"])
    assert out2.startswith("[Fallback-LLM]")
    assert calls == ["openai/bad-model"]


def test_model_cooldown_persists_across_clients(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(settings, "llm_health_cache_enabled", True)
    monkeypatch.setattr(settings, "llm_health_cache_path", tmp_path / "llm_health.json")
    monkeypatch.setattr(settings, "openai_api_key", "test-key")
    monkeypatch.setattr(settings, "llm_model_failure_cooldown_seconds", 600)

    calls: list[str] = []

    def fake_completion(**kwargs):
        model = kwargs["model"]
        calls.append(model)
        if model == "openai/bad-model":
            raise RuntimeError("timeout")
        return _Response("ok")

    monkeypatch.setattr(llm_module, "completion", fake_completion)

    first = LLMClient()
    out1 = first.generate("hello", force_models=["openai/bad-model"])
    assert out1.startswith("[Fallback-LLM]")
    assert calls == ["openai/bad-model"]

    calls.clear()
    second = LLMClient()
    out2 = second.generate(
        "hello-again",
        force_models=["openai/bad-model", "openai/good-model"],
    )
    assert out2 == "ok"
    assert calls == ["openai/good-model"]

    report = second.health_report(["openai/bad-model", "openai/good-model"])
    by_model = {row["model"]: row for row in report}
    assert by_model["openai/bad-model"]["status"] == "cooled_down"
    assert by_model["openai/bad-model"]["failures"] == 1


def test_model_attempts_are_limited(monkeypatch) -> None:
    monkeypatch.setattr(settings, "llm_max_candidate_attempts", 2)
    assert LLMClient._limit_model_attempts(["a", "b", "c"]) == ["a", "b"]
