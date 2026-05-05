from evoagent.core.llm import LLMClient


def test_normalize_model_infers_deepseek() -> None:
    client = LLMClient()
    assert client._normalize_model("deepseek-chat") == "deepseek/deepseek-chat"


def test_auth_for_deepseek_contains_base_url() -> None:
    client = LLMClient()
    auth = client._auth_for_model("deepseek/deepseek-chat")
    assert "api_key" in auth
    assert auth.get("base_url") is not None


def test_endpoint_id_routes_to_volcengine() -> None:
    client = LLMClient()
    normalized = client._normalize_model("ep-m-20260420135846-wtwl9")
    assert normalized == "volcengine/ep-m-20260420135846-wtwl9"
    assert client._model_provider_family(normalized) == "volcengine"


def test_ark_prefix_routes_to_volcengine() -> None:
    client = LLMClient()
    assert client._normalize_model("ark/ep-abc") == "volcengine/ep-abc"


def test_qwen_backup_routes_to_qwen_credentials() -> None:
    client = LLMClient()
    normalized = client._normalize_model("qwen-plus")
    assert normalized == "openai/qwen-plus"
    assert client._model_provider_family(normalized) == "qwen"
    assert client._auth_for_model(normalized).get("base_url") is not None
