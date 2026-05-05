from evoagent.core.llm import LLMClient


def test_normalize_qwen_model_prefix() -> None:
    client = LLMClient()
    normalized = client._normalize_model("qwen-plus")
    assert normalized.startswith("openai/")
