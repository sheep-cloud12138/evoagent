from evoagent.core.llm import LLMClient
from evoagent.core.config import settings


def test_fallback_skill_blocks_format() -> None:
    llm = LLMClient()
    prompt = (
        "请生成一个 Python Skill 草案，输出格式固定为 5 段，用 ### 分隔：\n"
        "### name\n### description\n### tags(comma-separated)\n### code\n### tests\n"
    )
    out = llm._fallback(prompt)
    assert out.startswith("[Fallback-LLM]")
    assert "Prompt 摘要" in out


def test_fallback_final_aggregation() -> None:
    llm = LLMClient()
    prompt = "你是核心 Agent，负责冲突消解与最终整合。"
    out = llm._fallback(prompt)
    assert out.startswith("[Fallback-LLM]")


def test_fallback_message_distinguishes_api_failure(monkeypatch) -> None:
    monkeypatch.setattr(settings, "deepseek_api_key", "test-key")
    llm = LLMClient()
    out = llm._fallback("hello", last_error="timeout")
    assert "API 暂不可用" in out
    assert "未配置可用 API Key" not in out
