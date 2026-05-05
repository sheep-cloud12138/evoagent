import asyncio

from evoagent.app import EvoAgentSystem


def test_time_query_uses_builtin_fastpath() -> None:
    system = EvoAgentSystem()
    result = asyncio.run(system.run("现在几点", context={"session_id": "test-time-fastpath"}))

    assert result.metadata.get("mode") == "builtin_time_fastpath"
    assert result.decision.difficulty.value == "simple"
    assert result.outputs == []
    assert result.final_answer.startswith("当前时间：")
    assert "UTC" in result.final_answer
