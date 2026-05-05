import asyncio

from evoagent.app import EvoAgentSystem


def test_run_contains_trace_metadata() -> None:
    system = EvoAgentSystem()
    result = asyncio.run(system.run("请给出两步计划"))
    assert "trace" in result.metadata
    assert isinstance(result.metadata["trace"], list)
    assert len(result.metadata["trace"]) >= 3
