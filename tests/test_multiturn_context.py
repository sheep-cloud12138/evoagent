import asyncio

from evoagent.app import EvoAgentSystem


def test_multiturn_context_persists_by_session() -> None:
    system = EvoAgentSystem()
    session_id = "test_multiturn_context"
    system.memory.conversation.clear_session(session_id)

    first = asyncio.run(system.run("我叫小王", context={"session_id": session_id}))
    second = asyncio.run(system.run("我刚才说我叫什么？", context={"session_id": session_id}))

    assert first.metadata.get("session_id") == session_id
    assert second.metadata.get("session_id") == session_id
    assert int(second.metadata.get("conversation_turns_used", 0)) >= 2

    turns = system.memory.conversation.recent_turns(session_id, limit=10)
    assert len(turns) >= 4
    assert turns[0]["role"] == "user"
    assert turns[1]["role"] == "assistant"

    system.memory.conversation.clear_session(session_id)
