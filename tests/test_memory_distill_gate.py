from evoagent.memory.layers import MemoryManager


def test_low_score_episode_not_promoted_to_semantic(tmp_path) -> None:
    mm = MemoryManager(db_path=tmp_path / "evo.db", semantic_store_path=tmp_path / "semantic")
    mm.distill_episode_to_semantic(
        task_id="t1",
        summary="low quality case",
        score=0.5,
        success=True,
        failure_reason=None,
    )
    hits = mm.recall_semantic("low quality case", top_k=3)
    assert all(item.get("id") != "episode:t1" for item in hits)


def test_failure_case_saved_for_negative_learning(tmp_path) -> None:
    mm = MemoryManager(db_path=tmp_path / "evo.db", semantic_store_path=tmp_path / "semantic")
    mm.distill_episode_to_semantic(
        task_id="t2",
        summary="tool timeout and api failed",
        score=0.2,
        success=False,
        failure_reason="tool_failure",
    )
    hits = mm.recall_negative("tool timeout and api failed", top_k=3)
    assert any(item.get("id") == "failure:t2" for item in hits)
