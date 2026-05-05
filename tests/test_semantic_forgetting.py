from datetime import datetime, timedelta, timezone

from evoagent.core.config import settings
from evoagent.memory.layers import MemoryManager


def test_semantic_forgetting_removes_stale_low_access(monkeypatch, tmp_path) -> None:
    mm = MemoryManager(db_path=tmp_path / "evo.db", semantic_store_path=tmp_path / "semantic")

    old_time = (datetime.now(tz=timezone.utc) - timedelta(days=100)).isoformat()
    mm.semantic.upsert_fact(
        key="episode:old",
        fact="stale fact",
        metadata={"created_at": old_time, "last_access_at": old_time, "access_count": 0},
    )

    monkeypatch.setattr(settings, "semantic_forget_max_age_days", 30)
    monkeypatch.setattr(settings, "semantic_forget_min_access_count", 0)
    monkeypatch.setattr(settings, "semantic_forget_max_delete", 10)

    removed = mm.run_semantic_forgetting()
    assert removed >= 1

    hits = mm.recall_semantic("stale fact", top_k=3)
    assert all(item.get("id") != "episode:old" for item in hits)
