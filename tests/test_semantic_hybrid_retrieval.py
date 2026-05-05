from evoagent.core.config import settings
from evoagent.memory.layers import MemoryManager


def test_semantic_query_uses_bm25_fallback_without_embedding_key(monkeypatch, tmp_path) -> None:
    mm = MemoryManager(db_path=tmp_path / "evo.db", semantic_store_path=tmp_path / "semantic")
    mm.semantic.upsert_fact("episode:1", "python async timeout retry strategy", {"kind": "distilled_episode"})
    mm.semantic.upsert_fact("episode:2", "javascript dom rendering tips", {"kind": "distilled_episode"})

    monkeypatch.setattr(settings, "semantic_retrieval_mode", "hybrid")
    monkeypatch.setattr(settings, "semantic_embedding_model", "openai/text-embedding-3-small")
    monkeypatch.setattr(settings, "openai_api_key", None)

    hits = mm.recall_semantic("async timeout", top_k=2)
    assert hits
    assert hits[0]["id"] == "episode:1"
    assert "score" in hits[0]


def test_semantic_query_bm25_mode_returns_ranked_hits(monkeypatch, tmp_path) -> None:
    mm = MemoryManager(db_path=tmp_path / "evo.db", semantic_store_path=tmp_path / "semantic")
    mm.semantic.upsert_fact("episode:db", "sqlite transaction lock and retry", {"kind": "distilled_episode"})
    mm.semantic.upsert_fact("episode:web", "http cache control best practice", {"kind": "distilled_episode"})

    monkeypatch.setattr(settings, "semantic_retrieval_mode", "bm25")
    hits = mm.recall_semantic("sqlite retry", top_k=2)

    assert len(hits) >= 1
    assert hits[0]["id"] == "episode:db"
