from __future__ import annotations

import json
import hashlib
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chromadb
from sqlmodel import Field as SQLField
from sqlmodel import Session, SQLModel, create_engine, select

from evoagent.core.config import settings

try:
    from litellm import embedding as litellm_embedding
except Exception:  # pragma: no cover
    litellm_embedding = None


class EpisodeRecord(SQLModel, table=True):
    id: int | None = SQLField(default=None, primary_key=True)
    task_id: str = SQLField(index=True)
    summary: str
    success: bool
    score: float
    created_at: str = SQLField(index=True)


class ConversationTurnRecord(SQLModel, table=True):
    id: int | None = SQLField(default=None, primary_key=True)
    session_id: str = SQLField(index=True)
    task_id: str = SQLField(index=True)
    role: str = SQLField(index=True)
    content: str
    created_at: str = SQLField(index=True)


@dataclass
class WorkingMemory:
    variables: dict[str, Any] = field(default_factory=dict)
    intermediate_steps: list[str] = field(default_factory=list)

    def reset(self) -> None:
        self.variables.clear()
        self.intermediate_steps.clear()


class EpisodicMemory:
    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{db_path}")
        SQLModel.metadata.create_all(self.engine)

    def add(self, task_id: str, summary: str, success: bool, score: float) -> None:
        record = EpisodeRecord(
            task_id=task_id,
            summary=summary,
            success=success,
            score=score,
            created_at=datetime.now(tz=timezone.utc).isoformat(),
        )
        with Session(self.engine) as session:
            session.add(record)
            session.commit()

    def search_recent(self, limit: int = 5) -> list[EpisodeRecord]:
        with Session(self.engine) as session:
            stmt = select(EpisodeRecord).order_by(EpisodeRecord.id.desc()).limit(limit)
            return list(session.exec(stmt))


class ConversationMemory:
    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{db_path}")
        SQLModel.metadata.create_all(self.engine)

    def add_turn(self, session_id: str, task_id: str, role: str, content: str) -> None:
        record = ConversationTurnRecord(
            session_id=session_id,
            task_id=task_id,
            role=role,
            content=content,
            created_at=datetime.now(tz=timezone.utc).isoformat(),
        )
        with Session(self.engine) as session:
            session.add(record)
            session.commit()

    def recent_turns(self, session_id: str, limit: int = 8) -> list[dict[str, str]]:
        with Session(self.engine) as session:
            stmt = (
                select(ConversationTurnRecord)
                .where(ConversationTurnRecord.session_id == session_id)
                .order_by(ConversationTurnRecord.id.desc())
                .limit(max(1, limit))
            )
            rows = list(session.exec(stmt))

        rows.reverse()
        return [
            {
                "role": r.role,
                "content": r.content,
                "task_id": r.task_id,
                "created_at": r.created_at,
            }
            for r in rows
        ]

    def clear_session(self, session_id: str) -> int:
        with Session(self.engine) as session:
            stmt = select(ConversationTurnRecord).where(
                ConversationTurnRecord.session_id == session_id
            )
            rows = list(session.exec(stmt))
            deleted = len(rows)
            for row in rows:
                session.delete(row)
            session.commit()
        return deleted


class SemanticMemory:
    def __init__(
        self, store_path: Path, collection_name: str = "semantic_knowledge"
    ) -> None:
        store_path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(store_path))
        self.collection = self.client.get_or_create_collection(collection_name)
        self._embed_cache: dict[str, list[float]] = {}

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        lowered = text.lower()
        latin = re.findall(r"[a-z0-9_]{2,}", lowered)
        cjk = [ch for ch in lowered if "\u4e00" <= ch <= "\u9fff"]
        return latin + cjk

    @staticmethod
    def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
        if not scores:
            return {}
        max_score = max(scores.values())
        min_score = min(scores.values())
        if abs(max_score - min_score) < 1e-9:
            return {k: 1.0 for k in scores}
        gap = max_score - min_score
        return {k: (v - min_score) / gap for k, v in scores.items()}

    @staticmethod
    def _embedding_auth(model: str) -> dict[str, Any]:
        normalized = model.strip().lower()
        if normalized.startswith("deepseek/"):
            return {
                "api_key": settings.deepseek_api_key,
                "base_url": settings.deepseek_base_url,
            }
        if normalized.startswith("volcengine/"):
            return {
                "api_key": settings.volcengine_api_key,
                "base_url": settings.volcengine_base_url,
            }
        if normalized.startswith("openrouter/"):
            return {
                "api_key": settings.openrouter_api_key,
                "base_url": settings.openrouter_base_url,
            }
        if (
            normalized.startswith("openai/")
            and settings.llm_provider.strip().lower() == "qwen"
        ):
            return {
                "api_key": settings.qwen_api_key,
                "base_url": settings.qwen_base_url,
            }
        if normalized.startswith("openai/qwen") or normalized.startswith("qwen/"):
            return {
                "api_key": settings.qwen_api_key,
                "base_url": settings.qwen_base_url,
            }
        if normalized.startswith("anthropic/"):
            return {"api_key": settings.anthropic_api_key}
        if normalized.startswith("gemini/"):
            return {"api_key": settings.google_api_key}
        return {"api_key": settings.openai_api_key}

    @staticmethod
    def _legacy_embed(text: str, dim: int = 32) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values = list(digest)
        if len(values) < dim:
            values = (values * ((dim // len(values)) + 1))[:dim]
        else:
            values = values[:dim]
        return [(v / 127.5) - 1.0 for v in values]

    @staticmethod
    def _bm25_scores(
        query_tokens: list[str], docs_tokens: list[list[str]]
    ) -> list[float]:
        if not query_tokens or not docs_tokens:
            return [0.0 for _ in docs_tokens]

        n_docs = len(docs_tokens)
        avg_len = sum(len(tokens) for tokens in docs_tokens) / max(1, n_docs)
        avg_len = max(avg_len, 1.0)
        k1 = 1.5
        b = 0.75

        doc_freq: dict[str, int] = {}
        for tokens in docs_tokens:
            for token in set(tokens):
                doc_freq[token] = doc_freq.get(token, 0) + 1

        idf: dict[str, float] = {}
        for token in query_tokens:
            df = doc_freq.get(token, 0)
            idf[token] = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))

        scores: list[float] = []
        for tokens in docs_tokens:
            tf: dict[str, int] = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1
            doc_len = max(1, len(tokens))
            score = 0.0
            for token in query_tokens:
                freq = tf.get(token, 0)
                if freq <= 0:
                    continue
                numerator = freq * (k1 + 1.0)
                denominator = freq + k1 * (1.0 - b + b * (doc_len / avg_len))
                score += idf.get(token, 0.0) * (numerator / max(1e-9, denominator))
            scores.append(score)
        return scores

    def _embed(self, text: str) -> list[float] | None:
        if not text.strip() or litellm_embedding is None:
            return None

        cached = self._embed_cache.get(text)
        if cached is not None:
            return cached

        model = settings.semantic_embedding_model.strip()
        if not model:
            return None

        auth = self._embedding_auth(model)
        api_key = auth.get("api_key")
        if not api_key:
            return None

        kwargs: dict[str, Any] = {
            "model": model,
            "input": [text],
            "api_key": api_key,
            "timeout": max(5, settings.llm_timeout_seconds),
        }
        if auth.get("base_url"):
            kwargs["base_url"] = auth["base_url"]

        try:
            resp = litellm_embedding(**kwargs)
            data = getattr(resp, "data", None) or []
            if not data:
                return None
            vector = list(getattr(data[0], "embedding", None) or [])
            if not vector:
                return None
            self._embed_cache[text] = vector
            return vector
        except Exception:
            return None

    def _existing_embedding_dim(self) -> int | None:
        try:
            payload = self.collection.get(include=["embeddings"], limit=1)
        except Exception:
            return None

        embeddings = payload.get("embeddings")
        if embeddings is None or len(embeddings) == 0:
            return None

        first = embeddings[0]
        if first is None:
            return None
        try:
            return len(first)
        except Exception:
            return None

    def _storage_embedding(self, text: str) -> list[float]:
        existing_dim = self._existing_embedding_dim()
        external = self._embed(text)
        if external is not None and (
            existing_dim is None or len(external) == existing_dim
        ):
            return external
        return self._legacy_embed(text, dim=existing_dim or 32)

    def _upsert_document(self, item_id: str, doc: str, meta: dict[str, Any]) -> bool:
        payload: dict[str, Any] = {
            "ids": [item_id],
            "documents": [doc],
            "metadatas": [meta],
            "embeddings": [self._storage_embedding(doc)],
        }
        try:
            self.collection.upsert(**payload)
            return True
        except Exception as exc:
            match = re.search(r"dimension of (\d+)", str(exc))
            if match:
                payload["embeddings"] = [
                    self._legacy_embed(doc, dim=int(match.group(1)))
                ]
                try:
                    self.collection.upsert(**payload)
                    return True
                except Exception:
                    return False
            return False

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(tz=timezone.utc).isoformat()

    @staticmethod
    def _parse_iso(text: str | None) -> datetime | None:
        if not text:
            return None
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00"))
        except Exception:
            return None

    def _touch(
        self, ids: list[str], docs: list[str], metas: list[dict[str, Any]]
    ) -> None:
        now = self._now_iso()
        for idx in range(min(len(ids), len(docs), len(metas))):
            item_id = ids[idx]
            doc = docs[idx]
            meta = dict(metas[idx] or {})
            old_access = meta.get("access_count", 0)
            try:
                access_count = int(old_access)
            except Exception:
                access_count = 0
            meta["access_count"] = max(0, access_count) + 1
            meta["last_access_at"] = now
            meta.setdefault("created_at", now)
            self._upsert_document(item_id=item_id, doc=doc, meta=meta)

    def upsert_fact(
        self, key: str, fact: str, metadata: dict[str, Any] | None = None
    ) -> None:
        now = self._now_iso()
        merged = dict(metadata or {})
        merged.setdefault("created_at", now)
        merged.setdefault("last_access_at", now)
        merged.setdefault("access_count", 0)
        self._upsert_document(item_id=key, doc=fact, meta=merged)

    def query(self, text: str, top_k: int = 3) -> list[dict[str, Any]]:
        top_k = max(1, top_k)
        mode = settings.semantic_retrieval_mode.strip().lower()
        if mode not in {"embedding", "bm25", "hybrid"}:
            mode = "hybrid"

        alpha = max(0.0, min(settings.semantic_hybrid_alpha, 1.0))
        candidates: dict[str, dict[str, Any]] = {}
        vector_scores: dict[str, float] = {}

        query_embedding = self._embed(text)
        if query_embedding is not None and mode in {"embedding", "hybrid"}:
            try:
                result = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=max(top_k * 3, 8),
                    include=["documents", "metadatas", "distances"],
                )
            except Exception:
                result = {}

            ids = list(result.get("ids", [[]])[0])
            docs = list(result.get("documents", [[]])[0])
            metas = [m or {} for m in list(result.get("metadatas", [[]])[0])]
            distances = list(result.get("distances", [[]])[0])
            for idx in range(min(len(ids), len(docs), len(metas))):
                item_id = ids[idx]
                distance = float(distances[idx]) if idx < len(distances) else 1.0
                vector_scores[item_id] = max(0.0, 1.0 / (1.0 + max(0.0, distance)))
                candidates[item_id] = {
                    "id": item_id,
                    "fact": docs[idx],
                    "metadata": metas[idx],
                }

        bm25_scores_by_id: dict[str, float] = {}
        if mode in {"bm25", "hybrid"}:
            try:
                payload = self.collection.get(include=["documents", "metadatas"])
            except Exception:
                payload = {}

            all_ids = list(payload.get("ids", []) or [])
            all_docs = list(payload.get("documents", []) or [])
            all_metas = [m or {} for m in list(payload.get("metadatas", []) or [])]
            docs_tokens = [self._tokenize(str(doc)) for doc in all_docs]
            raw_scores = self._bm25_scores(self._tokenize(text), docs_tokens)
            for idx in range(
                min(len(all_ids), len(all_docs), len(all_metas), len(raw_scores))
            ):
                item_id = all_ids[idx]
                bm25_scores_by_id[item_id] = float(raw_scores[idx])
                candidates.setdefault(
                    item_id,
                    {
                        "id": item_id,
                        "fact": all_docs[idx],
                        "metadata": all_metas[idx],
                    },
                )

        norm_vector = self._normalize_scores(vector_scores)
        norm_bm25 = self._normalize_scores(bm25_scores_by_id)

        scored: list[tuple[float, dict[str, Any]]] = []
        for item_id, item in candidates.items():
            v = norm_vector.get(item_id, 0.0)
            b = norm_bm25.get(item_id, 0.0)
            if mode == "embedding":
                final_score = v
            elif mode == "bm25":
                final_score = b
            else:
                final_score = alpha * v + (1.0 - alpha) * b
            if final_score <= 0.0:
                continue
            scored.append((final_score, item))

        if not scored:
            return []

        scored.sort(key=lambda x: x[0], reverse=True)
        selected = scored[:top_k]
        self._touch(
            ids=[item["id"] for _, item in selected],
            docs=[item["fact"] for _, item in selected],
            metas=[item["metadata"] for _, item in selected],
        )
        return [
            {
                "id": item["id"],
                "fact": item["fact"],
                "metadata": item["metadata"],
                "score": round(score, 6),
            }
            for score, item in selected
        ]

    def prune_stale(
        self, max_age_days: int, min_access_count: int, max_delete: int = 200
    ) -> int:
        try:
            payload = self.collection.get(include=["documents", "metadatas"])
        except Exception:
            return 0

        ids = payload.get("ids", []) or []
        metas = payload.get("metadatas", []) or []
        if not ids:
            return 0

        now = datetime.now(tz=timezone.utc)
        delete_ids: list[str] = []
        for idx in range(min(len(ids), len(metas))):
            item_id = ids[idx]
            meta = metas[idx] or {}

            old_access = meta.get("access_count", 0)
            try:
                access_count = int(old_access)
            except Exception:
                access_count = 0

            last_access = self._parse_iso(str(meta.get("last_access_at", "")))
            if last_access is None:
                last_access = self._parse_iso(str(meta.get("created_at", "")))
            if last_access is None:
                continue

            age_days = (now - last_access).days
            if age_days >= max_age_days and access_count <= min_access_count:
                delete_ids.append(item_id)
                if len(delete_ids) >= max_delete:
                    break

        if delete_ids:
            try:
                self.collection.delete(ids=delete_ids)
            except Exception:
                return 0
        return len(delete_ids)


class MemoryManager:
    def __init__(self, db_path: Path, semantic_store_path: Path) -> None:
        self.working = WorkingMemory()
        self.episodic = EpisodicMemory(db_path)
        self.conversation = ConversationMemory(db_path)
        self.semantic = SemanticMemory(
            semantic_store_path, collection_name="semantic_knowledge"
        )
        self.negative_semantic = SemanticMemory(
            semantic_store_path, collection_name="negative_knowledge"
        )

    def add_step(self, text: str) -> None:
        self.working.intermediate_steps.append(text)

    def snapshot_working(self) -> str:
        payload = {
            "variables": self.working.variables,
            "intermediate_steps": self.working.intermediate_steps,
        }
        return json.dumps(payload, ensure_ascii=False)

    def distill_episode_to_semantic(
        self,
        task_id: str,
        summary: str,
        score: float,
        success: bool,
        failure_reason: str | None = None,
    ) -> None:
        if success and score >= settings.memory_distill_score_threshold:
            key = f"episode:{task_id}"
            metadata = {"score": score, "kind": "distilled_episode"}
            self.semantic.upsert_fact(key=key, fact=summary, metadata=metadata)
            return

        if not success:
            key = f"failure:{task_id}"
            metadata = {
                "score": score,
                "kind": "negative_episode",
                "failure_reason": failure_reason or "unknown",
            }
            self.negative_semantic.upsert_fact(key=key, fact=summary, metadata=metadata)

    def recall_semantic(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        return self.semantic.query(text=query, top_k=top_k)

    def recall_negative(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        return self.negative_semantic.query(text=query, top_k=top_k)

    def run_semantic_forgetting(self) -> int:
        return self.semantic.prune_stale(
            max_age_days=max(1, settings.semantic_forget_max_age_days),
            min_access_count=max(0, settings.semantic_forget_min_access_count),
            max_delete=max(1, settings.semantic_forget_max_delete),
        )
