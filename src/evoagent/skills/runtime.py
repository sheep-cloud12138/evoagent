from __future__ import annotations

import json
import random
import re
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

from sqlmodel import Session, create_engine, select

from evoagent.core.config import settings
from evoagent.core.models import SkillManifest
from evoagent.core.llm import LLMClient
from evoagent.core.tools import register_skill_tool, unregister_tools
from evoagent.skills.manifest import SkillManifest as PersistentSkillManifest
from evoagent.skills.sandbox import SandboxValidationError, SkillSandbox


def _allowed_skill_permissions() -> set[str]:
    return {
        item.strip()
        for item in settings.skill_allowed_permissions.split(",")
        if item.strip()
    }


def _normalize_permission(permission: str) -> str:
    return "sandbox" if permission == "pure_python" else permission


class SkillManifestRuntime:
    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or settings.db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{self.db_path}")
        PersistentSkillManifest.metadata.create_all(self.engine)

    @staticmethod
    def check_permissions(manifest: PersistentSkillManifest) -> None:
        allowed = _allowed_skill_permissions()
        denied = [
            perm
            for perm in manifest.permissions
            if _normalize_permission(str(perm)) not in allowed
        ]
        if denied:
            raise PermissionError(f"Skill permission blocked: {', '.join(denied)}")

    def register(self, manifest: PersistentSkillManifest) -> PersistentSkillManifest:
        manifest.min_successes_to_activate = max(
            1, manifest.min_successes_to_activate or settings.skill_min_successes
        )
        with Session(self.engine) as session:
            session.add(manifest)
            session.commit()
            session.refresh(manifest)
            return manifest

    def get_skill(self, name: str) -> PersistentSkillManifest | None:
        with Session(self.engine) as session:
            stmt = (
                select(PersistentSkillManifest)
                .where(PersistentSkillManifest.name == name)
                .where(PersistentSkillManifest.status == "active")
                .limit(1)
            )
            manifest = session.exec(stmt).first()
            if manifest is not None:
                self.check_permissions(manifest)
            return manifest

    def add_result(
        self, skill_id: str, success: bool
    ) -> PersistentSkillManifest | None:
        with Session(self.engine) as session:
            manifest = session.get(PersistentSkillManifest, skill_id)
            if manifest is None:
                return None
            if success:
                manifest.success_count += 1
            else:
                manifest.failure_count += 1
            total = max(manifest.success_count + manifest.failure_count, 1)
            manifest.success_rate = manifest.success_count / total
            manifest.last_used_at = datetime.now(tz=timezone.utc)
            if manifest.success_count >= manifest.min_successes_to_activate:
                manifest.status = "active"
            elif manifest.failure_count / total > 0.5:
                manifest.status = "rejected"
            session.add(manifest)
            session.commit()
            session.refresh(manifest)
            return manifest


class SkillArtifactManager:
    def __init__(self, store_path: Path, llm: LLMClient | None = None) -> None:
        self.store_path = store_path
        self.llm = llm or LLMClient()
        self.store_path.mkdir(parents=True, exist_ok=True)
        self.index_path = self.store_path / "index.json"
        self.ab_path = self.store_path / "ab_metrics.json"
        if not self.index_path.exists():
            self.index_path.write_text("[]", encoding="utf-8")
        if not self.ab_path.exists():
            self.ab_path.write_text("{}", encoding="utf-8")

    @staticmethod
    def _extract_json_object(raw_text: str) -> dict[str, Any] | None:
        text = raw_text.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            pass

        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            parsed = json.loads(text[start : end + 1])
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None

    def _llm_query_tokens(self, query: str) -> set[str]:
        prompt = (
            "你是技能检索重写器。请从用户请求提取有助于技能匹配的关键词。仅输出 JSON。\n"
            "JSON schema:\n"
            "{\n"
            '  "tokens": ["keyword1", "keyword2", "keyword3"]\n'
            "}\n"
            "约束: token 仅包含字母、数字、下划线，长度 2-32。\n"
            f"用户请求: {query}"
        )
        raw = self.llm.generate(prompt, temperature=0.0, profile=LLMClient.Profile.FAST)
        if raw.startswith("[Fallback-LLM]"):
            return set()
        payload = self._extract_json_object(raw)
        if payload is None:
            return set()
        raw_tokens = payload.get("tokens", [])
        if not isinstance(raw_tokens, list):
            return set()
        tokens: set[str] = set()
        for item in raw_tokens:
            token = str(item).strip().lower()
            if re.fullmatch(r"[a-z0-9_]{2,32}", token):
                tokens.add(token)
        return tokens

    def _safe_name(self, text: str) -> str:
        normalized = re.sub(r"[^a-zA-Z0-9_]+", "_", text.strip())
        normalized = normalized.strip("_")
        return normalized or "generated_skill"

    @staticmethod
    def _cjk_ngrams(text: str) -> set[str]:
        chars = [ch for ch in text if "\u4e00" <= ch <= "\u9fff"]
        tokens: set[str] = set()
        for size in (2, 3, 4):
            for idx in range(0, max(0, len(chars) - size + 1)):
                tokens.add("".join(chars[idx : idx + size]))
        return tokens

    @classmethod
    def _text_tokens(cls, text: str) -> set[str]:
        lowered = text.lower()
        tokens = {
            t for t in re.split(r"[^a-z0-9]+", lowered.replace("_", " ")) if len(t) >= 2
        }
        tokens |= {t for t in re.findall(r"[a-z0-9_]{2,}", lowered) if "_" in t}
        tokens |= cls._cjk_ngrams(lowered)
        return tokens

    @staticmethod
    def _latin_tokens(tokens: set[str]) -> set[str]:
        return {token for token in tokens if re.fullmatch(r"[a-z0-9_]+", token)}

    @staticmethod
    def _cjk_tokens(tokens: set[str]) -> set[str]:
        return {
            token for token in tokens if any("\u4e00" <= ch <= "\u9fff" for ch in token)
        }

    @staticmethod
    def _default_input_schema() -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "input_text": {
                    "type": "string",
                    "description": "Original user task or text to process with this evolved skill.",
                }
            },
            "required": ["input_text"],
            "additionalProperties": False,
        }

    @staticmethod
    def _default_output_schema() -> dict[str, Any]:
        return {"type": "string", "description": "Skill execution result."}

    def _build_manifest(self, draft: Any, version: str, status: str) -> SkillManifest:
        return SkillManifest(
            name=str(draft.name),
            version=version,
            description=str(draft.description),
            tags=[str(tag) for tag in getattr(draft, "tags", [])],
            input_schema=self._default_input_schema(),
            output_schema=self._default_output_schema(),
            permissions=["pure_python"],
            applicability=str(draft.description),
            status=status,
        )

    @staticmethod
    def _overlap_score(left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        overlap = len(left & right)
        return overlap / max(1, min(len(left), len(right)))

    @staticmethod
    def _tool_name_for_item(item: dict[str, Any]) -> str:
        safe_name = re.sub(
            r"[^a-zA-Z0-9_]+",
            "_",
            str(item.get("safe_name") or item.get("name") or "generated_skill"),
        )
        safe_version = re.sub(
            r"[^a-zA-Z0-9_]+", "_", str(item.get("version") or "0_0_0")
        )
        return f"skill_{safe_name.strip('_')}_{safe_version.strip('_')}"

    def _read_index(self) -> list[dict[str, Any]]:
        try:
            return json.loads(self.index_path.read_text(encoding="utf-8"))
        except Exception:
            return []

    def _write_index(self, items: list[dict[str, Any]]) -> None:
        self.index_path.write_text(
            json.dumps(items, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _read_ab(self) -> dict[str, Any]:
        try:
            return json.loads(self.ab_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _write_ab(self, payload: dict[str, Any]) -> None:
        self.ab_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def persist(self, draft: Any, version: str, status: str = "active") -> Path:
        skill_name = self._safe_name(draft.name)
        file_name = f"{skill_name}_v{version.replace('.', '_')}.py"
        test_name = f"test_{skill_name}_v{version.replace('.', '_')}.py"
        skill_path = self.store_path / file_name
        test_path = self.store_path / test_name

        skill_path.write_text(draft.code, encoding="utf-8")
        test_path.write_text(draft.tests, encoding="utf-8")

        items = self._read_index()
        if status == "active":
            for item in items:
                if (
                    item.get("safe_name") == skill_name
                    and item.get("status") == "active"
                ):
                    item["status"] = "archived"
        manifest = self._build_manifest(draft, version, status)
        item = {
            "name": draft.name,
            "safe_name": skill_name,
            "version": version,
            "description": draft.description,
            "tags": draft.tags,
            "manifest": manifest.model_dump(mode="json"),
            "skill_path": str(skill_path),
            "test_path": str(test_path),
            "status": status,
            "created_at": datetime.now(tz=timezone.utc).isoformat(),
        }
        items.append(item)
        self._write_index(items)
        if status == "active":
            self.register_active_skill_tools()
        return skill_path

    def list_versions(self, skill_name: str) -> list[dict[str, Any]]:
        safe = self._safe_name(skill_name)
        items = [i for i in self._read_index() if i.get("safe_name") == safe]
        return sorted(items, key=lambda x: x.get("created_at", ""), reverse=True)

    def activate_version(self, skill_name: str, version: str) -> bool:
        safe = self._safe_name(skill_name)
        updated = False
        items = self._read_index()
        for item in items:
            if item.get("safe_name") != safe:
                continue
            if item.get("version") == version:
                item["status"] = "active"
                updated = True
            else:
                item["status"] = "archived"
        if updated:
            self._write_index(items)
            self.register_active_skill_tools()
        return updated

    def set_version_status(self, skill_name: str, version: str, status: str) -> bool:
        safe = self._safe_name(skill_name)
        changed = False
        items = self._read_index()
        for item in items:
            if item.get("safe_name") == safe and item.get("version") == version:
                item["status"] = status
                changed = True
                break
        if changed:
            self._write_index(items)
            if status == "active":
                self.register_active_skill_tools()
        return changed

    def _best_match(
        self, items: list[dict[str, Any]], tokens: set[str]
    ) -> tuple[dict[str, Any] | None, float]:
        best_item: dict[str, Any] | None = None
        best_score = 0.0
        query_latin_tokens = self._latin_tokens(tokens)
        query_cjk_tokens = self._cjk_tokens(tokens)
        for item in items:
            corpus = " ".join(
                [
                    item.get("name", ""),
                    item.get("description", ""),
                    " ".join(item.get("tags", [])),
                ]
            ).lower()
            words = self._text_tokens(corpus)
            if not words:
                continue
            score = max(
                self._overlap_score(tokens, words),
                self._overlap_score(query_latin_tokens, self._latin_tokens(words)),
                self._overlap_score(query_cjk_tokens, self._cjk_tokens(words)),
            )
            if score > best_score:
                best_score = score
                best_item = item
        return best_item, best_score

    def find_relevant(self, query: str) -> tuple[dict[str, Any] | None, float]:
        tokens = self._text_tokens(query)
        # Removed task-specific hint injection (github/hot/download etc.) so
        # relevance expansion is generated by LLM and stays query-agnostic.
        tokens |= self._llm_query_tokens(query)
        if not tokens:
            return None, 0.0

        all_items = self._read_index()
        active_items = [i for i in all_items if i.get("status") == "active"]
        candidate_items = [i for i in all_items if i.get("status") == "candidate"]

        best_active, active_score = self._best_match(active_items, tokens)
        best_candidate, candidate_score = self._best_match(candidate_items, tokens)

        if best_active is None and best_candidate is None:
            return None, 0.0
        if best_candidate is None:
            return best_active, active_score
        if best_active is None:
            return best_candidate, candidate_score

        rollout_rate = max(0.0, min(settings.skill_candidate_rollout_rate, 1.0))
        candidate_eligible = candidate_score >= max(0.1, active_score * 0.8)
        if candidate_eligible and random.random() < rollout_rate:
            return best_candidate, candidate_score
        return best_active, active_score

    def get_skill(self, name: str) -> dict[str, Any] | None:
        safe = self._safe_name(name)
        for item in self._read_index():
            if item.get("safe_name") == safe and item.get("status") == "active":
                return item
        return None

    def add_result(self, skill_id: str, success: bool) -> dict[str, Any] | None:
        items = self._read_index()
        matched: dict[str, Any] | None = None
        for item in items:
            if skill_id not in {
                str(item.get("name")),
                str(item.get("safe_name")),
                str(item.get("version")),
            }:
                continue
            manifest = item.get("manifest", {})
            if not isinstance(manifest, dict):
                manifest = {}
            success_count = int(manifest.get("success_count", 0))
            failure_count = int(manifest.get("failure_count", 0))
            if success:
                success_count += 1
            else:
                failure_count += 1
            total = max(success_count + failure_count, 1)
            success_rate = success_count / total
            manifest["success_count"] = success_count
            manifest["failure_count"] = failure_count
            manifest["success_rate"] = success_rate
            manifest["last_used_at"] = datetime.now(tz=timezone.utc).isoformat()
            min_successes = int(
                manifest.get("min_successes_to_activate", settings.skill_min_successes)
            )
            if success_count >= min_successes:
                manifest["status"] = "active"
                item["status"] = "active"
            elif failure_count / total > 0.5:
                manifest["status"] = "rejected"
                item["status"] = "rejected"
            item["manifest"] = manifest
            matched = item
            break
        if matched is not None:
            self._write_index(items)
            self.register_active_skill_tools()
        return matched

    def register_skill_tool(self, item: dict[str, Any]) -> str | None:
        if item.get("status") != "active":
            return None
        tool_name = self._tool_name_for_item(item)
        description = str(
            item.get("description") or item.get("name") or "Evolved skill"
        )
        tags = item.get("tags", [])
        if isinstance(tags, list) and tags:
            description = (
                f"{description} Tags: {', '.join(str(tag) for tag in tags[:8])}"
            )
        item_snapshot = dict(item)

        def _handler(args: dict[str, Any]) -> str:
            input_text = str(args.get("input_text") or args.get("query") or "").strip()
            if not input_text:
                return "error: missing input_text"
            return self.execute(item_snapshot, input_text)

        register_skill_tool(
            name=tool_name,
            description=description,
            parameters=(item_snapshot.get("manifest") or {}).get(
                "input_schema", self._default_input_schema()
            ),
            handler=_handler,
            category="skill",
        )
        return tool_name

    def register_active_skill_tools(self) -> list[str]:
        unregister_tools(source="skill", category="skill", name_prefix="skill_")
        tool_names: list[str] = []
        for item in self._read_index():
            if item.get("status") != "active":
                continue
            tool_name = self.register_skill_tool(item)
            if tool_name:
                tool_names.append(tool_name)
        return tool_names

    def execute(self, item: dict[str, Any], input_text: str) -> str:
        manifest = item.get("manifest", {})
        permissions = (
            manifest.get("permissions", []) if isinstance(manifest, dict) else []
        )
        if not isinstance(permissions, list):
            permissions = []
        allowed = _allowed_skill_permissions()
        denied = [
            perm
            for perm in permissions
            if _normalize_permission(str(perm)) not in allowed
        ]
        if denied:
            raise RuntimeError(
                f"Skill permission blocked: {', '.join(str(item) for item in denied)}"
            )

        path = Path(item["skill_path"])
        if not path.exists() or not path.is_file():
            raise RuntimeError("Skill artifact not found")

        code = path.read_text(encoding="utf-8")
        try:
            SkillSandbox().assert_code_safe(code)
        except SandboxValidationError as exc:
            raise RuntimeError(f"Unsafe skill artifact blocked: {exc}") from exc

        runner = (
            "import importlib.util, json, sys\n"
            "path = sys.argv[1]\n"
            "payload = json.loads(sys.stdin.read() or '{}')\n"
            "spec = importlib.util.spec_from_file_location('skill_runtime_impl', path)\n"
            "if spec is None or spec.loader is None:\n"
            "    raise RuntimeError('Cannot load skill module')\n"
            "module = importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(module)\n"
            "execute_fn = getattr(module, 'execute', None)\n"
            "if execute_fn is None:\n"
            "    raise RuntimeError('Skill missing execute(input_text: str) -> str')\n"
            "result = execute_fn(str(payload.get('input_text', '')))\n"
            "sys.stdout.write(json.dumps({'ok': True, 'result': str(result)}, ensure_ascii=False))\n"
        )
        payload = json.dumps({"input_text": input_text}, ensure_ascii=False)
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", runner, str(path)],
                input=payload,
                capture_output=True,
                text=True,
                timeout=max(1, settings.skill_runtime_timeout_seconds),
                cwd=str(path.parent),
                env={
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONUTF8": "1",
                },
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("Skill execution timed out") from exc

        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "unknown error").strip()
            raise RuntimeError(f"Skill execution failed: {detail[:500]}")

        try:
            result_payload = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Skill execution returned invalid payload") from exc
        if not isinstance(result_payload, dict) or not result_payload.get("ok"):
            raise RuntimeError("Skill execution returned unsuccessful payload")
        return str(result_payload.get("result", ""))

    def record_ab_result(
        self, skill_name: str, skill_version: str, winner: str
    ) -> None:
        key = f"{self._safe_name(skill_name)}:{skill_version}"
        payload = self._read_ab()
        stat = payload.get(
            key, {"skill_wins": 0, "baseline_wins": 0, "ties": 0, "total": 0}
        )
        if winner == "skill":
            stat["skill_wins"] += 1
        elif winner == "baseline":
            stat["baseline_wins"] += 1
        else:
            stat["ties"] += 1
        stat["total"] += 1
        payload[key] = stat
        self._write_ab(payload)

    def ab_report(self) -> dict[str, Any]:
        return self._read_ab()
