from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from evoagent.core.config import settings
from evoagent.core.llm import LLMClient
from evoagent.core.models import (
    Difficulty,
    TaskCapability,
    TaskDecision,
    TaskFeatures,
    TaskRequest,
)


@dataclass
class RouterWeights:
    step_weight: float = 0.35
    tool_weight: float = 0.25
    history_weight: float = 0.15
    confidence_weight: float = 0.25


class DifficultyAssessor:
    def __init__(self, llm: LLMClient | None = None) -> None:
        self.weights = RouterWeights()
        self.llm = llm or LLMClient()

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

    @staticmethod
    def _normalize_capabilities(raw: Any) -> list[TaskCapability]:
        allowed = {item.value: item for item in TaskCapability}
        normalized: list[TaskCapability] = []
        seen: set[TaskCapability] = set()
        if isinstance(raw, str):
            raw_items: list[Any] = [raw]
        elif isinstance(raw, list):
            raw_items = raw
        else:
            raw_items = []

        for item in raw_items:
            value = str(item).strip().lower().replace("-", "_")
            cap = allowed.get(value)
            if cap is not None and cap not in seen:
                normalized.append(cap)
                seen.add(cap)
        return normalized

    def _infer_features_with_llm(
        self, query: str
    ) -> tuple[int, bool, float, list[TaskCapability]] | None:
        prompt = (
            "你是任务路由器，请根据用户请求估计复杂度特征。仅输出 JSON，不要解释。\n"
            "JSON schema:\n"
            "{\n"
            '  "estimated_steps": 1..6 的整数,\n'
            '  "needs_external_tools": true/false,\n'
            '  "confidence": 0..1 的小数,\n'
            '  "required_capabilities": ["retrieval"|"coding"|"filesystem"|"web"|"integration"|"skill"|"planning"|"utility"]\n'
            "}\n"
            "约束: required_capabilities 只描述完成任务所需能力，不要按关键词硬凑。\n"
            f"用户请求: {query}"
        )
        raw = self.llm.generate(prompt, temperature=0.0, profile=LLMClient.Profile.FAST)
        if raw.startswith("[Fallback-LLM]"):
            return None
        payload = self._extract_json_object(raw)
        if payload is None:
            return None

        try:
            estimated_steps = int(payload.get("estimated_steps", 1))
        except Exception:
            estimated_steps = 1
        estimated_steps = max(1, min(estimated_steps, 6))

        needs_external_tools = bool(payload.get("needs_external_tools", False))
        try:
            llm_confidence = float(payload.get("confidence", 0.7))
        except Exception:
            llm_confidence = 0.7
        llm_confidence = max(0.05, min(llm_confidence, 0.95))
        capabilities = self._normalize_capabilities(
            payload.get("required_capabilities", [])
        )
        return estimated_steps, needs_external_tools, llm_confidence, capabilities

    @staticmethod
    def _fallback_structural_features(
        query: str,
    ) -> tuple[int, bool, float, list[TaskCapability]]:
        text = query.strip()
        if not text:
            return 1, False, 0.7, [TaskCapability.PLANNING]

        clause_separators = sum(text.count(ch) for ch in [",", "，", ";", "；", "\n"])
        conjunctions = (
            text.count("并")
            + text.count("和")
            + text.count("及")
            + text.lower().count(" and ")
        )
        length_factor = 1 if len(text) >= 28 else 0
        lowered = text.lower()

        planning_markers = (
            "分析",
            "比较",
            "差异",
            "报告",
            "方案",
            "设计",
            "架构",
            "关键代码",
            "错误处理",
            "并发",
            "高效处理",
            "选型",
            "理由",
            "tradeoff",
            "architecture",
            "design",
            "report",
            "compare",
        )
        marker_hits = sum(1 for marker in planning_markers if marker in lowered)
        estimated_steps = (
            1 + clause_separators + conjunctions + length_factor + min(marker_hits, 3)
        )

        if any(
            marker in lowered
            for marker in ("报告", "系统", "架构", "关键代码", "architecture", "report")
        ):
            estimated_steps = max(estimated_steps, 5)
        elif marker_hits >= 2:
            estimated_steps = max(estimated_steps, 4)
        estimated_steps = max(1, min(6, estimated_steps))

        external_freshness_markers = (
            "现在",
            "当前",
            "最新",
            "2024",
            "2025",
            "2026",
            "trending",
            "github",
            "today",
            "current",
            "latest",
        )
        needs_external_tools = any(
            marker in lowered for marker in external_freshness_markers
        )
        capabilities: list[TaskCapability] = []

        code_markers = (
            "代码",
            "程序",
            "脚本",
            "函数",
            "实现",
            "python",
            "java",
            "javascript",
            "typescript",
            "cpp",
            "c++",
            "write code",
            "implement",
            "script",
            "function",
        )
        if any(marker in lowered for marker in code_markers):
            capabilities.append(TaskCapability.CODING)

        filesystem_markers = (
            "当前文件夹",
            "当前目录",
            "写入文件",
            "保存到",
            "读取文件",
            "列出目录",
            "本地文件",
            "workspace",
            "working directory",
            "write file",
            "save file",
            "read file",
            "list directory",
        )
        if any(marker in lowered for marker in filesystem_markers):
            capabilities.append(TaskCapability.FILESYSTEM)

        retrieval_markers = (
            "检索",
            "搜索",
            "查找",
            "资料",
            "经验",
            "之前",
            "上次",
            "刚才",
            "remember",
            "previous",
            "search",
            "retrieve",
            "lookup",
        )
        if any(marker in lowered for marker in retrieval_markers):
            capabilities.append(TaskCapability.RETRIEVAL)

        web_markers = (
            "网页",
            "网址",
            "url",
            "http",
            "https",
            "github",
            "trending",
            "最新",
            "当前",
            "2024",
            "2025",
            "2026",
            "today",
            "latest",
            "current",
        )
        if any(marker in lowered for marker in web_markers):
            capabilities.append(TaskCapability.WEB)

        integration_markers = (
            "api",
            "mcp",
            "集成",
            "endpoint",
            "webhook",
            "provider",
            "database",
            "数据库",
        )
        if any(marker in lowered for marker in integration_markers):
            capabilities.append(TaskCapability.INTEGRATION)

        if "skill" in lowered or "技能" in lowered:
            capabilities.append(TaskCapability.SKILL)

        if marker_hits > 0 or not capabilities:
            capabilities.append(TaskCapability.PLANNING)

        capabilities = list(dict.fromkeys(capabilities))
        base_confidence = 0.68
        return estimated_steps, needs_external_tools, base_confidence, capabilities

    @staticmethod
    def _escalate_one_level(difficulty: Difficulty) -> Difficulty:
        if difficulty == Difficulty.SIMPLE:
            return Difficulty.MEDIUM
        if difficulty == Difficulty.MEDIUM:
            return Difficulty.COMPLEX
        return Difficulty.COMPLEX

    @staticmethod
    def _fallback_plan_for(difficulty: Difficulty) -> str:
        if difficulty == Difficulty.SIMPLE:
            return (
                "若结果不完整，升级到 MEDIUM 路径（search-agent + code-agent）；"
                "仍失败则升级 COMPLEX 并启用 tool-agent。"
            )
        if difficulty == Difficulty.MEDIUM:
            return "若 MEDIUM 路径失败，升级 COMPLEX，启用并行调研、工具调用与代码方案联合求解。"
        return "若 COMPLEX 路径失败，触发确定性聚合兜底并保留失败信号用于后续修正。"

    def _decision_confidence(
        self, features: TaskFeatures, score: float, difficulty: Difficulty
    ) -> float:
        if difficulty == Difficulty.SIMPLE:
            margin = max(0.0, settings.simple_threshold - score)
            norm = min(1.0, margin / max(settings.simple_threshold, 1e-6))
        elif difficulty == Difficulty.MEDIUM:
            to_simple = abs(score - settings.simple_threshold)
            to_complex = abs(settings.medium_threshold - score)
            margin = min(to_simple, to_complex)
            norm = min(
                1.0,
                margin
                / max(settings.medium_threshold - settings.simple_threshold, 1e-6),
            )
        else:
            margin = max(0.0, score - settings.medium_threshold)
            norm = min(1.0, margin / max(1.0 - settings.medium_threshold, 1e-6))

        confidence = 0.6 * features.confidence + 0.4 * norm
        return max(0.05, min(confidence, 0.99))

    def extract_features(self, task: TaskRequest) -> TaskFeatures:
        # Removed keyword-list intent simulation; route signals are now primarily
        # inferred by LLM with a generic structural fallback when LLM is unavailable.
        inferred = self._infer_features_with_llm(task.query)
        if inferred is None:
            estimated_steps, needs_external_tools, confidence, required_capabilities = (
                self._fallback_structural_features(task.query)
            )
        else:
            estimated_steps, needs_external_tools, confidence, required_capabilities = (
                inferred
            )

        if required_capabilities:
            needs_external_tools = needs_external_tools or any(
                cap
                in {
                    TaskCapability.FILESYSTEM,
                    TaskCapability.WEB,
                    TaskCapability.INTEGRATION,
                    TaskCapability.SKILL,
                    TaskCapability.UTILITY,
                }
                for cap in required_capabilities
            )

        context = task.context or {}
        if "has_similar_success" in context:
            historical_pattern_score = (
                1.0 if bool(context.get("has_similar_success")) else 0.0
            )
        else:
            raw_episode_count = context.get("episode_count", 0)
            raw_success_rate = context.get("historical_success_rate", 0.0)
            try:
                episode_count = max(0, int(raw_episode_count))
            except Exception:
                episode_count = 0
            try:
                observed_success_rate = float(raw_success_rate)
            except Exception:
                observed_success_rate = 0.0

            prior = max(0.0, min(settings.router_cold_start_history_prior, 1.0))
            min_episodes = max(1, settings.router_cold_start_min_episodes)
            observed_success_rate = max(0.0, min(observed_success_rate, 1.0))

            if episode_count <= 0:
                historical_pattern_score = prior
            elif episode_count < min_episodes:
                alpha = episode_count / min_episodes
                historical_pattern_score = (
                    alpha * observed_success_rate + (1.0 - alpha) * prior
                )
            else:
                historical_pattern_score = observed_success_rate

        historical_pattern_score = max(0.0, min(historical_pattern_score, 1.0))
        has_historical_pattern = (
            historical_pattern_score >= settings.router_history_pattern_threshold
        )

        confidence = max(0.05, min(confidence, 0.95))
        if estimated_steps >= 5:
            confidence -= 0.12
        if needs_external_tools:
            confidence -= 0.1
        confidence += 0.1 * historical_pattern_score

        confidence = max(0.05, min(confidence, 0.95))
        return TaskFeatures(
            estimated_steps=estimated_steps,
            needs_external_tools=needs_external_tools,
            has_historical_pattern=has_historical_pattern,
            historical_pattern_score=historical_pattern_score,
            confidence=confidence,
            required_capabilities=required_capabilities,
        )

    def score(self, features: TaskFeatures) -> float:
        normalized_steps = min(features.estimated_steps / 6.0, 1.0)
        tool_flag = 1.0 if features.needs_external_tools else 0.0
        history_score = max(0.0, min(features.historical_pattern_score, 1.0))
        uncertainty = 1.0 - features.confidence

        score = (
            self.weights.step_weight * normalized_steps
            + self.weights.tool_weight * tool_flag
            + self.weights.history_weight * (1.0 - history_score)
            + self.weights.confidence_weight * uncertainty
        )
        return max(0.0, min(score, 1.0))

    def decide(self, task: TaskRequest) -> TaskDecision:
        features = self.extract_features(task)
        score = self.score(features)

        original_difficulty: Difficulty | None = None
        if score <= settings.simple_threshold:
            difficulty = Difficulty.SIMPLE
        elif score <= settings.medium_threshold:
            difficulty = Difficulty.MEDIUM
        else:
            difficulty = Difficulty.COMPLEX

        decision_confidence = self._decision_confidence(features, score, difficulty)
        escalated = False
        if decision_confidence < settings.router_low_confidence_threshold:
            original_difficulty = difficulty
            difficulty = self._escalate_one_level(difficulty)
            escalated = difficulty != original_difficulty

        reasoning = (
            f"steps={features.estimated_steps}, tools={features.needs_external_tools}, "
            f"capabilities={[cap.value for cap in features.required_capabilities]}, "
            f"history={features.has_historical_pattern}, history_score={features.historical_pattern_score:.2f}, "
            f"confidence={features.confidence:.2f}, "
            f"score={score:.2f}, decision_confidence={decision_confidence:.2f}, escalated={escalated}"
        )

        return TaskDecision(
            difficulty=difficulty,
            score=score,
            reasoning=reasoning,
            features=features,
            confidence=decision_confidence,
            fallback_plan=self._fallback_plan_for(difficulty),
            escalated_due_to_low_confidence=escalated,
            original_difficulty=original_difficulty,
        )

    def update_weights(self, error_signal: float) -> None:
        # error_signal > 0 表示路由偏保守，< 0 表示偏激进。
        step_delta = 0.02 * error_signal
        tool_delta = 0.015 * error_signal
        confidence_delta = -0.015 * error_signal

        self.weights.step_weight = min(
            0.6, max(0.1, self.weights.step_weight + step_delta)
        )
        self.weights.tool_weight = min(
            0.5, max(0.05, self.weights.tool_weight + tool_delta)
        )
        self.weights.confidence_weight = min(
            0.5, max(0.05, self.weights.confidence_weight + confidence_delta)
        )

        total = (
            self.weights.step_weight
            + self.weights.tool_weight
            + self.weights.history_weight
            + self.weights.confidence_weight
        )
        self.weights.step_weight /= total
        self.weights.tool_weight /= total
        self.weights.history_weight /= total
        self.weights.confidence_weight /= total
