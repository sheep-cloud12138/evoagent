from __future__ import annotations

from typing import Any


FALLBACK_MARKERS = (
    "[Fallback-LLM]",
    "[subagent-error]",
    "[subagent-timeout]",
)

INABILITY_MARKERS = (
    "无法读取",
    "无法访问",
    "无法直接访问",
    "无法获取",
    "不能访问",
    "未能读取",
    "没有读取",
    "请提供",
    "粘贴",
    "cannot access",
    "can't access",
    "unable to access",
    "could not read",
    "couldn't read",
    "no access",
    "please provide",
)


def content_is_fallback(text: str) -> bool:
    stripped = str(text or "").strip()
    return any(stripped.startswith(marker) for marker in FALLBACK_MARKERS)


def successful_tool_events_from_metadata(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    raw_events = metadata.get("tool_events", [])
    if not isinstance(raw_events, list):
        return []
    return [
        event
        for event in raw_events
        if isinstance(event, dict)
        and bool(event.get("success", False))
        and str(event.get("result", "")).strip()
    ]


def successful_tool_events_from_outputs(outputs: list[Any]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for output in outputs:
        metadata = getattr(output, "metadata", {})
        if isinstance(metadata, dict):
            events.extend(successful_tool_events_from_metadata(metadata))
    return events


def format_tool_evidence(
    events: list[dict[str, Any]],
    *,
    max_event_chars: int = 1800,
    max_total_chars: int = 8000,
) -> str:
    if not events:
        return ""

    chunks: list[str] = ["工具执行事实（真实工具返回，优先于模型自述）:"]
    total = len(chunks[0])
    for idx, event in enumerate(events, start=1):
        name = str(event.get("name", "unknown_tool"))
        category = str(event.get("category", "unknown"))
        source = str(event.get("source", "unknown"))
        result = str(event.get("result", "")).strip()
        if not result:
            continue
        if len(result) > max_event_chars:
            result = f"{result[:max_event_chars]}... [truncated]"
        chunk = f"{idx}. {name} [{category}/{source}]\n{result}"
        if total + len(chunk) > max_total_chars:
            chunks.append("... [tool evidence truncated]")
            break
        chunks.append(chunk)
        total += len(chunk)
    return "\n\n".join(chunks)


def answer_conflicts_with_successful_tools(
    answer: str, events: list[dict[str, Any]]
) -> bool:
    if not events:
        return False
    lowered = str(answer or "").lower()
    return any(marker in lowered for marker in INABILITY_MARKERS)
