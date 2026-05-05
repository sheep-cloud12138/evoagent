from __future__ import annotations

import json
from typing import Any

from evoagent.core.config import settings
from evoagent.core.tools import execute_tool_with_meta
from evoagent.llm.adapter import ProviderAdapter


class ToolCallRunner:
    def __init__(self, adapter: ProviderAdapter | None = None) -> None:
        self.adapter = adapter or ProviderAdapter()

    @staticmethod
    def tool_call_id(call: Any, idx: int) -> str:
        return getattr(call, "id", None) or f"tool_call_{idx}"

    @staticmethod
    def tool_call_name(call: Any) -> str:
        fn = getattr(call, "function", None)
        if fn is None:
            return ""
        return getattr(fn, "name", "") or ""

    @staticmethod
    def tool_call_args(call: Any) -> dict[str, Any]:
        fn = getattr(call, "function", None)
        raw = getattr(fn, "arguments", "{}") if fn is not None else "{}"
        if isinstance(raw, dict):
            return raw
        try:
            parsed = json.loads(raw or "{}")
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}

    def run(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_rounds: int | None = None,
        **kwargs: Any,
    ) -> tuple[str, list[dict[str, Any]]]:
        tool_events: list[dict[str, Any]] = []
        rounds = max(1, int(max_rounds or settings.llm_tool_max_rounds))
        working_messages = list(messages)
        for _ in range(rounds):
            response = self.adapter.call(
                model=model,
                messages=working_messages,
                tools=tools,
                tool_choice="auto",
                **kwargs,
            )
            message = response.choices[0].message
            tool_calls = getattr(message, "tool_calls", None) or []
            content = getattr(message, "content", "") or ""
            if not tool_calls:
                return content.strip(), tool_events

            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": content,
                "tool_calls": [],
            }
            for idx, call in enumerate(tool_calls):
                tool_name = self.tool_call_name(call)
                call_id = self.tool_call_id(call, idx)
                args = self.tool_call_args(call)
                assistant_message["tool_calls"].append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(args, ensure_ascii=True),
                        },
                    }
                )
            working_messages.append(assistant_message)

            for idx, call in enumerate(tool_calls):
                tool_name = self.tool_call_name(call)
                call_id = self.tool_call_id(call, idx)
                args = self.tool_call_args(call)
                tool_meta = execute_tool_with_meta(tool_name, args)
                result_text = str(tool_meta.get("result", ""))
                tool_events.append(
                    {
                        "name": tool_meta.get("name", tool_name),
                        "source": tool_meta.get("source", "unknown"),
                        "category": tool_meta.get("category", "unknown"),
                        "success": bool(tool_meta.get("success", False)),
                        "error": str(tool_meta.get("error", ""))[:240],
                        "success_rate": float(tool_meta.get("success_rate", 0.0)),
                        "result": result_text[:4000],
                        "result_truncated": len(result_text) > 4000,
                    }
                )
                working_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": tool_name,
                        "content": str(tool_meta.get("result", "")),
                    }
                )

        response = self.adapter.call(model=model, messages=working_messages, **kwargs)
        return response.choices[0].message.content.strip(), tool_events
