from __future__ import annotations

import asyncio
import json
import threading
from typing import TYPE_CHECKING, Any

from evoagent.core.tools import execute_tool_with_meta
from evoagent.llm.service import LLMService
from evoagent.sandbox import get_sandbox

if TYPE_CHECKING:
    from evoagent.core.graph import AgentState
else:
    AgentState = dict


def _current_spec(state: AgentState) -> dict[str, Any]:
    meta = state.get("metadata", {}) if isinstance(state, dict) else {}
    spec = meta.get("current_step_spec", {}) if isinstance(meta, dict) else {}
    return spec if isinstance(spec, dict) else {}


def _description(state: AgentState) -> str:
    spec = _current_spec(state)
    return str(spec.get("description") or state.get("user_query", ""))


def _worker_result(
    role: str,
    output: str,
    artifacts: list | None = None,
    confidence: float = 0.7,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "role": role,
        "output": output,
        "artifacts": artifacts or [],
        "confidence": confidence,
        "error": error,
    }


def _run_async(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result_holder: dict[str, Any] = {}

    def runner() -> None:
        try:
            result_holder["result"] = asyncio.run(coro)
        except Exception as exc:  # pragma: no cover
            result_holder["error"] = exc

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()
    if "error" in result_holder:
        raise result_holder["error"]
    return result_holder.get("result")


def researcher_worker(state: AgentState, llm: LLMService) -> dict:
    try:
        prompt = f"Research the following task and return concise findings:\n{_description(state)}"
        output = llm.generate(
            prompt, temperature=0.0, profile=LLMService.Profile.STANDARD
        )
        return _worker_result(
            "researcher",
            output,
            [output],
            0.7 if not output.startswith("[Fallback-LLM]") else 0.2,
        )
    except Exception as exc:
        return _worker_result("researcher", "", [], 0.0, str(exc))


def coder_worker(state: AgentState, llm: LLMService) -> dict:
    try:
        prompt = (
            "Generate Python code for this step. Return only executable Python when possible.\n"
            f"Step: {_description(state)}"
        )
        code = llm.generate(
            prompt, temperature=0.0, profile=LLMService.Profile.STANDARD
        )
        if code.startswith("[Fallback-LLM]"):
            return _worker_result("coder", code, [code], 0.2)
        result = _run_async(get_sandbox().run_code(code, language="python"))
        payload = {
            "code": code,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.exit_code,
            "success": result.success,
        }
        confidence = 0.75 if result.success else 0.35
        return _worker_result(
            "coder",
            json.dumps(payload, ensure_ascii=False),
            [
                {"type": "code", "content": code},
                {"type": "tool_result", "content": result.model_dump_json()},
            ],
            confidence,
            None if result.success else result.stderr,
        )
    except Exception as exc:
        return _worker_result("coder", "", [], 0.0, str(exc))


def tool_worker(state: AgentState, llm: LLMService) -> dict:
    _ = llm
    try:
        spec = _current_spec(state)
        tool_name = str(spec.get("tool_name") or spec.get("tool") or "").strip()
        args = spec.get("input") or spec.get("args") or {}
        if not tool_name:
            text = _description(state)
            if text.strip().split():
                tool_name = text.strip().split()[0]
            else:
                tool_name = "unknown"
        if not isinstance(args, dict):
            args = {}
        result = execute_tool_with_meta(tool_name, args)
        output = str(result.get("result", ""))
        return _worker_result(
            "tool",
            output,
            [{"type": "tool_result", "content": output, "metadata": result}],
            0.75 if result.get("success") else 0.25,
            str(result.get("error")) if result.get("error") else None,
        )
    except Exception as exc:
        return _worker_result("tool", "", [], 0.0, str(exc))


def memory_worker(state: AgentState, llm: LLMService) -> dict:
    _ = llm
    try:
        result = execute_tool_with_meta(
            "semantic_memory_search",
            {"query": state.get("user_query", ""), "top_k": 5, "use_negative": False},
        )
        output = str(result.get("result", ""))
        return _worker_result(
            "memory",
            output,
            [{"type": "memory_ref", "content": output, "metadata": result}],
            0.7 if result.get("success") else 0.25,
            str(result.get("error")) if result.get("error") else None,
        )
    except Exception as exc:
        return _worker_result("memory", "", [], 0.0, str(exc))


def reviewer_worker(state: AgentState, llm: LLMService) -> dict:
    try:
        prompt = (
            "Review the previous worker outputs and provide assessment plus suggested revisions.\n"
            f"Outputs: {state.get('worker_outputs', [])}"
        )
        output = llm.generate(
            prompt, temperature=0.0, profile=LLMService.Profile.REASONING
        )
        return _worker_result(
            "reviewer",
            output,
            [output],
            0.7 if not output.startswith("[Fallback-LLM]") else 0.2,
        )
    except Exception as exc:
        return _worker_result("reviewer", "", [], 0.0, str(exc))
