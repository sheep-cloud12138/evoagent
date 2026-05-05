from __future__ import annotations

import argparse
import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Any


PROMPTS: list[tuple[str, str]] = [
    ("simple", "现在几点？"),
    ("simple", "用 Python 写一个快速排序"),
    ("simple", "解释一下什么是 TCP 三次握手"),
    ("medium", "帮我分析一下 React 和 Vue 现在哪个更值得学，给出理由"),
    ("medium", "写一个能处理并发请求的 Python 爬虫，要有错误处理"),
    ("medium", "我有一个 1000 万行的 CSV 文件，怎么高效处理"),
    ("complex", "帮我做一份 2024 年大模型领域的技术发展报告"),
    ("complex", "我要做一个个人博客系统，帮我设计技术方案并给出关键代码"),
    ("complex", "分析一下 AutoGen 和 LangGraph 的架构差异，哪个更适合做研究型 Agent"),
    ("hardcode", "帮我写两数之和的 Java 版本"),
    ("hardcode", "帮我实现冒泡排序，但要求支持自定义比较器"),
    ("hardcode", "写一个 GitHub trending 爬虫但只看 Rust 项目"),
]


def _configure_isolated_runtime() -> Path:
    base = Path(tempfile.mkdtemp(prefix="evoagent_eval_"))
    os.environ["EVO_DB_PATH"] = str(base / "evoagent.db")
    os.environ["EVO_SEMANTIC_PATH"] = str(base / "semantic_store")
    os.environ["EVO_SKILL_STORE_PATH"] = str(base / "skills_store")
    os.environ["OBSERVABILITY_LOG_PATH"] = str(base / "observability" / "events.jsonl")
    os.environ["OBSERVABILITY_METRICS_PATH"] = str(base / "observability" / "metrics.json")
    os.environ["OBSERVABILITY_STDOUT"] = "false"
    return base


def _tool_events(result: Any) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for output in result.outputs:
        metadata = output.metadata or {}
        raw_events = metadata.get("tool_events")
        if not isinstance(raw_events, list):
            continue
        events.extend(item for item in raw_events if isinstance(item, dict))
    return events


def _write_rows(json_out: Path | None, rows: list[dict[str, Any]]) -> None:
    if json_out is None:
        return
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


async def _run_eval(json_out: Path | None, case_timeout: float) -> list[dict[str, Any]]:
    from evoagent.app import EvoAgentSystem

    system = EvoAgentSystem()
    rows: list[dict[str, Any]] = []
    for idx, (category, prompt) in enumerate(PROMPTS, start=1):
        print(f"\n===== CASE {idx:02d} [{category}] {prompt}", flush=True)
        try:
            result = await asyncio.wait_for(
                system.run(prompt, context={"session_id": f"eval-{idx:02d}"}),
                timeout=max(1.0, case_timeout),
            )
        except TimeoutError:
            row = {
                "idx": idx,
                "category": category,
                "prompt": prompt,
                "mode": "case_timeout",
                "difficulty": "unknown",
                "score": 0.0,
                "confidence": 0.0,
                "quality_score": 0.0,
                "latency_seconds": round(case_timeout, 2),
                "outputs": 0,
                "tool_count": 0,
                "tool_names": [],
                "fallback": True,
                "answer_chars": 0,
                "answer_preview": f"case timed out after {case_timeout:.1f}s",
            }
            rows.append(row)
            _write_rows(json_out, rows)
            print(json.dumps(row, ensure_ascii=False, indent=2), flush=True)
            continue
        tool_events = _tool_events(result)
        answer = result.final_answer.strip()
        row = {
            "idx": idx,
            "category": category,
            "prompt": prompt,
            "mode": result.metadata.get("mode", "orchestrated"),
            "difficulty": result.decision.difficulty.value,
            "score": round(result.decision.score, 3),
            "confidence": round(result.decision.confidence, 3),
            "quality_score": round(result.quality_score, 3),
            "latency_seconds": round(result.latency_seconds, 2),
            "outputs": len(result.outputs),
            "tool_count": len(tool_events),
            "tool_names": sorted({str(event.get("name", "")) for event in tool_events}),
            "fallback": "[Fallback-LLM]" in answer,
            "answer_chars": len(answer),
            "answer_preview": answer[:900],
        }
        rows.append(row)
        _write_rows(json_out, rows)
        print(json.dumps(row, ensure_ascii=False, indent=2), flush=True)

    if json_out is not None:
        print(f"\nRESULT_PATH={json_out}", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EvoAgent black-box prompt evaluation.")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional path for compact JSON results.")
    parser.add_argument(
        "--use-default-state",
        action="store_true",
        help="Use configured data paths instead of an isolated temporary runtime.",
    )
    parser.add_argument(
        "--case-timeout",
        type=float,
        default=120.0,
        help="Maximum seconds to spend on one prompt before recording a timeout row.",
    )
    args = parser.parse_args()

    if not args.use_default_state:
        base = _configure_isolated_runtime()
        if args.json_out is None:
            args.json_out = base / "eval_results.json"
        print(f"ISOLATED_RUNTIME={base}", flush=True)

    asyncio.run(_run_eval(args.json_out, args.case_timeout))


if __name__ == "__main__":
    main()
