from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any


def _collect_tool_events(result: Any) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for output in result.outputs:
        payload = output.metadata if isinstance(output.metadata, dict) else {}
        raw = payload.get("tool_events", [])
        if isinstance(raw, list):
            events.extend(item for item in raw if isinstance(item, dict))
    return events


def _load_cases(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("dataset must be a JSON list")
    rows: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        query = str(item.get("query", "")).strip()
        if not query:
            continue
        rows.append(item)
    return rows


async def _run(dataset: Path, json_out: Path | None, case_timeout: float) -> list[dict[str, Any]]:
    from evoagent.app import EvoAgentSystem

    cases = _load_cases(dataset)
    system = EvoAgentSystem()
    rows: list[dict[str, Any]] = []

    for idx, case in enumerate(cases, start=1):
        case_id = str(case.get("id", f"case-{idx:03d}"))
        query = str(case.get("query", "")).strip()
        expect = case.get("expect", {})
        expect_non_fallback = bool(expect.get("require_non_fallback", False)) if isinstance(expect, dict) else False
        min_answer_chars = int(expect.get("min_answer_chars", 0)) if isinstance(expect, dict) else 0

        try:
            result = await asyncio.wait_for(
                system.run(query, context={"session_id": f"search-e2e-{idx:03d}"}),
                timeout=max(1.0, case_timeout),
            )
            answer = result.final_answer.strip()
            events = _collect_tool_events(result)
            retrieval_events = [evt for evt in events if str(evt.get("category", "")) == "retrieval"]
            fallback = answer.startswith("[Fallback-LLM]")
            pass_flags = {
                "non_fallback": (not expect_non_fallback) or (not fallback),
                "min_answer_chars": len(answer) >= max(0, min_answer_chars),
            }
            passed = all(pass_flags.values())
            row = {
                "id": case_id,
                "query": query,
                "passed": passed,
                "pass_flags": pass_flags,
                "mode": result.metadata.get("mode", "orchestrated"),
                "difficulty": result.decision.difficulty.value,
                "latency_seconds": round(result.latency_seconds, 3),
                "quality_score": round(result.quality_score, 3),
                "tool_calls": len(events),
                "retrieval_tool_calls": len(retrieval_events),
                "tool_names": sorted({str(evt.get("name", "")) for evt in events}),
                "answer_chars": len(answer),
                "answer_preview": answer[:240],
            }
        except TimeoutError:
            row = {
                "id": case_id,
                "query": query,
                "passed": False,
                "pass_flags": {
                    "timeout": False,
                },
                "mode": "case_timeout",
                "difficulty": "unknown",
                "latency_seconds": round(case_timeout, 3),
                "quality_score": 0.0,
                "tool_calls": 0,
                "retrieval_tool_calls": 0,
                "tool_names": [],
                "answer_chars": 0,
                "answer_preview": "",
            }

        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

        if json_out is not None:
            json_out.parent.mkdir(parents=True, exist_ok=True)
            json_out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run end-to-end retrieval evaluation for EvoAgent search flow.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/observability/eval_search_e2e.json"),
        help="JSON dataset path",
    )
    parser.add_argument("--json-out", type=Path, default=None, help="Optional JSON output path")
    parser.add_argument("--case-timeout", type=float, default=120.0, help="Per-case timeout in seconds")
    args = parser.parse_args()

    rows = asyncio.run(_run(dataset=args.dataset, json_out=args.json_out, case_timeout=args.case_timeout))
    passed = sum(1 for row in rows if bool(row.get("passed", False)))
    total = len(rows)
    retrieval_calls = sum(int(row.get("retrieval_tool_calls", 0)) for row in rows)
    print(
        json.dumps(
            {
                "summary": {
                    "total": total,
                    "passed": passed,
                    "pass_rate": round((passed / max(1, total)), 3),
                    "retrieval_tool_calls": retrieval_calls,
                }
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
