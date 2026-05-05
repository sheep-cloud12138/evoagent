from __future__ import annotations

import builtins

from evoagent.llm.service import LLMService
from evoagent.llm.structured import Plan
from evoagent.llm.structured_service import StructuredLLMService


class _LLM(LLMService):
    def generate(self, *args, **kwargs) -> str:
        return (
            '{"reasoning":"ok","steps":[{"sequence":1,"role":"researcher",'
            '"description":"d","depends_on":[],"expected_output":"o"}]}'
        )


def test_structured_service_returns_correct_pydantic_type() -> None:
    result = StructuredLLMService(_LLM()).generate_structured_sync("prompt", Plan)
    assert isinstance(result, Plan)
    assert result.steps[0].role == "researcher"


def test_fallback_to_manual_parse_when_instructor_unavailable(monkeypatch) -> None:
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "instructor":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    result = StructuredLLMService(_LLM()).generate_structured_sync("prompt", Plan)
    assert result.reasoning == "ok"


def test_plan_model_validates_correctly() -> None:
    plan = Plan.model_validate(
        {
            "reasoning": "ok",
            "steps": [
                {
                    "sequence": 1,
                    "role": "coder",
                    "description": "write code",
                    "depends_on": [],
                    "expected_output": "code",
                }
            ],
        }
    )
    assert plan.steps[0].role == "coder"
