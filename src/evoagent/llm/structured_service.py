from __future__ import annotations

import json
import logging
from typing import TypeVar

from pydantic import BaseModel

from evoagent.llm.service import LLMService

T = TypeVar("T", bound=BaseModel)
logger = logging.getLogger(__name__)


class StructuredLLMService:
    def __init__(self, llm: LLMService | None = None) -> None:
        self.llm = llm or LLMService()

    async def generate_structured(
        self,
        prompt: str,
        response_model: type[T],
        system: str | None = None,
    ) -> T:
        return self.generate_structured_sync(prompt, response_model, system=system)

    def generate_structured_sync(
        self,
        prompt: str,
        response_model: type[T],
        system: str | None = None,
    ) -> T:
        text = self._generate_text(prompt, system=system)
        return self._parse_text(text, response_model)

    def _generate_text(self, prompt: str, system: str | None = None) -> str:
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        try:
            import instructor  # noqa: F401
        except Exception:
            logger.warning(
                "instructor unavailable; falling back to manual JSON parsing"
            )
        return self.llm.generate(
            full_prompt, temperature=0.0, profile=LLMService.Profile.REASONING
        )

    @staticmethod
    def _extract_json_object(raw_text: str) -> dict:
        text = raw_text.strip()
        if not text:
            raise ValueError("empty structured response")
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("structured response did not contain a JSON object")
        parsed = json.loads(text[start : end + 1])
        if not isinstance(parsed, dict):
            raise ValueError("structured response root is not an object")
        return parsed

    def _parse_text(self, text: str, response_model: type[T]) -> T:
        payload = self._extract_json_object(text)
        return response_model.model_validate(payload)
