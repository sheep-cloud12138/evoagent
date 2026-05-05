from evoagent.llm.adapter import completion, litellm
from evoagent.llm.service import LLMService

LLMClient = LLMService

__all__ = ["LLMClient", "completion", "litellm"]
