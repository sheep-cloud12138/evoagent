from __future__ import annotations

from evoagent.agents.base import BaseSubAgent
from evoagent.core.llm import LLMClient


class SearchAgent(BaseSubAgent):
    name = "search-agent"
    model_profile = LLMClient.Profile.REASONING
    enable_function_call = True
    tool_categories = ("retrieval", "web", "integration", "utility")


class ReasoningAgent(BaseSubAgent):
    name = "reasoning-agent"
    model_profile = LLMClient.Profile.STANDARD


class CodeAgent(BaseSubAgent):
    name = "code-agent"
    model_profile = LLMClient.Profile.STANDARD


class FileAgent(BaseSubAgent):
    name = "file-agent"
    model_profile = LLMClient.Profile.FAST
    enable_function_call = True
    tool_categories = ("filesystem",)


class IntegrationAgent(BaseSubAgent):
    name = "integration-agent"
    model_profile = LLMClient.Profile.FAST
    enable_function_call = True
    tool_categories = ("integration", "web", "utility")


class SkillAgent(BaseSubAgent):
    name = "skill-agent"
    model_profile = LLMClient.Profile.FAST
    enable_function_call = True
    tool_categories = ("skill",)


class ToolAgent(BaseSubAgent):
    name = "tool-agent"
    model_profile = LLMClient.Profile.FAST
    enable_function_call = True
    tool_categories = ("utility", "web", "integration", "filesystem", "skill")
    # Removed task-specific filesystem planning/execution branches so tool usage
    # is decided by the generic BaseSubAgent function-calling flow via the LLM.
