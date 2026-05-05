from evoagent.core.llm import LLMClient
from evoagent.skills.evolution import SkillEvolutionEngine
from evoagent.skills.registry import SkillRegistry
from evoagent.skills.runtime import SkillArtifactManager
from evoagent.skills.sandbox import SkillSandbox


class _FakeLLM(LLMClient):
    def __init__(self, text: str) -> None:
        self._text = text

    def generate(self, prompt: str, temperature: float | None = None, profile: str = LLMClient.Profile.STANDARD) -> str:
        return self._text


def test_generate_skill_fallback_on_invalid_format(tmp_path) -> None:
    llm = _FakeLLM("random unstructured output")
    registry = SkillRegistry(tmp_path / "test.db")
    artifacts = SkillArtifactManager(tmp_path / "skills")
    engine = SkillEvolutionEngine(
        llm=llm,
        registry=registry,
        sandbox=SkillSandbox(),
        artifacts=artifacts,
    )
    draft = engine.generate_skill("x", "y")
    assert "def execute(" in draft.code
    assert "def test_" in draft.tests
