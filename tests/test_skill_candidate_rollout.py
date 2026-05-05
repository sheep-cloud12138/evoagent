import asyncio

from evoagent.app import EvoAgentSystem
from evoagent.core.config import settings
from evoagent.skills.registry import SkillRegistry
from evoagent.skills.runtime import SkillArtifactManager


class _Draft:
    def __init__(self, name: str, ret: str) -> None:
        self.name = name
        self.description = "Fetch github hot skills list"
        self.tags = ["github", "skills", "hot"]
        self.code = f"def execute(input_text: str) -> str:\n    return '{ret}'\n"
        self.tests = (
            "from skill_impl import execute\n\n"
            "def test_x():\n"
            f"    assert execute('a') == '{ret}'\n"
        )


def test_candidate_promotes_after_stable_trials(monkeypatch, tmp_path) -> None:
    system = EvoAgentSystem()
    system.skill_registry = SkillRegistry(tmp_path / "db.sqlite")
    system.skill_artifacts = SkillArtifactManager(tmp_path / "store")
    system.skill_engine.registry = system.skill_registry
    system.skill_engine.artifacts = system.skill_artifacts

    active = _Draft("github_hot_skills", "ACTIVE")
    candidate = _Draft("github_hot_skills", "CANDIDATE")

    system.skill_registry.register("github_hot_skills", "0.1.0", active.description, active.tags, status="active")
    system.skill_artifacts.persist(active, "0.1.0", status="active")
    system.skill_registry.register("github_hot_skills", "0.1.1", candidate.description, candidate.tags, status="candidate")
    system.skill_artifacts.persist(candidate, "0.1.1", status="candidate")

    monkeypatch.setattr(settings, "skill_candidate_rollout_rate", 1.0)
    monkeypatch.setattr(settings, "skill_candidate_min_calls", 1)
    monkeypatch.setattr(settings, "skill_candidate_promote_success_rate", 0.5)
    monkeypatch.setattr(settings, "skill_candidate_archive_success_rate", 0.1)
    monkeypatch.setattr(settings, "skill_ab_test_rate", 0.0)

    result = asyncio.run(system.run("查询 github skills 热门 列表"))
    assert result.metadata.get("mode") == "auto_evolved_skill"
    assert "CANDIDATE" in result.final_answer

    versions = system.skill_artifacts.list_versions("github_hot_skills")
    active_versions = [v for v in versions if v.get("status") == "active"]
    assert len(active_versions) == 1
    assert active_versions[0]["version"] == "0.1.1"
