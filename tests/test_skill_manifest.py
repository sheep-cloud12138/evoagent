from __future__ import annotations

import pytest

from evoagent.core.config import settings
from evoagent.skills.manifest import SkillManifest
from evoagent.skills.runtime import SkillManifestRuntime


def _manifest(skill_id: str, permissions: list[str] | None = None) -> SkillManifest:
    return SkillManifest(
        skill_id=skill_id,
        name=skill_id,
        version="1.0.0",
        description="demo",
        input_schema={},
        output_schema={},
        permissions=permissions or ["sandbox"],
        tags=["demo"],
        source_code="def execute(input_text: str) -> str: return input_text",
        test_code="def test_x(): assert True",
    )


def test_new_skill_starts_as_candidate(tmp_path) -> None:
    runtime = SkillManifestRuntime(tmp_path / "skills.db")
    manifest = runtime.register(_manifest("skill-a"))
    assert manifest.status == "candidate"


def test_skill_promoted_to_active_after_three_successes(tmp_path) -> None:
    runtime = SkillManifestRuntime(tmp_path / "skills.db")
    runtime.register(_manifest("skill-b"))
    runtime.add_result("skill-b", True)
    runtime.add_result("skill-b", True)
    manifest = runtime.add_result("skill-b", True)
    assert manifest is not None
    assert manifest.status == "active"
    assert runtime.get_skill("skill-b") is not None


def test_skill_rejected_after_failure_rate_over_half(tmp_path) -> None:
    runtime = SkillManifestRuntime(tmp_path / "skills.db")
    runtime.register(_manifest("skill-c"))
    manifest = runtime.add_result("skill-c", False)
    assert manifest is not None
    assert manifest.status == "rejected"


def test_permission_check_blocks_unauthorized_skills(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(settings, "skill_allowed_permissions", "sandbox,memory")
    runtime = SkillManifestRuntime(tmp_path / "skills.db")
    manifest = runtime.register(_manifest("skill-d", ["network"]))
    manifest.status = "active"
    runtime.register(manifest)
    with pytest.raises(PermissionError):
        runtime.get_skill("skill-d")


def test_only_active_skills_returned_by_runtime(tmp_path) -> None:
    runtime = SkillManifestRuntime(tmp_path / "skills.db")
    runtime.register(_manifest("skill-e"))
    assert runtime.get_skill("skill-e") is None
