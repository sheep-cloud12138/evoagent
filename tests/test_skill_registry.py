from evoagent.skills.registry import SkillRegistry


def test_register_active_archives_previous_active_version(tmp_path) -> None:
    registry = SkillRegistry(tmp_path / "skills.sqlite")

    registry.register(
        name="same_skill",
        version="0.1.0",
        description="first",
        tags=["demo"],
        status="active",
    )
    registry.register(
        name="same_skill",
        version="0.2.0",
        description="second",
        tags=["demo"],
        status="active",
    )

    active = registry.list_skills(status="active", limit=10)
    archived = registry.list_skills(status="archived", limit=10)

    assert [(item.name, item.version) for item in active] == [("same_skill", "0.2.0")]
    assert ("same_skill", "0.1.0") in [(item.name, item.version) for item in archived]
