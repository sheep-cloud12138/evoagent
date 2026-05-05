from evoagent.skills.runtime import SkillArtifactManager


class _Draft:
    def __init__(self, name: str, code: str, tests: str) -> None:
        self.name = name
        self.description = "desc"
        self.tags = ["x"]
        self.code = code
        self.tests = tests


def test_activate_version(tmp_path) -> None:
    mgr = SkillArtifactManager(tmp_path / "store")
    d = _Draft(
        "my_skill",
        "def execute(input_text: str) -> str:\n    return 'v'\n",
        "from skill_impl import execute\n\ndef test_x():\n    assert execute('a') == 'v'\n",
    )
    mgr.persist(d, "0.1.0")
    mgr.persist(d, "0.1.1")

    ok = mgr.activate_version("my_skill", "0.1.0")
    assert ok
    versions = mgr.list_versions("my_skill")
    active = [v for v in versions if v.get("status") == "active"]
    assert len(active) == 1
    assert active[0]["version"] == "0.1.0"
