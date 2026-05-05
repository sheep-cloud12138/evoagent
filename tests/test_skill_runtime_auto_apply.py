import asyncio

from evoagent.app import EvoAgentSystem
from evoagent.core.tools import execute_tool, resolve_tool_names
from evoagent.skills.runtime import SkillArtifactManager


def test_auto_apply_evolved_skill(tmp_path) -> None:
    system = EvoAgentSystem()
    system.skill_artifacts = SkillArtifactManager(tmp_path / "skills_store")
    system.skill_engine.artifacts = system.skill_artifacts
    # Manually persist a skill artifact to simulate an evolved skill.
    draft = type(
        "Draft",
        (),
        {
            "name": "github_hot_skills",
            "description": "Fetch github hot skills list",
            "tags": ["github", "skills", "hot"],
            "code": "def execute(input_text: str) -> str:\n    return 'AUTO_SKILL_OK'\n",
            "tests": "from skill_impl import execute\n\ndef test_x():\n    assert execute('a') == 'AUTO_SKILL_OK'\n",
        },
    )()
    system.skill_artifacts.persist(draft, "0.1.0")

    result = asyncio.run(system.run("查询 github skills 热门 列表"))
    assert result.metadata.get("mode") == "auto_evolved_skill"
    assert "AUTO_SKILL_OK" in result.final_answer


def test_runtime_blocks_unsafe_persisted_skill(tmp_path) -> None:
    artifacts = SkillArtifactManager(tmp_path / "skills_store")
    draft = type(
        "Draft",
        (),
        {
            "name": "unsafe_skill",
            "description": "Unsafe persisted skill",
            "tags": ["unsafe"],
            "code": "import os\n\ndef execute(input_text: str) -> str:\n    return os.getcwd()\n",
            "tests": "from skill_impl import execute\n\ndef test_x():\n    assert execute('a')\n",
        },
    )()
    artifacts.persist(draft, "0.1.0")
    skill, score = artifacts.find_relevant("unsafe skill")
    assert skill is not None
    assert score > 0
    try:
        artifacts.execute(skill, "unsafe skill")
    except RuntimeError as exc:
        assert "Unsafe skill artifact blocked" in str(exc)
    else:
        raise AssertionError("unsafe skill should be blocked")


def test_active_skill_is_registered_as_tool(tmp_path) -> None:
    artifacts = SkillArtifactManager(tmp_path / "skills_store")
    draft = type(
        "Draft",
        (),
        {
            "name": "echo_skill_unique",
            "description": "Echo input text as a reusable tool",
            "tags": ["echo", "tool"],
            "code": "def execute(input_text: str) -> str:\n    return 'ECHO:' + input_text\n",
            "tests": "from skill_impl import execute\n\ndef test_x():\n    assert execute('a') == 'ECHO:a'\n",
        },
    )()
    artifacts.persist(draft, "0.1.0")
    tool_names = artifacts.register_active_skill_tools()

    assert "skill_echo_skill_unique_0_1_0" in tool_names
    assert "skill_echo_skill_unique_0_1_0" in resolve_tool_names(categories=("skill",), min_success_rate=0.0)
    assert execute_tool("skill_echo_skill_unique_0_1_0", {"input_text": "abc"}) == "ECHO:abc"


def test_archived_skill_tool_is_removed_when_new_active_version_exists(tmp_path) -> None:
    artifacts = SkillArtifactManager(tmp_path / "skills_store")
    first = type(
        "Draft",
        (),
        {
            "name": "versioned_skill_unique",
            "description": "Versioned reusable skill",
            "tags": ["versioned", "tool"],
            "code": "def execute(input_text: str) -> str:\n    return 'OLD:' + input_text\n",
            "tests": "from skill_impl import execute\n\ndef test_x():\n    assert execute('a') == 'OLD:a'\n",
        },
    )()
    second = type(
        "Draft",
        (),
        {
            "name": "versioned_skill_unique",
            "description": "Versioned reusable skill",
            "tags": ["versioned", "tool"],
            "code": "def execute(input_text: str) -> str:\n    return 'NEW:' + input_text\n",
            "tests": "from skill_impl import execute\n\ndef test_x():\n    assert execute('a') == 'NEW:a'\n",
        },
    )()

    artifacts.persist(first, "0.1.0")
    artifacts.persist(second, "0.2.0")

    selected = resolve_tool_names(categories=("skill",), min_success_rate=0.0)
    assert "skill_versioned_skill_unique_0_1_0" not in selected
    assert "skill_versioned_skill_unique_0_2_0" in selected
    assert execute_tool("skill_versioned_skill_unique_0_2_0", {"input_text": "abc"}) == "NEW:abc"


def test_chinese_query_can_match_chinese_skill_metadata(tmp_path) -> None:
    artifacts = SkillArtifactManager(tmp_path / "skills_store")
    draft = type(
        "Draft",
        (),
        {
            "name": "weather_query_helper",
            "description": "天气查询和气温分析工具",
            "tags": ["天气", "气温", "查询"],
            "code": "def execute(input_text: str) -> str:\n    return 'WEATHER:' + input_text\n",
            "tests": "from skill_impl import execute\n\ndef test_x():\n    assert execute('珠海天气').startswith('WEATHER:')\n",
        },
    )()
    artifacts.persist(draft, "0.1.0")

    skill, score = artifacts.find_relevant("请帮我查询珠海天气和气温")
    assert skill is not None
    assert skill["name"] == "weather_query_helper"
    assert score > 0
