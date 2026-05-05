from evoagent.skills.sandbox import SkillSandbox


def test_sandbox_blocks_dangerous_import() -> None:
    sandbox = SkillSandbox()
    code = "import os\n\ndef execute(input_text: str) -> str:\n    return input_text"
    tests = "from skill_impl import execute\n\ndef test_x():\n    assert execute('a') == 'a'"
    ok, report = sandbox.validate(code, tests)
    assert not ok
    assert "Blocked import" in report


def test_sandbox_blocks_dangerous_builtin_call() -> None:
    sandbox = SkillSandbox()
    code = "def execute(input_text: str) -> str:\n    return eval(input_text)\n"
    tests = "from skill_impl import execute\n\ndef test_x():\n    assert execute('1 + 1') == 2"
    ok, report = sandbox.validate(code, tests)
    assert not ok
    assert "Blocked call" in report
