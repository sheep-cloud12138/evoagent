from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


BLOCKED_IMPORTS = {
    "builtins",
    "ctypes",
    "httpx",
    "importlib",
    "multiprocessing",
    "os",
    "pathlib",
    "requests",
    "shutil",
    "socket",
    "subprocess",
    "sys",
    "threading",
    "urllib",
}
BLOCKED_CALLS = {
    "__import__",
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "input",
    "open",
}
BLOCKED_ATTRIBUTES = {
    "__base__",
    "__bases__",
    "__class__",
    "__closure__",
    "__code__",
    "__dict__",
    "__globals__",
    "__mro__",
    "__subclasses__",
}


class SandboxValidationError(RuntimeError):
    pass


class SkillSandbox:
    def __init__(self, timeout_seconds: int = 15) -> None:
        self.timeout_seconds = timeout_seconds

    def assert_code_safe(self, code: str) -> None:
        self._ast_safety_check(code)

    def _ast_safety_check(self, code: str) -> None:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for n in node.names:
                    if n.name.split(".")[0] in BLOCKED_IMPORTS:
                        raise SandboxValidationError(f"Blocked import: {n.name}")
            if isinstance(node, ast.ImportFrom):
                if node.module and node.module.split(".")[0] in BLOCKED_IMPORTS:
                    raise SandboxValidationError(f"Blocked import: {node.module}")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in BLOCKED_CALLS:
                    raise SandboxValidationError(f"Blocked call: {node.func.id}")
            if isinstance(node, ast.Attribute) and node.attr in BLOCKED_ATTRIBUTES:
                raise SandboxValidationError(f"Blocked attribute: {node.attr}")

    def validate(self, skill_code: str, test_code: str) -> tuple[bool, str]:
        try:
            self._ast_safety_check(skill_code)
            self._ast_safety_check(test_code)
        except (SandboxValidationError, SyntaxError) as exc:
            return False, str(exc)

        with TemporaryDirectory(prefix="evo_skill_") as tmp:
            path = Path(tmp)
            (path / "skill_impl.py").write_text(skill_code, encoding="utf-8")
            (path / "test_skill.py").write_text(test_code, encoding="utf-8")

            cmd = [
                sys.executable,
                "-I",
                "-m",
                "pytest",
                "-q",
                str(path / "test_skill.py"),
            ]
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(path),
                env={
                    "PATH": os.environ.get("PATH", ""),
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONUTF8": "1",
                },
                timeout=self.timeout_seconds,
            )
            ok = proc.returncode == 0
            output = (proc.stdout + "\n" + proc.stderr).strip()
            return ok, output
