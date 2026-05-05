from __future__ import annotations

import asyncio
import logging
import time
from tempfile import TemporaryDirectory
from pathlib import Path

from evoagent.core.config import settings
from evoagent.sandbox.docker_sandbox import DockerSandbox, SandboxResult
from evoagent.skills.sandbox import SkillSandbox

logger = logging.getLogger(__name__)


class ASTSandbox(SkillSandbox):
    async def run_code(self, code: str, language: str = "python") -> SandboxResult:
        if language != "python":
            return SandboxResult(
                stdout="",
                stderr=f"unsupported language: {language}",
                exit_code=2,
                duration_ms=0,
                timed_out=False,
            )
        start = time.perf_counter()
        with TemporaryDirectory(prefix="evoagent_ast_") as tmp:
            path = Path(tmp) / "snippet.py"
            try:
                self.assert_code_safe(code)
            except Exception as exc:
                return SandboxResult(
                    stdout="",
                    stderr=str(exc),
                    exit_code=2,
                    duration_ms=int((time.perf_counter() - start) * 1000),
                    timed_out=False,
                )
            path.write_text(code, encoding="utf-8")
            proc = await asyncio.create_subprocess_exec(
                "python",
                str(path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=self.timeout_seconds
                )
                timed_out = False
            except asyncio.TimeoutError:
                proc.kill()
                stdout, stderr = await proc.communicate()
                timed_out = True
            return SandboxResult(
                stdout=stdout.decode("utf-8", errors="replace"),
                stderr=stderr.decode("utf-8", errors="replace"),
                exit_code=proc.returncode if proc.returncode is not None else 124,
                duration_ms=int((time.perf_counter() - start) * 1000),
                timed_out=timed_out,
            )

    async def run_pytest(self, test_code: str) -> SandboxResult:
        ok, output = self.validate("", test_code)
        return SandboxResult(
            stdout=output if ok else "",
            stderr="" if ok else output,
            exit_code=0 if ok else 1,
            duration_ms=0,
            timed_out=False,
        )


def docker_available() -> bool:
    try:
        import docker

        docker.from_env().ping()
        return True
    except Exception:
        return False


def get_sandbox() -> DockerSandbox | ASTSandbox:
    backend = settings.sandbox_backend.strip().lower()
    if backend == "ast":
        return ASTSandbox()
    if backend in {"docker", "auto"} and docker_available():
        return DockerSandbox()
    if backend == "docker":
        logger.warning(
            "Docker sandbox requested but Docker is unavailable; falling back to AST sandbox"
        )
    return ASTSandbox()
