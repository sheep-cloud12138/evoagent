from __future__ import annotations

import asyncio
import time
from tempfile import TemporaryDirectory
from pathlib import Path

from pydantic import BaseModel, computed_field


class SandboxResult(BaseModel):
    stdout: str
    stderr: str
    exit_code: int
    duration_ms: int
    timed_out: bool

    @computed_field
    @property
    def success(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


class DockerSandbox:
    def __init__(
        self,
        image: str = "python:3.12-slim",
        timeout: int = 30,
        memory_limit: str = "256m",
        cpu_period: int = 100000,
        cpu_quota: int = 50000,
    ) -> None:
        self.image = image
        self.timeout = timeout
        self.memory_limit = memory_limit
        self.cpu_period = cpu_period
        self.cpu_quota = cpu_quota

    def _client(self):
        import docker

        client = docker.from_env()
        try:
            client.images.get(self.image)
        except Exception:
            client.images.pull(self.image)
        return client

    def _run_container(
        self,
        command: list[str],
        volumes: dict | None = None,
        workdir: str | None = None,
    ) -> SandboxResult:
        start = time.perf_counter()
        container = None
        try:
            client = self._client()
            container = client.containers.run(
                self.image,
                command=command,
                detach=True,
                mem_limit=self.memory_limit,
                cpu_period=self.cpu_period,
                cpu_quota=self.cpu_quota,
                network_disabled=True,
                read_only=True,
                remove=False,
                volumes=volumes,
                working_dir=workdir,
                tmpfs={"/tmp": "rw,noexec,nosuid,size=64m"},
                environment={"PYTHONDONTWRITEBYTECODE": "1", "PYTHONUTF8": "1"},
            )
            try:
                result = container.wait(timeout=self.timeout)
                exit_code = int(result.get("StatusCode", 1))
                timed_out = False
            except Exception:
                timed_out = True
                exit_code = 124
                try:
                    container.kill()
                except Exception:
                    pass
            stdout = container.logs(stdout=True, stderr=False).decode(
                "utf-8", errors="replace"
            )
            stderr = container.logs(stdout=False, stderr=True).decode(
                "utf-8", errors="replace"
            )
            return SandboxResult(
                stdout=stdout,
                stderr=stderr,
                exit_code=exit_code,
                duration_ms=int((time.perf_counter() - start) * 1000),
                timed_out=timed_out,
            )
        finally:
            if container is not None:
                try:
                    container.remove(force=True)
                except Exception:
                    pass

    async def run_code(self, code: str, language: str = "python") -> SandboxResult:
        if language != "python":
            return SandboxResult(
                stdout="",
                stderr=f"unsupported language: {language}",
                exit_code=2,
                duration_ms=0,
                timed_out=False,
            )
        return await asyncio.to_thread(
            self._run_container, ["python", "-c", code], None, None
        )

    async def run_pytest(self, test_code: str) -> SandboxResult:
        with TemporaryDirectory(prefix="evoagent_pytest_") as tmp:
            path = Path(tmp)
            test_file = path / "test_sandbox.py"
            test_file.write_text(test_code, encoding="utf-8")
            volumes = {str(path): {"bind": "/workspace", "mode": "ro"}}
            return await asyncio.to_thread(
                self._run_container,
                ["python", "-m", "pytest", "-q", "/workspace/test_sandbox.py"],
                volumes,
                "/workspace",
            )

    def validate(self, skill_code: str, test_code: str) -> tuple[bool, str]:
        source = (
            f"{skill_code}\n\n{test_code.replace('from skill_impl import execute', '')}"
        )
        try:
            result = asyncio.run(self.run_pytest(source))
        except Exception as exc:
            return False, str(exc)
        return result.success, (result.stdout + "\n" + result.stderr).strip()
