from __future__ import annotations

import pytest

from evoagent.sandbox import docker_available, get_sandbox
from evoagent.sandbox.docker_sandbox import DockerSandbox, SandboxResult


def test_sandbox_result_schema_correct() -> None:
    result = SandboxResult(stdout="out", stderr="", exit_code=0, duration_ms=1, timed_out=False)
    assert result.success is True
    assert result.model_dump()["stdout"] == "out"


@pytest.mark.skipif(not docker_available(), reason="Docker not available")
async def test_timeout_is_enforced() -> None:
    result = await DockerSandbox(timeout=1).run_code("while True: pass")
    assert result.timed_out is True


@pytest.mark.skipif(not docker_available(), reason="Docker not available")
async def test_network_access_blocked() -> None:
    code = "import urllib.request; urllib.request.urlopen('https://example.com', timeout=2)"
    result = await DockerSandbox(timeout=5).run_code(code)
    assert result.success is False


def test_fallback_to_ast_sandbox_when_docker_unavailable(monkeypatch) -> None:
    monkeypatch.setattr("evoagent.sandbox.docker_available", lambda: False)
    monkeypatch.setattr("evoagent.sandbox.settings.sandbox_backend", "auto")
    assert get_sandbox().__class__.__name__ == "ASTSandbox"
