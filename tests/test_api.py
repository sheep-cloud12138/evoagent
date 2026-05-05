from __future__ import annotations

from fastapi.testclient import TestClient

from evoagent.api import deps
from evoagent.api.main import app
from evoagent.core.runtime import AgentRuntime
from evoagent.models import RunStatus


async def _fake_run(self, query: str, run_id: str | None = None):
    return self._set_run_status(run_id, status=RunStatus.succeeded, final_answer="ok")


def test_post_runs_returns_run_id(monkeypatch) -> None:
    runtime = AgentRuntime()
    monkeypatch.setattr(AgentRuntime, "run", _fake_run)
    app.dependency_overrides[deps.get_runtime] = lambda: runtime
    client = TestClient(app)
    response = client.post("/runs", json={"query": "hello"})
    assert response.status_code == 200
    assert response.json()["run_id"]
    app.dependency_overrides.clear()


def test_get_run_returns_correct_status(monkeypatch) -> None:
    runtime = AgentRuntime()
    monkeypatch.setattr(AgentRuntime, "run", _fake_run)
    app.dependency_overrides[deps.get_runtime] = lambda: runtime
    client = TestClient(app)
    run_id = client.post("/runs", json={"query": "hello"}).json()["run_id"]
    response = client.get(f"/runs/{run_id}")
    assert response.status_code == 200
    assert response.json()["status"] in {"running", "succeeded"}
    app.dependency_overrides.clear()


def test_get_run_steps_returns_steps_list(monkeypatch) -> None:
    runtime = AgentRuntime()
    monkeypatch.setattr(AgentRuntime, "run", _fake_run)
    app.dependency_overrides[deps.get_runtime] = lambda: runtime
    client = TestClient(app)
    run_id = client.post("/runs", json={"query": "hello"}).json()["run_id"]
    response = client.get(f"/runs/{run_id}/steps")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    app.dependency_overrides.clear()
