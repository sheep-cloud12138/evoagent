from __future__ import annotations

from pathlib import Path

from evoagent.core.models import RunArtifact


class WorkspaceSandbox:
    """Local workspace boundary for future worker execution and artifacts."""

    def __init__(self, root_path: Path) -> None:
        self.root_path = root_path
        self.root_path.mkdir(parents=True, exist_ok=True)

    def create(self, run_id: str) -> Path:
        safe_id = (
            "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in run_id)
            or "run"
        )
        workspace = (self.root_path / safe_id).resolve()
        workspace.mkdir(parents=True, exist_ok=True)
        workspace.relative_to(self.root_path.resolve())
        return workspace

    def resolve(self, run_id: str, raw_path: str) -> Path:
        workspace = self.create(run_id)
        target = (workspace / raw_path).resolve()
        target.relative_to(workspace)
        return target

    def artifact(self, run_id: str) -> RunArtifact:
        workspace = self.create(run_id)
        return RunArtifact(
            artifact_id="workspace",
            type="workspace",
            name=workspace.name,
            uri=str(workspace),
            metadata={"root": str(self.root_path.resolve())},
        )
