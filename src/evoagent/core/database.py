from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from sqlalchemy.engine import Engine
from sqlmodel import Session, SQLModel, create_engine

from evoagent.core.config import settings

# Import SQLModel table classes so create_all() sees the runtime tables.
from evoagent.models import Artifact, Run, Step, ToolCall  # noqa: F401
from evoagent.skills.manifest import SkillManifest  # noqa: F401


def _database_url(db_path: Path | str | None = None) -> str:
    path = Path(db_path or settings.db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{path}"


def create_db_engine(
    db_path: Path | str | None = None, *, echo: bool = False
) -> Engine:
    return create_engine(
        _database_url(db_path),
        echo=echo,
        connect_args={"check_same_thread": False},
    )


engine = create_db_engine()


def create_all(db_engine: Engine | None = None) -> None:
    SQLModel.metadata.create_all(db_engine or engine)


def init_db(db_engine: Engine | None = None) -> None:
    create_all(db_engine)


def get_session() -> Iterator[Session]:
    with Session(engine) as session:
        yield session
