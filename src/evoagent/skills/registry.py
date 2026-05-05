from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from sqlmodel import Field as SQLField
from sqlmodel import Session, SQLModel, create_engine, select


class SkillRecord(SQLModel, table=True):
    id: int | None = SQLField(default=None, primary_key=True)
    name: str = SQLField(index=True)
    version: str
    description: str
    tags: str
    success_rate: float = 0.5
    calls: int = 0
    status: str = "active"
    created_at: str = SQLField(index=True)


class SkillRegistry:
    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{db_path}")
        SQLModel.metadata.create_all(self.engine)

    def register(
        self,
        name: str,
        version: str,
        description: str,
        tags: list[str],
        status: str = "active",
    ) -> SkillRecord:
        record = SkillRecord(
            name=name,
            version=version,
            description=description,
            tags=",".join(tags),
            status=status,
            created_at=datetime.now(tz=timezone.utc).isoformat(),
        )
        with Session(self.engine) as session:
            if status == "active":
                stmt = (
                    select(SkillRecord)
                    .where(SkillRecord.name == name)
                    .where(SkillRecord.status == "active")
                )
                for existing in session.exec(stmt):
                    existing.status = "archived"
                    session.add(existing)
            session.add(record)
            session.commit()
            session.refresh(record)
            return record

    def has_active(self, name: str) -> bool:
        with Session(self.engine) as session:
            stmt = (
                select(SkillRecord)
                .where(SkillRecord.name == name)
                .where(SkillRecord.status == "active")
                .limit(1)
            )
            return session.exec(stmt).first() is not None

    def next_version(self, name: str) -> str:
        with Session(self.engine) as session:
            stmt = (
                select(SkillRecord)
                .where(SkillRecord.name == name)
                .order_by(SkillRecord.id.desc())
                .limit(1)
            )
            latest = session.exec(stmt).first()
            if not latest:
                return "0.1.0"
            try:
                major, minor, patch = latest.version.split(".")
                return f"{major}.{minor}.{int(patch) + 1}"
            except Exception:
                return "0.1.0"

    def search(self, keyword: str, limit: int = 10) -> list[SkillRecord]:
        with Session(self.engine) as session:
            stmt = (
                select(SkillRecord)
                .where(SkillRecord.status == "active")
                .where(
                    SkillRecord.description.contains(keyword)
                    | SkillRecord.tags.contains(keyword)
                )
                .limit(limit)
            )
            return list(session.exec(stmt))

    def update_score(self, skill_id: int, success: bool) -> None:
        with Session(self.engine) as session:
            record = session.get(SkillRecord, skill_id)
            if not record:
                return
            old_calls = record.calls
            old_rate = record.success_rate
            record.calls += 1
            record.success_rate = (
                (old_rate * old_calls) + (1.0 if success else 0.0)
            ) / record.calls
            session.add(record)
            session.commit()

    def update_version_score(
        self, name: str, version: str, success: bool
    ) -> tuple[int, float]:
        with Session(self.engine) as session:
            stmt = (
                select(SkillRecord)
                .where(SkillRecord.name == name)
                .where(SkillRecord.version == version)
                .limit(1)
            )
            record = session.exec(stmt).first()
            if record is None:
                return 0, 0.0
            old_calls = record.calls
            old_rate = record.success_rate
            record.calls += 1
            record.success_rate = (
                (old_rate * old_calls) + (1.0 if success else 0.0)
            ) / max(record.calls, 1)
            session.add(record)
            session.commit()
            return record.calls, record.success_rate

    def decay_and_archive(self, min_calls: int = 5, threshold: float = 0.3) -> int:
        archived = 0
        with Session(self.engine) as session:
            stmt = select(SkillRecord).where(SkillRecord.status == "active")
            records = list(session.exec(stmt))
            for record in records:
                if record.calls >= min_calls and record.success_rate < threshold:
                    record.status = "archived"
                    session.add(record)
                    archived += 1
            session.commit()
        return archived

    def list_skills(self, status: str = "active", limit: int = 20) -> list[SkillRecord]:
        with Session(self.engine) as session:
            stmt = (
                select(SkillRecord)
                .where(SkillRecord.status == status)
                .order_by(SkillRecord.id.desc())
                .limit(limit)
            )
            return list(session.exec(stmt))

    def activate_version(self, name: str, version: str) -> bool:
        updated = False
        with Session(self.engine) as session:
            stmt = select(SkillRecord).where(SkillRecord.name == name)
            records = list(session.exec(stmt))
            for record in records:
                if record.version == version:
                    record.status = "active"
                    updated = True
                else:
                    record.status = "archived"
                session.add(record)
            session.commit()
        return updated

    def set_version_status(self, name: str, version: str, status: str) -> bool:
        updated = False
        with Session(self.engine) as session:
            stmt = (
                select(SkillRecord)
                .where(SkillRecord.name == name)
                .where(SkillRecord.version == version)
                .limit(1)
            )
            record = session.exec(stmt).first()
            if record is None:
                return False
            record.status = status
            session.add(record)
            session.commit()
            updated = True
        return updated
