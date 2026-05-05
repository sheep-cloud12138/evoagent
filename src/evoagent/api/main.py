from __future__ import annotations

import asyncio
import json
from uuid import uuid4

from fastapi import BackgroundTasks, Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlmodel import Session, select

from evoagent import __version__
from evoagent.api.deps import get_db_session, get_runtime
from evoagent.core.runtime import AgentRuntime
from evoagent.models import Artifact, Run, RunStatus, Step

app = FastAPI(title="EvoAgent API", version=__version__)

# TODO: Add authentication before exposing this gateway beyond localhost/dev networks.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://127.0.0.1",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RunCreateRequest(BaseModel):
    query: str
    metadata: dict = Field(default_factory=dict)


async def _run_background(
    runtime: AgentRuntime, query: str, run_id: str, metadata: dict
) -> None:
    try:
        await runtime.run(query, run_id=run_id, context=metadata)
    except TypeError as exc:
        if "unexpected keyword argument 'context'" not in str(exc):
            raise
        await runtime.run(query, run_id=run_id)


def _run_to_dict(run: Run) -> dict:
    return {
        "run_id": run.run_id,
        "user_query": run.user_query,
        "status": run.status.value if hasattr(run.status, "value") else str(run.status),
        "plan": run.plan,
        "final_answer": run.final_answer,
        "error": run.error,
        "started_at": run.started_at,
        "ended_at": run.ended_at,
        "metadata": run.metadata,
    }


@app.post("/runs")
async def create_run(
    payload: RunCreateRequest,
    background_tasks: BackgroundTasks,
    runtime: AgentRuntime = Depends(get_runtime),
) -> dict:
    run_id = str(uuid4())
    runtime._create_run(payload.query, run_id)
    if payload.metadata:
        runtime._set_run_status(run_id, RunStatus.running, metadata=payload.metadata)
    background_tasks.add_task(
        _run_background, runtime, payload.query, run_id, payload.metadata
    )
    return {"run_id": run_id, "status": "running"}


@app.get("/runs/{run_id}")
def get_run(run_id: str, session: Session = Depends(get_db_session)) -> dict:
    run = session.get(Run, run_id)
    if run is None:
        return {"run_id": run_id, "status": "not_found"}
    return _run_to_dict(run)


@app.get("/runs/{run_id}/steps")
def get_steps(run_id: str, session: Session = Depends(get_db_session)) -> list[dict]:
    rows = session.exec(
        select(Step).where(Step.run_id == run_id).order_by(Step.sequence)
    ).all()
    return [row.model_dump(mode="json") for row in rows]


@app.get("/runs/{run_id}/artifacts")
def get_artifacts(
    run_id: str, session: Session = Depends(get_db_session)
) -> list[dict]:
    rows = session.exec(
        select(Artifact).where(Artifact.run_id == run_id).order_by(Artifact.created_at)
    ).all()
    return [row.model_dump(mode="json", by_alias=True) for row in rows]


@app.post("/runs/{run_id}/cancel")
def cancel_run(run_id: str, runtime: AgentRuntime = Depends(get_runtime)) -> dict:
    run = runtime._set_run_status(run_id, RunStatus.cancelled)
    return {"run_id": run.run_id, "status": run.status.value}


@app.get("/runs/{run_id}/stream")
async def stream_run(run_id: str, session: Session = Depends(get_db_session)):
    async def event_stream():
        seen_steps: set[str] = set()
        while True:
            steps = session.exec(
                select(Step).where(Step.run_id == run_id).order_by(Step.sequence)
            ).all()
            for step in steps:
                if step.step_id in seen_steps:
                    continue
                seen_steps.add(step.step_id)
                event = (
                    "step_completed"
                    if step.status.value in {"succeeded", "failed", "skipped"}
                    else "step_started"
                )
                yield f"event: {event}\ndata: {json.dumps(step.model_dump(mode='json'), ensure_ascii=False)}\n\n"
            run = session.get(Run, run_id)
            if run is None:
                yield f"event: error\ndata: {json.dumps({'error': 'run not found'})}\n\n"
                break
            if run.status in {
                RunStatus.succeeded,
                RunStatus.failed,
                RunStatus.cancelled,
            }:
                yield f"event: run_completed\ndata: {json.dumps(_run_to_dict(run), ensure_ascii=False, default=str)}\n\n"
                break
            await asyncio.sleep(0.5)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}
