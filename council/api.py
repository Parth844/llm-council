"""FastAPI layer: POST /ask, SSE GET /stream/{id}, GET /sessions/{id}, static UI."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from council.client import NIMClient
from council.config import load_api_key, load_config
from council.engine import DebateEngine, DebateEvent
from council.tracing import TraceLogger

logger = logging.getLogger("council.api")

STATIC_DIR = Path(__file__).resolve().parent / "static"
STREAM_END = None  # queue sentinel


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=8000)
    rounds: int = Field(default=2, ge=1, le=5)


class AskResponse(BaseModel):
    session_id: str


class SessionState:
    def __init__(self) -> None:
        self.queues: list[asyncio.Queue[DebateEvent | None]] = []
        self.history: list[DebateEvent] = []  # replay for late subscribers
        self.done = False


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_config()
    app.state.config = config
    app.state.client = NIMClient(
        api_key=load_api_key(),
        base_url=config.base_url,
        requests_per_minute=config.requests_per_minute,
    )
    app.state.tracer = TraceLogger()
    app.state.sessions = {}
    app.state.tasks = set()
    yield
    await app.state.client.aclose()


app = FastAPI(title="LLM Council", lifespan=lifespan)


async def _run_debate(session_id: str, question: str, rounds: int) -> None:
    state: SessionState = app.state.sessions[session_id]
    tracer: TraceLogger = app.state.tracer

    async def on_event(event: DebateEvent) -> None:
        tracer.log_event(event)
        state.history.append(event)
        for queue in state.queues:
            queue.put_nowait(event)

    engine = DebateEngine(app.state.config, app.state.client, on_event=on_event)
    try:
        result = await engine.run(question, rounds=rounds, session_id=session_id)
        tracer.save_result(result)
    except Exception:
        logger.exception("debate %s crashed", session_id)
        await on_event(
            DebateEvent(type="error", session_id=session_id, text="internal error")
        )
    finally:
        state.done = True
        for queue in state.queues:
            queue.put_nowait(STREAM_END)


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    session_id = uuid.uuid4().hex[:12]
    app.state.sessions[session_id] = SessionState()
    task = asyncio.create_task(_run_debate(session_id, req.question, req.rounds))
    app.state.tasks.add(task)
    task.add_done_callback(app.state.tasks.discard)
    return AskResponse(session_id=session_id)


@app.get("/stream/{session_id}")
async def stream(session_id: str) -> StreamingResponse:
    state: SessionState | None = app.state.sessions.get(session_id)
    if state is None:
        raise HTTPException(404, "unknown session")

    queue: asyncio.Queue[DebateEvent | None] = asyncio.Queue()
    # replay events already emitted, then follow live
    for event in state.history:
        queue.put_nowait(event)
    if state.done:
        queue.put_nowait(STREAM_END)
    else:
        state.queues.append(queue)

    async def generate():
        try:
            while True:
                event = await queue.get()
                if event is STREAM_END:
                    break
                yield f"event: {event.type}\ndata: {event.model_dump_json()}\n\n"
        finally:
            if queue in state.queues:
                state.queues.remove(queue)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/sessions/{session_id}")
async def get_session(session_id: str) -> dict:
    trace = app.state.tracer.load_session(session_id)
    if trace is None:
        raise HTTPException(404, "unknown session")
    return trace


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
