import asyncio
import json

import httpx
import pytest

import council.api as api_mod
from council.tracing import TraceLogger
from tests.conftest import FakeClient


@pytest.fixture
async def client(config, tmp_path, monkeypatch):
    monkeypatch.setattr(api_mod, "load_config", lambda: config)
    monkeypatch.setattr(api_mod, "load_api_key", lambda: "nvapi-test")
    monkeypatch.setattr(api_mod, "NIMClient", lambda **kw: FakeClient())
    monkeypatch.setattr(api_mod, "TraceLogger", lambda: TraceLogger(tmp_path))
    async with api_mod.lifespan(api_mod.app):
        transport = httpx.ASGITransport(app=api_mod.app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as http:
            yield http


async def test_ask_then_stream_then_trace(client):
    resp = await client.post("/ask", json={"question": "2+2?", "rounds": 2})
    assert resp.status_code == 200
    session_id = resp.json()["session_id"]

    events = []
    async with client.stream("GET", f"/stream/{session_id}") as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        async for line in resp.aiter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))

    types = [e["type"] for e in events]
    assert types[0] == "round_started"
    assert types.count("model_answered") == 3
    assert types.count("critique") == 3
    assert "synthesis" in types
    assert types[-1] == "done"
    # aliases only in stream payloads for answers
    answered = [e for e in events if e["type"] == "model_answered"]
    assert all(e["alias"].startswith("Analyst") for e in answered)

    # give save_result a tick, then fetch the persisted trace
    await asyncio.sleep(0.05)
    resp = await client.get(f"/sessions/{session_id}")
    assert resp.status_code == 200
    trace = resp.json()
    assert trace["question"] == "2+2?"
    assert trace["synthesis"] is not None
    assert len(trace["events"]) == len(events)


async def test_stream_unknown_session_404(client):
    resp = await client.get("/stream/nope")
    assert resp.status_code == 404


async def test_late_subscriber_gets_replay(client):
    resp = await client.post("/ask", json={"question": "q", "rounds": 1})
    session_id = resp.json()["session_id"]
    await asyncio.sleep(0.2)  # let the debate finish first

    events = []
    async with client.stream("GET", f"/stream/{session_id}") as resp:
        async for line in resp.aiter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
    assert [e["type"] for e in events][-1] == "done"
    assert any(e["type"] == "synthesis" for e in events)
