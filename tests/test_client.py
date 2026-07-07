import json

import httpx
import pytest

from council.client import MAX_RETRIES, ModelUnavailableError, NIMClient


def make_client(handler) -> NIMClient:
    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(transport=transport, headers={"Authorization": "Bearer x"})
    return NIMClient(api_key="nvapi-test", http_client=http)


def ok_response(text: str = "hello") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": text}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        },
    )


async def test_chat_success():
    client = make_client(lambda req: ok_response("answer"))
    result = await client.chat("m1", [{"role": "user", "content": "q"}])
    assert result.text == "answer"
    assert result.prompt_tokens == 10
    assert result.completion_tokens == 5


async def test_retries_on_429_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(429, text="rate limited")
        return ok_response("finally")

    client = make_client(handler)
    monkeypatch.setattr("council.client.asyncio.sleep", _no_sleep)
    result = await client.chat("m1", [{"role": "user", "content": "q"}])
    assert result.text == "finally"
    assert calls["n"] == 3


async def test_gives_up_after_max_retries(monkeypatch):
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        return httpx.Response(503, text="down")

    client = make_client(handler)
    monkeypatch.setattr("council.client.asyncio.sleep", _no_sleep)
    with pytest.raises(ModelUnavailableError):
        await client.chat("m1", [{"role": "user", "content": "q"}])
    assert calls["n"] == MAX_RETRIES + 1


async def test_non_retryable_status_fails_immediately():
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        return httpx.Response(404, text="unknown model")

    client = make_client(handler)
    with pytest.raises(ModelUnavailableError):
        await client.chat("bad/model", [{"role": "user", "content": "q"}])
    assert calls["n"] == 1


async def test_top_p_extra_body_sent_and_reasoning_parsed():
    captured = {}

    def handler(req):
        captured.update(json.loads(req.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": "answer", "reasoning_content": "let me think"}}
                ],
                "usage": {},
            },
        )

    client = make_client(handler)
    result = await client.chat(
        "m1",
        [{"role": "user", "content": "q"}],
        top_p=0.95,
        extra_body={"chat_template_kwargs": {"thinking": True}},
    )
    assert captured["top_p"] == 0.95
    assert captured["chat_template_kwargs"] == {"thinking": True}
    assert result.reasoning == "let me think"
    assert result.text == "answer"


async def test_list_models_fetches_once_and_caches():
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        return httpx.Response(200, json={"data": [{"id": "a/x"}, {"id": "b/y"}]})

    client = make_client(handler)
    assert await client.list_models() == {"a/x", "b/y"}
    assert await client.list_models() == {"a/x", "b/y"}
    assert calls["n"] == 1


async def test_list_models_returns_none_on_error():
    client = make_client(lambda req: httpx.Response(500, text="boom"))
    assert await client.list_models() is None


async def _no_sleep(_seconds: float) -> None:
    return None
