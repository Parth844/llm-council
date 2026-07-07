from council.caching import CachingClient
from tests.conftest import FakeClient


async def test_cache_hit_skips_inner_call(tmp_path):
    inner = FakeClient(responses={"m": "cached text\nFINAL ANSWER: 1"})
    client = CachingClient(inner, tmp_path)
    msgs = [{"role": "user", "content": "q"}]

    r1 = await client.chat("m", msgs)
    r2 = await client.chat("m", msgs)
    assert r1.text == r2.text
    assert len(inner.calls) == 1
    assert (client.hits, client.misses) == (1, 1)


async def test_cache_key_varies_by_model_and_prompt(tmp_path):
    inner = FakeClient()
    client = CachingClient(inner, tmp_path)
    await client.chat("m1", [{"role": "user", "content": "q"}])
    await client.chat("m2", [{"role": "user", "content": "q"}])
    await client.chat("m1", [{"role": "user", "content": "other"}])
    assert client.misses == 3
    assert client.hits == 0


async def test_cache_persists_across_instances(tmp_path):
    inner1 = FakeClient()
    await CachingClient(inner1, tmp_path).chat("m", [{"role": "user", "content": "q"}])
    inner2 = FakeClient(broken={"m"})  # would raise if actually called
    r = await CachingClient(inner2, tmp_path).chat("m", [{"role": "user", "content": "q"}])
    assert "FINAL ANSWER" in r.text
    assert inner2.calls == [] or all(c["model"] != "m" for c in inner2.calls)
