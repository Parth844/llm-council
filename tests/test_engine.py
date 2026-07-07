import random

from council.engine import DebateEngine
from tests.conftest import FakeClient


async def test_full_debate_flow(config):
    client = FakeClient()
    engine = DebateEngine(config, client, rng=random.Random(0))
    result = await engine.run("What is 6*7?", rounds=2)

    assert set(result.answers) == {"Analyst A", "Analyst B", "Analyst C"}
    assert result.synthesis is not None
    assert result.failed_models == []
    types = [e.type for e in result.events]
    assert types.count("round_started") == 2
    assert types.count("model_answered") == 3
    assert types.count("critique") == 3
    assert types[-2:] == ["synthesis", "done"]


async def test_cross_exam_prompts_are_anonymized(config):
    client = FakeClient(
        responses={
            "vendor/m-a": "alpha\nFINAL ANSWER: 1",
            "vendor/m-b": "beta\nFINAL ANSWER: 2",
            "vendor/m-c": "gamma\nFINAL ANSWER: 3",
            "vendor/justice": "verdict\nFINAL ANSWER: 1",
        }
    )
    engine = DebateEngine(config, client, rng=random.Random(0))
    await engine.run("q", rounds=2)

    round2_calls = [c for c in client.calls if "Peer answers to review" in c["messages"][1]["content"]]
    assert len(round2_calls) == 3
    for call in round2_calls:
        body = call["messages"][1]["content"]
        # no brand/model ids and no stable aliases leak into cross-exam prompts
        assert "vendor/" not in body
        assert "Analyst" not in body
        assert "Peer " in body
        # each model sees exactly the 2 others
        assert body.count("--- Answer from Peer") == 2


async def test_council_continues_when_one_model_fails(config):
    client = FakeClient(broken={"vendor/m-b"})
    engine = DebateEngine(config, client, rng=random.Random(0))
    result = await engine.run("q", rounds=2)

    assert "vendor/m-b" in result.failed_models
    assert set(result.answers) == {"Analyst A", "Analyst C"}
    assert result.synthesis is not None


async def test_aborts_when_fewer_than_two_models(config):
    client = FakeClient(broken={"vendor/m-a", "vendor/m-b"})
    engine = DebateEngine(config, client, rng=random.Random(0))
    result = await engine.run("q", rounds=2)

    assert result.synthesis is None
    assert any(e.type == "error" for e in result.events)


async def test_chief_justice_fallback(config):
    client = FakeClient(broken={"vendor/justice"})
    engine = DebateEngine(config, client, rng=random.Random(0))
    result = await engine.run("q", rounds=2)

    assert result.synthesis is not None
    synth = [e for e in result.events if e.type == "synthesis"][0]
    assert synth.model_id == "vendor/justice-2"


async def test_single_round_skips_critiques(config):
    client = FakeClient()
    engine = DebateEngine(config, client, rng=random.Random(0))
    result = await engine.run("q", rounds=1)

    types = [e.type for e in result.events]
    assert types.count("round_started") == 1
    assert types.count("critique") == 0
    assert result.synthesis is not None
