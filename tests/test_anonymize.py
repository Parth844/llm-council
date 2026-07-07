import random

from council.engine import make_pseudonym_map


def test_pseudonyms_are_bijective_and_branded_names_absent():
    aliases = ["Analyst A", "Analyst B", "Analyst C"]
    mapping = make_pseudonym_map(aliases, random.Random(1))
    assert set(mapping.keys()) == set(aliases)
    assert len(set(mapping.values())) == 3
    assert all(v.startswith("Peer ") for v in mapping.values())


def test_mapping_shuffles_between_rounds():
    aliases = [f"Analyst {c}" for c in "ABCDEF"]
    rng = random.Random(42)
    seen = {tuple(sorted(make_pseudonym_map(aliases, rng).items())) for _ in range(20)}
    assert len(seen) > 1  # not the same assignment every round
