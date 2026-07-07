import pytest
from pydantic import ValidationError

from council.config import DEFAULT_CONFIG_PATH, CouncilConfig, load_config


def test_load_default_config():
    cfg = load_config(DEFAULT_CONFIG_PATH)
    assert len(cfg.council) >= 2
    assert len(cfg.justices) >= 1
    assert cfg.requests_per_minute == 40


def test_rejects_too_few_council_models():
    with pytest.raises(ValidationError):
        CouncilConfig.model_validate(
            {
                "models": [
                    {"id": "a", "alias": "A", "role": "council"},
                    {"id": "j", "alias": "J", "role": "chief_justice"},
                ]
            }
        )


def test_disabled_models_excluded():
    cfg = CouncilConfig.model_validate(
        {
            "models": [
                {"id": "a", "alias": "A", "role": "council"},
                {"id": "b", "alias": "B", "role": "council"},
                {"id": "c", "alias": "C", "role": "council", "enabled": False},
                {"id": "j", "alias": "J", "role": "chief_justice"},
            ]
        }
    )
    assert [m.id for m in cfg.council] == ["a", "b"]
