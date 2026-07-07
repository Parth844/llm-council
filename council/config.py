"""Config loading and validation for the model registry."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "models.yaml"


class ModelConfig(BaseModel):
    id: str
    alias: str
    role: Literal["council", "chief_justice"]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, gt=0)
    enabled: bool = True


class CouncilConfig(BaseModel):
    base_url: str = "https://integrate.api.nvidia.com/v1"
    requests_per_minute: int = Field(default=40, gt=0)
    models: list[ModelConfig]

    @field_validator("models")
    @classmethod
    def _validate_roster(cls, models: list[ModelConfig]) -> list[ModelConfig]:
        enabled = [m for m in models if m.enabled]
        if len([m for m in enabled if m.role == "council"]) < 2:
            raise ValueError("need at least 2 enabled council models")
        if not any(m.role == "chief_justice" for m in enabled):
            raise ValueError("need at least 1 enabled chief_justice model")
        return models

    @property
    def council(self) -> list[ModelConfig]:
        return [m for m in self.models if m.enabled and m.role == "council"]

    @property
    def justices(self) -> list[ModelConfig]:
        """Chief justice candidates in fallback order."""
        return [m for m in self.models if m.enabled and m.role == "chief_justice"]


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> CouncilConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return CouncilConfig.model_validate(raw)


def load_api_key() -> str:
    load_dotenv()
    key = os.environ.get("NVIDIA_API_KEY", "")
    if not key.startswith("nvapi-"):
        raise RuntimeError(
            "NVIDIA_API_KEY missing or malformed (must start with 'nvapi-'). "
            "Copy .env.example to .env and set your key from build.nvidia.com."
        )
    return key
