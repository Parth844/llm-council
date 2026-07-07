from __future__ import annotations

import pytest

from council.client import ChatResult, ModelUnavailableError
from council.config import CouncilConfig


class FakeClient:
    """Stands in for NIMClient. Responses keyed by model id; a model id in
    `broken` raises ModelUnavailableError. Records every call."""

    def __init__(
        self,
        responses: dict[str, str] | None = None,
        broken: set[str] | None = None,
    ) -> None:
        self.responses = responses or {}
        self.broken = broken or set()
        self.calls: list[dict] = []

    async def chat(self, model, messages, temperature=0.7, max_tokens=2048) -> ChatResult:
        self.calls.append({"model": model, "messages": messages})
        if model in self.broken:
            raise ModelUnavailableError(f"{model}: down")
        return ChatResult(
            model=model,
            text=self.responses.get(model, f"answer from {model}\nFINAL ANSWER: 42"),
            latency_s=0.01,
        )

    async def health_check(self, model: str) -> bool:
        return model not in self.broken


@pytest.fixture
def config() -> CouncilConfig:
    return CouncilConfig.model_validate(
        {
            "models": [
                {"id": "vendor/m-a", "alias": "Analyst A", "role": "council"},
                {"id": "vendor/m-b", "alias": "Analyst B", "role": "council"},
                {"id": "vendor/m-c", "alias": "Analyst C", "role": "council"},
                {"id": "vendor/justice", "alias": "Chief Justice", "role": "chief_justice"},
                {"id": "vendor/justice-2", "alias": "CJ fallback", "role": "chief_justice"},
            ]
        }
    )
