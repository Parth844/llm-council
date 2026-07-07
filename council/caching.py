"""Disk-caching wrapper around a chat client, keyed by hash(model + prompt).

Used by the eval harness so re-runs cost zero API calls.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from council.client import ChatResult


class CachingClient:
    """Duck-types the NIMClient chat interface; caches responses on disk."""

    def __init__(self, inner, cache_dir: Path | str) -> None:
        self.inner = inner
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def _key(self, model: str, messages: list[dict[str, str]], **params) -> Path:
        blob = json.dumps(
            {"model": model, "messages": messages, **params}, sort_keys=True
        )
        return self.cache_dir / (hashlib.sha256(blob.encode()).hexdigest() + ".json")

    async def chat(self, model: str, messages: list[dict[str, str]],
                   temperature: float = 0.7, max_tokens: int = 2048,
                   top_p: float | None = None,
                   extra_body: dict | None = None) -> ChatResult:
        path = self._key(model, messages, temperature=temperature,
                         max_tokens=max_tokens, top_p=top_p, extra_body=extra_body)
        if path.exists():
            self.hits += 1
            return ChatResult.model_validate_json(path.read_text())
        result = await self.inner.chat(
            model, messages, temperature=temperature, max_tokens=max_tokens,
            top_p=top_p, extra_body=extra_body,
        )
        self.misses += 1
        path.write_text(result.model_dump_json())
        return result

    async def list_models(self) -> set[str] | None:
        return await self.inner.list_models()

    async def aclose(self) -> None:
        await self.inner.aclose()
