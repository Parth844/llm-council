"""Thin async client for NVIDIA NIM's OpenAI-compatible chat API.

Per-model sliding-window rate limiting, exponential backoff on 429/5xx.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from typing import Any

import httpx
from pydantic import BaseModel

logger = logging.getLogger("council.client")

RETRYABLE_STATUSES = {429, 500, 502, 503, 504}
MAX_RETRIES = 3


class ChatResult(BaseModel):
    model: str
    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_s: float


class ModelUnavailableError(Exception):
    """Raised when a model fails after all retries."""


class SlidingWindowRateLimiter:
    """Allows at most `limit` acquisitions per `window` seconds."""

    def __init__(self, limit: int, window: float = 60.0) -> None:
        self.limit = limit
        self.window = window
        self._timestamps: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        while True:
            async with self._lock:
                now = time.monotonic()
                while self._timestamps and now - self._timestamps[0] >= self.window:
                    self._timestamps.popleft()
                if len(self._timestamps) < self.limit:
                    self._timestamps.append(now)
                    return
                wait = self.window - (now - self._timestamps[0])
            await asyncio.sleep(max(wait, 0.05))


class NIMClient:
    """OpenAI-compatible chat/completions client with per-model rate limits."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://integrate.api.nvidia.com/v1",
        requests_per_minute: int = 40,
        timeout: float = 180.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.requests_per_minute = requests_per_minute
        self._limiters: dict[str, SlidingWindowRateLimiter] = {}
        self._models_cache: set[str] | None = None
        self._http = http_client or httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=15.0),
            headers={"Authorization": f"Bearer {api_key}"},
        )

    def _limiter(self, model: str) -> SlidingWindowRateLimiter:
        if model not in self._limiters:
            self._limiters[model] = SlidingWindowRateLimiter(self.requests_per_minute)
        return self._limiters[model]

    async def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> ChatResult:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES + 1):
            await self._limiter(model).acquire()
            start = time.monotonic()
            try:
                resp = await self._http.post(
                    f"{self.base_url}/chat/completions", json=payload
                )
            except httpx.HTTPError as exc:
                last_error = exc
                logger.warning("%s: network error (attempt %d): %s", model, attempt + 1, exc)
            else:
                if resp.status_code == 200:
                    data = resp.json()
                    usage = data.get("usage") or {}
                    return ChatResult(
                        model=model,
                        text=data["choices"][0]["message"]["content"] or "",
                        prompt_tokens=usage.get("prompt_tokens"),
                        completion_tokens=usage.get("completion_tokens"),
                        latency_s=time.monotonic() - start,
                    )
                if resp.status_code not in RETRYABLE_STATUSES:
                    raise ModelUnavailableError(
                        f"{model}: HTTP {resp.status_code}: {resp.text[:300]}"
                    )
                last_error = ModelUnavailableError(
                    f"{model}: HTTP {resp.status_code}: {resp.text[:300]}"
                )
                logger.warning("%s: HTTP %d (attempt %d)", model, resp.status_code, attempt + 1)
            if attempt < MAX_RETRIES:
                await asyncio.sleep(2.0**attempt)  # 1s, 2s, 4s
        raise ModelUnavailableError(f"{model}: failed after {MAX_RETRIES + 1} attempts") from last_error

    async def list_models(self) -> set[str] | None:
        """Model ids currently served (GET /models), cached for the client's
        lifetime. Returns None if the catalog can't be fetched — callers
        should then assume models are alive and let per-round errors handle it."""
        if self._models_cache is not None:
            return self._models_cache
        try:
            resp = await self._http.get(f"{self.base_url}/models")
            resp.raise_for_status()
            self._models_cache = {m["id"] for m in resp.json()["data"]}
            return self._models_cache
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
            logger.warning("could not fetch model catalog: %s", exc)
            return None

    async def aclose(self) -> None:
        await self._http.aclose()
