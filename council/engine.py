"""Debate engine: independent answers → cross-examination rounds → synthesis."""

from __future__ import annotations

import asyncio
import logging
import random
import string
import time
import uuid
from datetime import datetime, timezone
from typing import Awaitable, Callable, Literal

from pydantic import BaseModel, Field

from council import prompts
from council.client import ChatResult, ModelUnavailableError, NIMClient
from council.config import CouncilConfig, ModelConfig

logger = logging.getLogger("council.engine")

EventType = Literal[
    "round_started",
    "model_answered",
    "critique",
    "synthesis",
    "model_failed",
    "done",
    "error",
]


class DebateEvent(BaseModel):
    type: EventType
    session_id: str
    round: int | None = None
    alias: str | None = None
    model_id: str | None = None  # kept in traces; the UI shows only aliases
    text: str | None = None
    reasoning: str | None = None  # model's thinking trace, when emitted
    prompt: str | None = None  # user prompt sent to the model (trace/replay)
    final_answer: str | None = None
    latency_s: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class DebateResult(BaseModel):
    session_id: str
    question: str
    rounds: int
    answers: dict[str, str]  # stable alias -> final answer text
    critiques: dict[str, str]  # stable alias -> last critique text
    synthesis: str | None
    failed_models: list[str]
    events: list[DebateEvent]


EventCallback = Callable[[DebateEvent], Awaitable[None] | None]


def make_pseudonym_map(aliases: list[str], rng: random.Random) -> dict[str, str]:
    """Fresh shuffled labels for one cross-exam round: stable alias -> pseudonym.

    Pseudonyms are 'Peer A', 'Peer B', ... assigned in shuffled order so a
    model can neither identify peers by brand nor track them across rounds.
    """
    letters = list(string.ascii_uppercase[: len(aliases)])
    rng.shuffle(letters)
    return {alias: f"Peer {letter}" for alias, letter in zip(aliases, letters)}


class DebateEngine:
    def __init__(
        self,
        config: CouncilConfig,
        client: NIMClient,
        on_event: EventCallback | None = None,
        rng: random.Random | None = None,
        skip_health_check: bool = False,
    ) -> None:
        self.config = config
        self.client = client
        self.on_event = on_event
        self.rng = rng or random.Random()
        self.skip_health_check = skip_health_check
        self._events: list[DebateEvent] = []

    async def _emit(self, event: DebateEvent) -> None:
        self._events.append(event)
        if self.on_event is not None:
            result = self.on_event(event)
            if asyncio.iscoroutine(result):
                await result

    async def _ask(self, model: ModelConfig, system: str, user: str) -> ChatResult:
        return await self.client.chat(
            model.id,
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=model.temperature,
            max_tokens=model.max_tokens,
            top_p=model.top_p,
            extra_body=model.extra_body,
        )

    async def _select_alive(self, models: list[ModelConfig]) -> list[ModelConfig]:
        """Filter to models present in the NIM catalog (one GET /models call).

        If the catalog is unreachable, assume all alive — transient per-model
        failures are handled (with retries) inside each round anyway.
        """
        if self.skip_health_check:
            return list(models)
        available = await self.client.list_models()
        if available is None:
            return list(models)
        return [m for m in models if m.id in available]

    async def run(
        self, question: str, rounds: int = 2, session_id: str | None = None
    ) -> DebateResult:
        session_id = session_id or uuid.uuid4().hex[:12]
        self._events = []
        failed: list[str] = []

        council = await self._select_alive(self.config.council)
        for m in self.config.council:
            if m not in council:
                failed.append(m.id)
                await self._emit(
                    DebateEvent(
                        type="model_failed",
                        session_id=session_id,
                        alias=m.alias,
                        model_id=m.id,
                        text="not in the NIM model catalog; excluded from council",
                    )
                )
        if len(council) < 2:
            await self._emit(
                DebateEvent(
                    type="error",
                    session_id=session_id,
                    text=f"only {len(council)} council model(s) available; need >= 2",
                )
            )
            return DebateResult(
                session_id=session_id,
                question=question,
                rounds=rounds,
                answers={},
                critiques={},
                synthesis=None,
                failed_models=failed,
                events=self._events,
            )

        answers: dict[str, str] = {}  # stable alias -> latest answer
        critiques: dict[str, str] = {}

        # --- Round 1: independent answers, fanned out in parallel ---
        await self._emit(
            DebateEvent(type="round_started", session_id=session_id, round=1)
        )

        async def _round1(model: ModelConfig) -> tuple[ModelConfig, ChatResult | Exception]:
            try:
                return model, await self._ask(
                    model, prompts.ROUND1_SYSTEM, prompts.round1_user(question)
                )
            except ModelUnavailableError as exc:
                return model, exc

        results = await asyncio.gather(*(_round1(m) for m in council))
        active: list[ModelConfig] = []
        for model, res in results:
            if isinstance(res, Exception):
                failed.append(model.id)
                await self._emit(
                    DebateEvent(
                        type="model_failed",
                        session_id=session_id,
                        round=1,
                        alias=model.alias,
                        model_id=model.id,
                        text=str(res),
                    )
                )
                continue
            active.append(model)
            answers[model.alias] = res.text
            await self._emit(
                DebateEvent(
                    type="model_answered",
                    session_id=session_id,
                    round=1,
                    alias=model.alias,
                    model_id=model.id,
                    text=res.text,
                    reasoning=res.reasoning,
                    prompt=prompts.round1_user(question),
                    latency_s=res.latency_s,
                    prompt_tokens=res.prompt_tokens,
                    completion_tokens=res.completion_tokens,
                )
            )

        if len(active) < 2:
            await self._emit(
                DebateEvent(
                    type="error",
                    session_id=session_id,
                    text=f"only {len(active)} model(s) answered round 1; need >= 2",
                )
            )
            return DebateResult(
                session_id=session_id,
                question=question,
                rounds=rounds,
                answers=answers,
                critiques={},
                synthesis=None,
                failed_models=failed,
                events=self._events,
            )

        # --- Rounds 2..N: anonymized cross-examination ---
        for round_no in range(2, rounds + 1):
            await self._emit(
                DebateEvent(
                    type="round_started", session_id=session_id, round=round_no
                )
            )
            pseudonyms = make_pseudonym_map([m.alias for m in active], self.rng)

            async def _critique(
                model: ModelConfig,
            ) -> tuple[ModelConfig, str, ChatResult | Exception]:
                peers = {
                    pseudonyms[m.alias]: answers[m.alias]
                    for m in active
                    if m.alias != model.alias
                }
                user = prompts.critique_user(question, answers[model.alias], peers)
                try:
                    return model, user, await self._ask(model, prompts.CRITIQUE_SYSTEM, user)
                except ModelUnavailableError as exc:
                    return model, user, exc

            results = await asyncio.gather(*(_critique(m) for m in active))
            still_active: list[ModelConfig] = []
            for model, user, res in results:
                if isinstance(res, Exception):
                    failed.append(model.id)
                    await self._emit(
                        DebateEvent(
                            type="model_failed",
                            session_id=session_id,
                            round=round_no,
                            alias=model.alias,
                            model_id=model.id,
                            text=str(res),
                        )
                    )
                    continue  # keeps its round-1 answer in `answers`
                still_active.append(model)
                answers[model.alias] = res.text
                critiques[model.alias] = res.text
                await self._emit(
                    DebateEvent(
                        type="critique",
                        session_id=session_id,
                        round=round_no,
                        alias=model.alias,
                        model_id=model.id,
                        text=res.text,
                        reasoning=res.reasoning,
                        prompt=user,
                        latency_s=res.latency_s,
                        prompt_tokens=res.prompt_tokens,
                        completion_tokens=res.completion_tokens,
                    )
                )
            if len(still_active) >= 2:
                active = still_active

        # --- Synthesis by the Chief Justice (with fallbacks) ---
        synthesis_text: str | None = None
        user = prompts.synthesis_user(question, answers, critiques)
        for justice in self.config.justices:
            try:
                res = await self._ask(justice, prompts.SYNTHESIS_SYSTEM, user)
            except ModelUnavailableError as exc:
                failed.append(justice.id)
                await self._emit(
                    DebateEvent(
                        type="model_failed",
                        session_id=session_id,
                        alias=justice.alias,
                        model_id=justice.id,
                        text=str(exc),
                    )
                )
                continue
            synthesis_text = res.text
            await self._emit(
                DebateEvent(
                    type="synthesis",
                    session_id=session_id,
                    alias=justice.alias,
                    model_id=justice.id,
                    text=res.text,
                    reasoning=res.reasoning,
                    prompt=user,
                    latency_s=res.latency_s,
                    prompt_tokens=res.prompt_tokens,
                    completion_tokens=res.completion_tokens,
                )
            )
            break
        if synthesis_text is None:
            await self._emit(
                DebateEvent(
                    type="error",
                    session_id=session_id,
                    text="all chief justice models failed; no synthesis produced",
                )
            )

        await self._emit(DebateEvent(type="done", session_id=session_id))
        return DebateResult(
            session_id=session_id,
            question=question,
            rounds=rounds,
            answers=answers,
            critiques=critiques,
            synthesis=synthesis_text,
            failed_models=failed,
            events=self._events,
        )
