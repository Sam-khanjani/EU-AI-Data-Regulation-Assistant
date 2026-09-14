"""Groq client for every model call in the pipeline.

Deliberately the raw ``groq`` SDK rather than LangChain's ``ChatGroq``. The whole
trustworthiness argument rests on ``strict: true`` reaching the API on every answer call:
constrained decoding is what makes it structurally impossible for the model to emit a claim
without a ``supporting_quotes`` array. Routing that through a chat abstraction puts a layer
between us and the one flag we cannot afford to have silently dropped or reshaped between
library versions. LangGraph still orchestrates the graph; only the model call is direct.

Strict mode constraints that shape the rest of the system:

* **No streaming.** Answers are rendered whole; the UI streams pipeline *progress* instead.
* **No tool use.** Not needed -- retrieval happens in our code, not the model's.

**What strict mode actually guarantees.** The documentation describes constrained decoding,
but in practice Groq also validates after generation and returns
``400 json_validate_failed`` when the result does not conform -- we hit exactly that, with a
response that was valid JSON but omitted two required fields after the model ran long and
closed the object early. So the guarantee is "a non-conforming response is rejected", not
"a non-conforming response cannot be produced". :class:`SchemaValidationFailed` surfaces
that case distinctly so callers can retry rather than treat it as a hard failure.

None of this affects the trustworthiness argument, because the argument never rested on it:
the schema shapes the request, and ``euaia.verify.citations`` independently checks that
every quote appears in the source. A response that satisfies the schema perfectly and
invents its quotes still fails there.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from groq import APIStatusError, BadRequestError, Groq, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from euaia.config import settings
from euaia.llm.ratelimit import Limits, estimate_tokens, limiter_for

log = logging.getLogger(__name__)

# Below this an answer cannot say anything useful, so failing is more honest than
# issuing a request guaranteed to truncate.
MIN_OUTPUT_TOKENS = 800
# Slack for the message envelope our character-based estimate does not model.
_ENVELOPE_TOKENS = 128


class LLMError(RuntimeError):
    pass


class SchemaValidationFailed(LLMError):
    """Groq rejected the model's output as non-conforming (400 json_validate_failed).

    Usually means the model ran out of output budget and closed the object before emitting
    every required field. Worth one retry; not worth pretending it succeeded.
    """


@dataclass(slots=True)
class Usage:
    """Token accounting, aggregated across a request pipeline."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    latency_ms: int = 0
    waited_ms: int = 0
    """Time spent parked by the rate limiter, not by the API."""

    def add(self, other: Usage) -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.calls += other.calls
        self.latency_ms += other.latency_ms
        self.waited_ms += other.waited_ms

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(slots=True)
class Completion:
    """One structured response plus what it cost."""

    data: dict[str, Any]
    model: str
    usage: Usage = field(default_factory=Usage)
    finish_reason: str | None = None


class GroqClient:
    """Structured-output calls against Groq."""

    def __init__(self, api_key: str | None = None, limits: Limits | None = None) -> None:
        key = api_key or settings.groq_api_key
        if not key:
            raise LLMError(
                "GROQ_API_KEY is not set. Copy .env.example to .env and fill it in."
            )
        self._client = Groq(api_key=key)
        self._limits = limits or Limits(
            requests_per_minute=settings.groq_rpm,
            tokens_per_minute=settings.groq_tpm,
            tokens_per_day=settings.groq_tpd,
        )

    def structured(
        self,
        *,
        system: str,
        user: str,
        response_format: dict[str, Any],
        model: str | None = None,
        max_completion_tokens: int = 8192,
        temperature: float = 0.0,
        reasoning_effort: str | None = None,
    ) -> Completion:
        """Make one strict-JSON-schema call and return the parsed object.

        ``response_format`` comes from :mod:`euaia.llm.schemas`, already wrapped in Groq's
        ``json_schema`` envelope with ``strict: true``.
        """
        chosen = model or settings.groq_model
        limiter = limiter_for(chosen, self._limits)

        # Pace against the free tier before spending anything. The full output allowance
        # counts, not a guess at what will be used: tokens-per-minute counts what the model
        # *may* generate, and a 429 costs more than a slow minute.
        #
        # If prompt + allowance would not fit even an empty minute, shrink the allowance
        # rather than refuse. Callers already size their prompts, but two independent
        # estimates agreeing to the token is not something to rely on -- a request that
        # missed by five tokens used to fail outright. Adapting here means the caller's
        # sizing only has to be close.
        prompt_tokens = estimate_tokens(system, user)
        headroom = limiter.limits.usable_tpm - prompt_tokens - _ENVELOPE_TOKENS
        allowance = min(max_completion_tokens, headroom)
        if allowance < MIN_OUTPUT_TOKENS:
            raise LLMError(
                f"prompt is ~{prompt_tokens} tokens, leaving only {headroom} of "
                f"{limiter.limits.usable_tpm} for the response -- too little to answer. "
                "Reduce the evidence budget."
            )
        if allowance < max_completion_tokens:
            log.info(
                "Reduced output allowance from %d to %d to fit the minute budget",
                max_completion_tokens, allowance,
            )
        max_completion_tokens = allowance

        estimate = prompt_tokens + max_completion_tokens
        waited = limiter.acquire(estimate)

        started = time.perf_counter()
        response = self._create(
            model=chosen,
            system=system,
            user=user,
            response_format=response_format,
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            limiter=limiter,
        )
        elapsed_ms = int((time.perf_counter() - started) * 1000)

        choice = response.choices[0]
        content = choice.message.content or ""
        finish_reason = choice.finish_reason

        if finish_reason == "length":
            # Constrained decoding guarantees schema conformance, not completeness. A
            # truncated answer would silently lose claims, so surface it rather than
            # parse a fragment.
            raise LLMError(
                f"{chosen} hit max_completion_tokens ({max_completion_tokens}); "
                "the structured answer was truncated"
            )

        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            # Should be unreachable under strict mode; treat as a hard failure if it happens.
            raise LLMError(f"{chosen} returned unparseable JSON under strict mode: {exc}") from exc

        usage = Usage(calls=1, latency_ms=elapsed_ms, waited_ms=int(waited * 1000))
        if response.usage is not None:
            usage.prompt_tokens = response.usage.prompt_tokens or 0
            usage.completion_tokens = response.usage.completion_tokens or 0

        # Correct the optimistic reservation with what the call actually cost.
        limiter.record(estimate, usage.total_tokens)

        log.debug(
            "%s: %d prompt + %d completion tokens in %d ms (waited %.1fs)",
            chosen,
            usage.prompt_tokens,
            usage.completion_tokens,
            elapsed_ms,
            waited,
        )
        return Completion(
            data=data, model=chosen, usage=usage, finish_reason=finish_reason
        )

    @retry(
        # SchemaValidationFailed is deliberately excluded: it is deterministic, so
        # retrying the identical request just fails identically.
        retry=retry_if_exception_type((RateLimitError, APIStatusError))
        & retry_if_not_exception_type(SchemaValidationFailed),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        reraise=True,
    )
    def _create(
        self,
        *,
        model: str,
        system: str,
        user: str,
        response_format: dict[str, Any],
        max_completion_tokens: int,
        temperature: float,
        reasoning_effort: str | None,
        limiter=None,
    ):
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": response_format,
            "max_completion_tokens": max_completion_tokens,
            "temperature": temperature,
        }
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort

        try:
            return self._client.chat.completions.create(**kwargs)
        except BadRequestError as exc:
            if "json_validate_failed" in str(exc):
                raise SchemaValidationFailed(
                    f"{model} produced output that did not satisfy the schema "
                    "(usually a truncated object). Consider raising max_completion_tokens "
                    "or asking for fewer, shorter claims."
                ) from exc
            raise
        except RateLimitError as exc:
            # Our estimate was optimistic, or another client shares the quota. Honour the
            # server's own backoff rather than tenacity's guess.
            if limiter is not None:
                limiter.note_retry_after(_retry_after_seconds(exc))
            raise


def _retry_after_seconds(exc: RateLimitError, default: float = 20.0) -> float:
    """Seconds Groq asked us to wait, from the 429 response headers."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or {}
    for header in ("retry-after", "x-ratelimit-reset-tokens", "x-ratelimit-reset-requests"):
        raw = headers.get(header)
        if not raw:
            continue
        try:
            return max(1.0, float(str(raw).rstrip("s")))
        except ValueError:
            continue
    return default
