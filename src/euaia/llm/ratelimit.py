"""Client-side rate limiting for Groq's free tier.

The free tier allows 30 requests and **8,000 tokens per minute** per model, and 200,000
tokens per day. Tokens-per-minute is the binding constraint by a wide margin: a single
answer call spends several thousand, so without pacing we would spend the minute's budget in
two requests and then collect 429s.

Waiting locally is better than being rejected. A 429 costs the request *and* the tokens it
consumed, and retry storms make it worse; sleeping until the window has room costs only
time. The limiter is a sliding window rather than a fixed one, because Groq's counters are
sliding too -- a fixed window would let us spend the whole budget at 59s and again at 61s.

Estimates are approximate (Groq's tokenizer is not ours), so a headroom factor keeps us
under the real ceiling. Actual usage is recorded after each call, which corrects the drift.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

WINDOW_SECONDS = 60.0


@dataclass(frozen=True, slots=True)
class Limits:
    """Per-model rate limits. Defaults are Groq's free tier."""

    requests_per_minute: int = 30
    tokens_per_minute: int = 8_000
    tokens_per_day: int = 200_000
    # Our token estimate is approximate; leave room so we do not overshoot the real limit.
    headroom: float = 0.85

    @property
    def usable_tpm(self) -> int:
        return int(self.tokens_per_minute * self.headroom)


class DailyBudgetExceeded(RuntimeError):
    """The daily token budget is spent. Retrying will not help until it resets."""


@dataclass
class RateLimiter:
    """Sliding-window limiter for one model."""

    limits: Limits = field(default_factory=Limits)
    name: str = "groq"

    _events: deque[tuple[float, int]] = field(default_factory=deque, init=False)
    _day_tokens: int = field(default=0, init=False)
    _day_started: float = field(default_factory=time.monotonic, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def _prune(self, now: float) -> None:
        cutoff = now - WINDOW_SECONDS
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()

    def _wait_time(self, now: float, estimated_tokens: int) -> float:
        """Seconds to wait before a call of this size fits inside both windows."""
        self._prune(now)
        waits = [0.0]

        if len(self._events) >= self.limits.requests_per_minute:
            waits.append(self._events[0][0] + WINDOW_SECONDS - now)

        used = sum(tokens for _, tokens in self._events)
        if used + estimated_tokens > self.limits.usable_tpm:
            # Drain events oldest-first until the request would fit.
            freed = 0
            for timestamp, tokens in self._events:
                freed += tokens
                if used - freed + estimated_tokens <= self.limits.usable_tpm:
                    waits.append(timestamp + WINDOW_SECONDS - now)
                    break
            else:
                # Even an empty window cannot hold it: the request itself is too large.
                raise ValueError(
                    f"a single request of ~{estimated_tokens} tokens exceeds the "
                    f"{self.limits.tokens_per_minute} tokens-per-minute limit for "
                    f"{self.name}; reduce the evidence budget"
                )

        return max(waits)

    def acquire(self, estimated_tokens: int) -> float:
        """Block until a call of this size may proceed. Returns seconds waited."""
        if self._day_tokens >= self.limits.tokens_per_day:
            raise DailyBudgetExceeded(
                f"{self.name}: {self._day_tokens:,} tokens used against a daily limit of "
                f"{self.limits.tokens_per_day:,}"
            )

        with self._lock:
            now = time.monotonic()
            wait = self._wait_time(now, estimated_tokens)
            if wait > 0:
                log.info(
                    "%s: pausing %.1fs to stay within %d tokens/min",
                    self.name, wait, self.limits.tokens_per_minute,
                )
            # Reserve optimistically; `record` corrects it with the real count.
            self._events.append((now + wait, estimated_tokens))

        if wait > 0:
            time.sleep(wait)
        return wait

    def record(self, estimated_tokens: int, actual_tokens: int) -> None:
        """Replace the estimate with what the call actually cost."""
        with self._lock:
            self._day_tokens += actual_tokens
            for i in range(len(self._events) - 1, -1, -1):
                timestamp, tokens = self._events[i]
                if tokens == estimated_tokens:
                    self._events[i] = (timestamp, actual_tokens)
                    return

    def note_retry_after(self, seconds: float) -> None:
        """Honour a server-side 429 by blocking the window for the stated period."""
        log.warning("%s: rate limited, sleeping %.1fs as instructed", self.name, seconds)
        with self._lock:
            self._events.append((time.monotonic() + seconds, 0))
        time.sleep(seconds)

    @property
    def tokens_used_today(self) -> int:
        return self._day_tokens

    def snapshot(self) -> dict[str, int | float]:
        now = time.monotonic()
        with self._lock:
            self._prune(now)
            return {
                "requests_in_window": len(self._events),
                "tokens_in_window": sum(t for _, t in self._events),
                "tokens_per_minute_limit": self.limits.tokens_per_minute,
                "tokens_used_today": self._day_tokens,
                "tokens_per_day_limit": self.limits.tokens_per_day,
                "day_elapsed_minutes": (now - self._day_started) / 60.0,
            }


_limiters: dict[str, RateLimiter] = {}
_registry_lock = threading.Lock()


def limiter_for(model: str, limits: Limits | None = None) -> RateLimiter:
    """Shared limiter per model -- Groq counts limits per model, not per key."""
    with _registry_lock:
        if model not in _limiters:
            _limiters[model] = RateLimiter(limits=limits or Limits(), name=model)
        return _limiters[model]


def estimate_tokens(*texts: str) -> int:
    """Rough token count for pacing.

    Deliberately crude and biased high: ~3.6 characters per token rather than the usual 4,
    plus a fixed allowance for message envelopes. Overestimating costs a little throughput;
    underestimating costs a 429.
    """
    characters = sum(len(t) for t in texts)
    return int(characters / 3.6) + 32
