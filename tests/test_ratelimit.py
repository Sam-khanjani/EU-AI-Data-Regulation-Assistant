"""Rate limiter tests.

Groq's free tier allows 8,000 tokens per minute per model. A single answer call spends
around 7,000 of them, so pacing is not an optimisation here -- without it the second request
of any minute is rejected. These tests use a fake clock so they run instantly.
"""

from __future__ import annotations

import pytest

from euaia.llm.ratelimit import (
    DailyBudgetExceeded,
    Limits,
    RateLimiter,
    estimate_tokens,
    limiter_for,
)


@pytest.fixture
def clock(monkeypatch):
    """A controllable clock; sleeping advances it instead of waiting."""

    class Clock:
        def __init__(self):
            self.now = 1000.0
            self.slept = 0.0

        def monotonic(self):
            return self.now

        def sleep(self, seconds):
            self.slept += seconds
            self.now += seconds

    c = Clock()
    monkeypatch.setattr("euaia.llm.ratelimit.time.monotonic", c.monotonic)
    monkeypatch.setattr("euaia.llm.ratelimit.time.sleep", c.sleep)
    return c


def limiter(**kw) -> RateLimiter:
    defaults = {"requests_per_minute": 30, "tokens_per_minute": 8_000, "headroom": 1.0}
    defaults.update(kw)
    return RateLimiter(limits=Limits(**defaults), name="test")


class TestTokenPacing:
    def test_first_call_does_not_wait(self, clock):
        assert limiter().acquire(3_000) == 0.0
        assert clock.slept == 0.0

    def test_calls_within_budget_do_not_wait(self, clock):
        rl = limiter()
        rl.acquire(3_000)
        rl.record(3_000, 3_000)
        assert rl.acquire(4_000) == 0.0

    def test_exceeding_the_minute_budget_waits(self, clock):
        rl = limiter()
        rl.acquire(6_000)
        rl.record(6_000, 6_000)
        waited = rl.acquire(4_000)
        assert waited > 0, "a call that overflows the minute must be delayed"
        assert waited <= 60.0

    def test_waits_only_until_the_window_frees_up(self, clock):
        rl = limiter()
        rl.acquire(5_000)
        rl.record(5_000, 5_000)
        clock.now += 50.0  # 10s left on the first entry
        waited = rl.acquire(5_000)
        assert 9.0 <= waited <= 11.0

    def test_window_is_sliding_not_fixed(self, clock):
        rl = limiter()
        rl.acquire(8_000)
        rl.record(8_000, 8_000)
        clock.now += 61.0  # the whole window has rolled off
        assert rl.acquire(8_000) == 0.0

    def test_headroom_reduces_the_usable_budget(self, clock):
        rl = limiter(headroom=0.5)
        assert rl.limits.usable_tpm == 4_000
        rl.acquire(3_500)
        rl.record(3_500, 3_500)
        assert rl.acquire(1_000) > 0

    def test_a_single_oversized_request_is_rejected_not_queued(self, clock):
        # Waiting cannot help: the request exceeds an entirely empty window.
        with pytest.raises(ValueError, match="exceeds the"):
            limiter().acquire(12_000)


class TestRequestPacing:
    def test_request_count_is_limited_independently_of_tokens(self, clock):
        rl = limiter(requests_per_minute=3)
        for _ in range(3):
            rl.acquire(10)
            rl.record(10, 10)
        assert rl.acquire(10) > 0, "the fourth request in the window must wait"


class TestActualUsageCorrection:
    def test_recording_replaces_the_estimate(self, clock):
        rl = limiter()
        rl.acquire(7_000)
        rl.record(7_000, 1_000)  # the call was far cheaper than feared
        # With only 1,000 actually spent there is room for a large follow-up.
        assert rl.acquire(6_500) == 0.0

    def test_daily_total_accumulates_actuals(self, clock):
        rl = limiter()
        for _ in range(3):
            rl.acquire(1_000)
            rl.record(1_000, 900)
        assert rl.tokens_used_today == 2_700


class TestDailyBudget:
    def test_raises_once_the_day_is_spent(self, clock):
        rl = limiter(tokens_per_day=1_000)
        rl.acquire(500)
        rl.record(500, 1_000)
        with pytest.raises(DailyBudgetExceeded, match="daily limit"):
            rl.acquire(10)


class TestRegistry:
    def test_limiters_are_shared_per_model(self):
        # Groq counts quota per model, so two clients must share one limiter.
        assert limiter_for("openai/gpt-oss-120b") is limiter_for("openai/gpt-oss-120b")

    def test_different_models_are_independent(self):
        assert limiter_for("openai/gpt-oss-120b") is not limiter_for("openai/gpt-oss-20b")


class TestEstimate:
    def test_scales_with_length(self):
        assert estimate_tokens("x" * 3600) > estimate_tokens("x" * 360)

    def test_biased_high_rather_than_low(self):
        # ~3.6 chars/token vs the usual ~4: overestimating costs throughput, under-
        # estimating costs a 429.
        text = "word " * 1000  # 5000 chars
        assert estimate_tokens(text) > len(text) / 4

    def test_combines_multiple_strings(self):
        assert estimate_tokens("a" * 100, "b" * 100) > estimate_tokens("a" * 100)


class TestSnapshot:
    def test_reports_current_usage(self, clock):
        rl = limiter()
        rl.acquire(2_000)
        rl.record(2_000, 2_000)
        snap = rl.snapshot()
        assert snap["requests_in_window"] == 1
        assert snap["tokens_in_window"] == 2_000
        assert snap["tokens_used_today"] == 2_000
