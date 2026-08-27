"""The configured token budgets must fit inside the provider's rate limit.

This is arithmetic, not behaviour, but getting it wrong makes the answer call *impossible*
rather than merely slow: the rate limiter refuses a request that could never fit in an empty
minute, so every question fails. That happened once already, which is why it is a test.
"""

from __future__ import annotations

from euaia.config import settings
from euaia.llm.ratelimit import Limits


def _usable() -> int:
    return Limits(
        requests_per_minute=settings.groq_rpm,
        tokens_per_minute=settings.groq_tpm,
        tokens_per_day=settings.groq_tpd,
    ).usable_tpm


class TestAnswerCallFits:
    def test_worst_case_answer_call_fits_the_usable_minute_budget(self):
        worst_case = (
            settings.evidence_token_budget
            + settings.prompt_overhead_tokens
            + settings.answer_max_tokens
        )
        assert worst_case <= _usable(), (
            f"an answer call may reach ~{worst_case} tokens but only {_usable()} are usable "
            f"per minute ({settings.groq_tpm} x headroom); no request could ever be issued"
        )

    def test_rerank_call_fits(self):
        worst_case = settings.rerank_token_budget + settings.prompt_overhead_tokens + 1024
        assert worst_case <= _usable()

    def test_there_is_real_headroom_not_just_a_bare_fit(self):
        worst_case = (
            settings.evidence_token_budget
            + settings.prompt_overhead_tokens
            + settings.answer_max_tokens
        )
        assert worst_case <= _usable() * 0.95, "leave margin for estimate error"


class TestDailyBudget:
    def test_a_full_evaluation_run_fits_in_a_day(self):
        # ~20 cases run the full pipeline; the rest short-circuit at intent classification.
        per_question = (
            settings.rerank_token_budget
            + settings.evidence_token_budget
            + settings.prompt_overhead_tokens * 2
            + settings.answer_max_tokens
        )
        full_run = 20 * per_question + 9 * 800
        assert full_run <= settings.groq_tpd, (
            f"a full evaluation would cost ~{full_run:,} tokens against a daily limit of "
            f"{settings.groq_tpd:,}"
        )


class TestNoDuplicatedHeadroom:
    def test_prompt_fitting_uses_the_limiter_s_own_headroom(self):
        # Two separately-tuned headroom constants would silently diverge, and a prompt
        # sized against the wrong one is a request the limiter refuses outright.
        import inspect

        from euaia.graph import nodes

        source = inspect.getsource(nodes._fit_to_prompt_budget)
        assert "Limits(" in source
        assert "0.85" not in source, "headroom must come from Limits, not a literal"


class TestEmbeddingBudget:
    def test_chunks_stay_under_the_embedding_input_limit(self):
        assert settings.chunk_max_tokens < settings.embed_input_token_limit

    def test_full_dimension_avoids_the_normalisation_trap(self):
        # gemini-embedding-001 does not re-normalise truncated vectors; staying at 3072
        # sidesteps it. Also the halfvec ceiling for an HNSW index is 4000.
        assert settings.embed_dim == 3072
        assert settings.embed_dim <= 4000
