"""The configured token budgets must fit inside the provider's rate limit.

This is arithmetic, not behaviour, but getting it wrong makes the answer call *impossible*
rather than merely slow: the rate limiter refuses a request that could never fit in an empty
minute, so every question fails. That happened once already, which is why it is a test.
"""

from __future__ import annotations

from euaia.config import settings

# euaia.retrieval.rerank and euaia.graph.state import from each other. Every real entry
# point into the app reaches euaia.graph first, which resolves that cleanly; importing
# euaia.retrieval.rerank first -- which a bare `from euaia.retrieval.rerank import ...`
# below would do, since this is the first thing in the whole suite to touch either module
# -- resolves it the other way round and fails. This import exists only to establish that
# order; nothing below uses it directly.
from euaia.graph import nodes as _  # noqa: F401
from euaia.llm.ratelimit import Limits
from euaia.retrieval.rerank import RERANK_MAX_COMPLETION_TOKENS


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
        worst_case = (
            settings.rerank_token_budget
            + settings.prompt_overhead_tokens
            + RERANK_MAX_COMPLETION_TOKENS
        )
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


class TestRerankBudgetCoversRealCandidates:
    def test_the_budget_fits_most_retrieved_candidates_at_real_chunk_size(self):
        # rerank_token_budget must be re-tuned whenever chunk_target_tokens changes, or
        # candidates are silently dropped before the reranker ever scores them -- exactly
        # what happened when a chunk-packing rewrite quadrupled average chunk size and left
        # only ~6 of 20 fused candidates reaching the reranker (including, in one measured
        # case, the single best dense match, ranked 10th after RRF fusion).
        #
        # Covering all of retrieve_candidates is not reachable at all: doing so needs
        # rerank_token_budget >= 20 * 500 = 10,000, but
        # TestDailyBudget caps it at 3,640 for a day of full-pipeline questions to fit at
        # all. 0.3 is a floor pinned to the current fix (3,600 covers 7), not a target --
        # raising it further means trading away daily question capacity, which is a product
        # decision, not a regression to catch here.
        fits = settings.rerank_token_budget // settings.chunk_target_tokens
        assert fits >= 0.3 * settings.retrieve_candidates, (
            f"at chunk_target_tokens={settings.chunk_target_tokens}, rerank_token_budget="
            f"{settings.rerank_token_budget} only covers {fits} of "
            f"{settings.retrieve_candidates} retrieved candidates"
        )


class TestEmbeddingBudget:
    def test_chunks_stay_under_the_embedding_input_limit(self):
        assert settings.chunk_max_tokens < settings.embed_input_token_limit

    def test_full_dimension_avoids_the_normalisation_trap(self):
        # gemini-embedding-001 does not re-normalise truncated vectors; staying at 3072
        # sidesteps it. Also the halfvec ceiling for an HNSW index is 4000.
        assert settings.embed_dim == 3072
        assert settings.embed_dim <= 4000
