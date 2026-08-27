"""LLM reranking of retrieved chunks.

Retrieval optimises for recall -- three legs fused, deliberately over-fetching. Reranking
trades that back for precision, because everything surviving here is spent as context in the
answer call, and irrelevant provisions do measurable harm: they give the model plausible
text to quote in support of the wrong claim.

**Reranking scores chunks, not articles.** A packed chunk targets ~500 tokens; the article
it belongs to averages 530 and reaches 3,369. Scoring twenty expanded articles would be a
~16,000-token request against a free-tier ceiling of 8,000 tokens per minute, so it would
simply fail. It is also the wrong unit to score: relevance is a property of the passage that
matched, not of everything else in the same article. Expansion happens afterwards, to the
survivors only.

``settings.rerank_token_budget`` bounds how many of ``retrieve_candidates`` actually reach
this call -- ``_budgeted`` below walks the fused candidate list in rank order and stops once
the budget is spent, so anything past that cutoff is silently unscored. That budget has to be
kept in step with real chunk size by hand; it fell out of sync once already when a chunk-
packing rewrite roughly quadrupled average chunk size, and the reranker was left seeing only
the top ~6 of 20 candidates instead of most of them.

Uses the smaller ``gpt-oss-20b`` under the same strict-schema discipline as everything else.
Scores are advisory -- the sufficiency gate downstream decides whether the best of them is
good enough to answer from at all.

**Known limitation, still open.** For questions phrased around a real-world scenario rather
than legal terms -- "our customer support chatbot" rather than "AI systems intended to
interact with natural persons" (Article 50's actual wording) -- the model has been observed
scoring *every* candidate 0/10, including provisions plainly on topic. Raising
``reasoning_effort`` from "low" to "medium" was tried and measured against the live model: it
did not change the outcome, so it was reverted (see the call below) rather than paying more
tokens and latency for no benefit. The schema itself carries no bias toward 0 (checked). This
looks like a genuine judgement gap in the small model on this question shape, not a mechanical
bug -- fixing it likely means rewriting ``RERANK_SYSTEM`` to explicitly ask for scenario-to-
concept matching, or reranking with the larger model, both unverified as of this writing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace

from euaia.config import settings
from euaia.graph.prompts import RERANK_SYSTEM, rerank_user_prompt
from euaia.llm.groq_client import GroqClient, Usage
from euaia.llm.schemas import RERANK_SCHEMA
from euaia.retrieval.hybrid import Candidate, RetrievedUnit

log = logging.getLogger(__name__)

# "medium" reasoning effort (see the rerank() call below) spends real completion tokens on
# hidden deliberation before it emits the scored JSON -- 1024 was enough at "low" but left
# zero room for that at "medium" and the call failed with json_validate_failed on an empty
# generation. Referenced by tests/test_budgets.py::test_rerank_call_fits too, so the two
# cannot drift apart the way rerank_token_budget and real chunk size once did.
RERANK_MAX_COMPLETION_TOKENS = 2048


@dataclass(slots=True)
class LabelledChunk:
    """A candidate chunk with the short label the reranker uses to refer to it."""

    label: str
    candidate: Candidate
    score: float | None = None


@dataclass(slots=True)
class LabelledUnit:
    """An expanded unit with the label the *answer* model cites."""

    label: str
    unit: RetrievedUnit
    rerank_score: float | None = None

    # Passthroughs so this can be handed straight to prompt formatting.
    @property
    def citation_label(self) -> str:
        return self.unit.citation_label

    @property
    def heading(self) -> str | None:
        return self.unit.heading

    @property
    def text(self) -> str:
        return self.unit.text

    @property
    def version_label(self) -> str:
        return self.unit.version_label


def label_units(units: list[RetrievedUnit]) -> list[LabelledUnit]:
    """Assign E1, E2, ... in rank order.

    Labels are deliberately opaque handles rather than database ids: the model never sees an
    id it could invent a plausible-looking variant of, and a label we never issued is
    trivially detected during verification.
    """
    return [LabelledUnit(label=f"E{i}", unit=unit) for i, unit in enumerate(units, start=1)]


@dataclass(slots=True)
class RerankResult:
    kept: list[Candidate] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    best_score: float = 0.0
    scored: int = 0
    skipped: bool = False
    """True when the model was not called at all."""


def _budgeted(candidates: list[Candidate], budget: int) -> list[LabelledChunk]:
    """Label candidates in rank order until the rerank token budget is spent."""
    labelled: list[LabelledChunk] = []
    spent = 0
    for candidate in candidates:
        cost = candidate.token_count or max(1, len(candidate.chunk_text) // 4)
        if labelled and spent + cost > budget:
            break
        labelled.append(LabelledChunk(label=f"C{len(labelled) + 1}", candidate=candidate))
        spent += cost
    return labelled


def format_chunks(labelled: list[LabelledChunk]) -> str:
    """Render candidate chunks as labelled blocks for the reranker."""
    return "\n\n---\n\n".join(
        f"[{lc.label}] {lc.candidate.chunk_text}" for lc in labelled
    )


def rerank(
    client: GroqClient,
    question: str,
    candidates: list[Candidate],
    keep: int | None = None,
    min_score: float | None = None,
) -> RerankResult:
    """Score and trim candidates. Falls back to retrieval order if the model misbehaves."""
    keep = keep or settings.rerank_keep
    threshold = settings.rerank_min_score if min_score is None else min_score

    if not candidates:
        return RerankResult()

    # Nothing to choose between: skip the call and save the tokens for the answer.
    if len(candidates) <= keep:
        log.debug("rerank skipped: %d candidates <= keep=%d", len(candidates), keep)
        return RerankResult(
            kept=candidates,
            best_score=threshold,
            scored=len(candidates),
            skipped=True,
        )

    labelled = _budgeted(candidates, settings.rerank_token_budget)
    completion = client.structured(
        system=RERANK_SYSTEM,
        user=rerank_user_prompt(question, format_chunks(labelled)),
        response_format=RERANK_SCHEMA,
        model=settings.groq_small_model,
        max_completion_tokens=RERANK_MAX_COMPLETION_TOKENS,
        # Tried "medium" here on the theory that matching a real-world scenario to the
        # legal concept it describes (e.g. "customer support chatbot" to Article 50's "AI
        # systems intended to interact with natural persons") needs more deliberation than
        # "low" gives it. Measured against the live model: "medium" did not change the
        # outcome -- every candidate still scored 0/10, including ones plainly on topic --
        # so this stays "low" rather than paying more tokens and latency for no measured
        # benefit. The flat-zero behaviour on colloquially-phrased questions is real and
        # still open; see the module docstring above.
        reasoning_effort="low",
    )

    scores = {
        str(row.get("label", "")).strip(): float(row.get("score", 0))
        for row in completion.data.get("rankings", [])
    }
    missing = [lc.label for lc in labelled if lc.label not in scores]
    if missing:
        # Unscored candidates keep their retrieval position rather than being discarded:
        # a reranker omission should not silently lose evidence.
        log.warning("Reranker did not score %d candidates: %s", len(missing), missing)

    scored: list[LabelledChunk] = []
    for position, lc in enumerate(labelled):
        score = scores.get(lc.label)
        if score is None:
            # Neutral score, decayed by retrieval rank so ordering stays stable.
            score = threshold - 0.01 * position
        scored.append(replace(lc, score=score))

    scored.sort(key=lambda lc: (lc.score or 0.0), reverse=True)
    passing = [lc for lc in scored if (lc.score or 0.0) >= threshold]
    best = max((lc.score or 0.0) for lc in scored)

    log.debug(
        "rerank: %d scored, %d passed threshold %.1f, best %.1f",
        len(scored), len(passing), threshold, best,
    )
    return RerankResult(
        kept=[lc.candidate for lc in passing[:keep]],
        usage=completion.usage,
        best_score=best,
        scored=len(scored),
    )
