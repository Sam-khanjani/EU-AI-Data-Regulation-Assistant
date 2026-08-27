"""LLM reranking of retrieved chunks.

Retrieval optimises for recall -- three legs fused, deliberately over-fetching. Reranking
trades that back for precision, because everything surviving here is spent as context in the
answer call, and irrelevant provisions do measurable harm: they give the model plausible
text to quote in support of the wrong claim.

**Reranking scores chunks, not articles.** A paragraph chunk averages ~99 tokens; the
article it belongs to averages 530 and reaches 3,369. Scoring twenty expanded articles would
be a ~16,000-token request against a free-tier ceiling of 8,000 tokens per minute, so it
would simply fail. It is also the wrong unit to score: relevance is a property of the passage
that matched, not of everything else in the same article. Expansion happens afterwards, to
the survivors only.

Uses the smaller ``gpt-oss-20b`` under the same strict-schema discipline as everything else.
Scores are advisory -- the sufficiency gate downstream decides whether the best of them is
good enough to answer from at all.
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
        max_completion_tokens=1024,
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
