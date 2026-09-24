"""State passed between LangGraph nodes.

Nodes never mutate it: each returns the fields it changed, and LangGraph merges them in.
Two fields accumulate instead of being replaced -- ``progress`` (every node appends its step)
and ``usage`` (every model call adds its cost) -- so they carry reducers.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field, replace
from typing import Annotated, Any, NamedTuple

from euaia.llm.groq_client import Usage
from euaia.retrieval.hybrid import Candidate, RetrievedUnit
from euaia.verify.citations import VerificationReport


@dataclass(slots=True)
class Progress:
    """One step of the pipeline, streamed to the UI.

    Strict-mode responses cannot stream token by token, so the interface shows the pipeline
    working instead. Showing the verification step is the point rather than a consolation:
    it is the part that distinguishes this from a chatbot with footnotes.
    """

    step: str
    detail: str = ""


class Turn(NamedTuple):
    """One earlier exchange in a conversation, as the follow-up rewriter reads it."""

    question: str
    """The standalone question that was answered, not necessarily what was typed."""
    answer: str
    """A plain-text recap of the answer."""


def _total(spent: Usage, more: Usage) -> Usage:
    total = replace(spent)
    total.add(more)
    return total


@dataclass(slots=True)
class QueryState:
    """Everything one question accumulates on its way through the graph."""

    question: str
    """The question the pipeline answers: a follow-up is rewritten to stand alone first."""
    asked: str = ""
    """What the user actually typed, when it differs from ``question``."""
    history: list[Turn] = field(default_factory=list)
    """The conversation so far, oldest first; empty for a first question."""

    # analyse
    intent: str = "lookup"
    search_queries: list[str] = field(default_factory=list)
    referenced_articles: list[str] = field(default_factory=list)
    referenced_annexes: list[str] = field(default_factory=list)

    # embed / retrieve / rerank / expand
    query_embedding: list[float] = field(default_factory=list)
    candidates: list[Candidate] = field(default_factory=list)
    """Chunk-level hits, before reranking and expansion."""
    kept: list[Candidate] = field(default_factory=list)
    """The candidates the reranker kept."""
    evidence: list[RetrievedUnit] = field(default_factory=list)
    """The kept chunks expanded to whole provisions that fit the answer prompt, E1..En."""
    best_rerank_score: float = 0.0

    # generate / verify
    raw_answer: dict[str, Any] | None = None
    overran: bool = False
    """The last draft overran the response schema."""
    shortened: bool = False
    """The current draft is asked for fewer claims, after an overrun."""
    rejected: list[str] = field(default_factory=list)
    """Quotes that failed verification, named back to the model on the repair round."""
    report: VerificationReport | None = None
    repair_attempted: bool = False

    # outcome
    verdict: str = "abstained"
    abstain_reason: str | None = None
    summary: str = ""
    unanswered_aspects: list[str] = field(default_factory=list)
    follow_up_questions: list[str] = field(default_factory=list)
    criteria: list[dict[str, Any]] = field(default_factory=list)

    # bookkeeping
    usage: Annotated[Usage, _total] = field(default_factory=Usage)
    progress: Annotated[list[Progress], operator.add] = field(default_factory=list)
    latency_ms: int = 0

    @property
    def document_version_ids(self) -> list[int]:
        """Versions that contributed evidence -- the provenance record for this answer."""
        return list(dict.fromkeys(unit.document_version_id for unit in self.evidence))

    @property
    def retrieved_chunk_ids(self) -> list[int]:
        return [i for unit in self.evidence for i in unit.matched_chunk_ids]
