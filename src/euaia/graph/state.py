"""State passed between LangGraph nodes.

Nodes never mutate it: each returns the fields it changed, and LangGraph merges them in.
Fields written by parallel searches, or added to by every step, accumulate instead of being
replaced -- ``found``, ``rounds``, ``progress`` and ``usage`` -- so they carry reducers.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field, replace
from typing import Annotated, Any, NamedTuple

from euaia.llm.groq_client import Usage
from euaia.retrieval.hybrid import Candidate, RetrievedUnit
from euaia.verify.citations import VerificationReport, VerifiedClaim


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
    total = replace(spent, by_model=dict(spent.by_model))
    total.add(more)
    return total


@dataclass(slots=True)
class Finding:
    """What one search found: its provisions, and how well the best passage scored."""

    round: int
    side: int
    units: list[RetrievedUnit]
    best_score: float


@dataclass(slots=True)
class Round:
    """One answered question: the user's, or a follow-up the review asked for."""

    question: str
    intent: str
    outcome: dict[str, Any]
    """What :func:`euaia.graph.nodes._outcome` decided: verdict, summary, criteria..."""
    report: VerificationReport | None
    evidence: list[RetrievedUnit]
    raw_answer: dict[str, Any] | None


@dataclass(slots=True)
class Research:
    """What a search hands back to the answer graph, merged in by the reducers."""

    found: Annotated[list[Finding], operator.add] = field(default_factory=list)
    usage: Annotated[Usage, _total] = field(default_factory=Usage)
    progress: Annotated[list[Progress], operator.add] = field(default_factory=list)
    timings: Annotated[list[tuple[str, int]], operator.add] = field(default_factory=list)


@dataclass(slots=True)
class SearchTask(Research):
    """One search, run by the research subgraph: several run in parallel for a comparison."""

    query: str = ""
    """What the passages are ranked against: the question, or one side of a comparison."""
    label: str = ""
    """Shown in the progress line when this is not the plain question."""
    search_queries: list[str] = field(default_factory=list)
    articles: list[str] = field(default_factory=list)
    annexes: list[str] = field(default_factory=list)
    round: int = 1
    side: int = 0
    named_only: bool = False
    """Search only the named provisions -- a legal test's -- ranked against the question."""

    query_embedding: list[float] = field(default_factory=list)
    candidates: list[Candidate] = field(default_factory=list)
    """Chunk-level hits, before reranking and expansion."""
    kept: list[Candidate] = field(default_factory=list)
    """The candidates the reranker kept."""
    best_score: float = 0.0


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
    legal_test: str = "none"
    """For applicability: which test of the Act decides it, e.g. "high_risk"."""
    sides: list[str] = field(default_factory=list)
    """For a comparison: the things compared, one search each."""

    # plan / research / collect -- once per round
    round: int = 0
    focus: str = ""
    """The question this round answers: the user's, then any follow-up the review asks."""
    focus_intent: str = ""
    tasks: list[SearchTask] = field(default_factory=list)
    found: Annotated[list[Finding], operator.add] = field(default_factory=list)
    evidence: list[RetrievedUnit] = field(default_factory=list)
    """This round's provisions that fit the answer prompt, E1..En; after finalise, all of
    the answer's."""
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

    # review
    rounds: Annotated[list[Round], operator.add] = field(default_factory=list)
    next_step: dict[str, Any] | None = None
    """A follow-up search the review asked for: kind, question and sides."""

    # outcome
    verdict: str = "abstained"
    abstain_reason: str | None = None
    summary: str = ""
    claims: list[VerifiedClaim] = field(default_factory=list)
    """Verified claims from every round, criteria excepted."""
    unanswered_aspects: list[str] = field(default_factory=list)
    follow_up_questions: list[str] = field(default_factory=list)
    criteria: list[dict[str, Any]] = field(default_factory=list)

    # bookkeeping
    usage: Annotated[Usage, _total] = field(default_factory=Usage)
    progress: Annotated[list[Progress], operator.add] = field(default_factory=list)
    timings: Annotated[list[tuple[str, int]], operator.add] = field(default_factory=list)
    """Every node run, in order, with its milliseconds: the path the question took."""
    latency_ms: int = 0
    trace_id: str | None = None
    """The question's Langfuse trace, when tracing is on."""

    @property
    def document_version_ids(self) -> list[int]:
        """Versions that contributed evidence -- the provenance record for this answer."""
        return list(dict.fromkeys(unit.document_version_id for unit in self.evidence))

    @property
    def retrieved_chunk_ids(self) -> list[int]:
        return [i for unit in self.evidence for i in unit.matched_chunk_ids]
