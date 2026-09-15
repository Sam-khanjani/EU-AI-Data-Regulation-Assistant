"""State passed between LangGraph nodes."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, NamedTuple

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


@dataclass(slots=True)
class QueryState:
    """Everything one question accumulates on its way through the graph."""

    question: str
    """The question the pipeline answers: a follow-up is rewritten to stand alone first."""
    asked: str = ""
    """What the user actually typed, when it differs from ``question``."""

    # analyse
    intent: str = "lookup"
    search_queries: list[str] = field(default_factory=list)
    referenced_articles: list[str] = field(default_factory=list)
    referenced_annexes: list[str] = field(default_factory=list)

    # retrieve / rerank
    candidates: list[Candidate] = field(default_factory=list)
    """Chunk-level hits, before reranking and expansion."""

    retrieved: list[RetrievedUnit] = field(default_factory=list)
    """Survivors, expanded to whole articles."""
    evidence: list[RetrievedUnit] = field(default_factory=list)
    """The survivors that fit the answer prompt, labelled E1..En."""
    best_rerank_score: float = 0.0

    # generate / verify
    raw_answer: dict[str, Any] | None = None
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
    usage: Usage = field(default_factory=Usage)
    progress: list[Progress] = field(default_factory=list)
    latency_ms: int = 0
    query_log_id: int | None = None
    on_progress: Callable[[Progress], None] | None = None
    """Called with each step as it happens, so a chat interface can show the work live."""

    def note(self, step: str, detail: str = "") -> None:
        progress = Progress(step=step, detail=detail)
        self.progress.append(progress)
        if self.on_progress is not None:
            self.on_progress(progress)

    @property
    def document_version_ids(self) -> list[int]:
        """Versions that contributed evidence -- the provenance record for this answer."""
        seen: list[int] = []
        for unit in self.evidence:
            if unit.document_version_id not in seen:
                seen.append(unit.document_version_id)
        return seen

    @property
    def retrieved_chunk_ids(self) -> list[int]:
        ids: list[int] = []
        for unit in self.evidence:
            ids.extend(unit.matched_chunk_ids)
        return ids
