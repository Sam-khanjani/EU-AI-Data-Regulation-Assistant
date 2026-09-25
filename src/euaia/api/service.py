"""Application service: run a question through the pipeline and record what happened.

Every terminal path writes a ``query_log`` row -- abstentions included. That is what makes
the system auditable rather than merely careful: months later you can ask which document
version produced a given answer, which provisions it read, how much of it verified, and
which prompt version was in force.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from euaia import observability
from euaia.config import settings
from euaia.db.models import CheckRun, Chunk, DocumentVersion, QueryLog, Source
from euaia.graph.nodes import run_pipeline
from euaia.graph.state import Progress, QueryState, Turn
from euaia.ingest.embeddings import Embedder
from euaia.llm.groq_client import GroqClient
from euaia.verify.citations import VerifiedQuote

log = logging.getLogger(__name__)


@dataclass(slots=True)
class Citation:
    """A verified quote as the interface renders it."""

    citation_label: str
    quote: str
    deeplink: str | None
    version_label: str
    method: str


@dataclass(slots=True)
class AnswerClaim:
    text: str
    citations: list[Citation] = field(default_factory=list)
    basis: str = "law"
    """Which kind of document this claim's verified quotes came from -- ``law``, ``guidance``
    or ``code``. Computed from those quotes, never stated by the model."""


@dataclass(slots=True)
class AnswerView:
    """Everything the template needs. No database objects leak into the view."""

    question: str
    verdict: str
    asked: str = ""
    """What the user typed, when a follow-up was rewritten into ``question``."""
    summary: str = ""
    claims: list[AnswerClaim] = field(default_factory=list)
    criteria: list[dict[str, Any]] = field(default_factory=list)
    abstain_reason: str | None = None
    unanswered_aspects: list[str] = field(default_factory=list)
    follow_up_questions: list[str] = field(default_factory=list)
    sources: list[dict[str, str]] = field(default_factory=list)
    intent: str = "lookup"

    coverage: float = 0.0
    quotes_total: int = 0
    quotes_dropped: int = 0
    claims_dropped: int = 0
    latency_ms: int = 0
    tokens: int = 0
    waited_ms: int = 0
    """Time parked by the free-tier rate limiter, distinct from real latency."""
    progress: list[dict[str, str]] = field(default_factory=list)
    query_log_id: int | None = None

    @property
    def answered(self) -> bool:
        return self.verdict in ("answered", "partial")


def ask(
    question: str,
    session: Session,
    client: GroqClient | None = None,
    embedder: Embedder | None = None,
    *,
    history: Sequence[Turn] = (),
    on_progress: Callable[[Progress], None] | None = None,
) -> AnswerView:
    """Answer a question, in the context of any earlier turns, and log the attempt."""
    client = client or GroqClient()
    embedder = embedder or Embedder()

    state = run_pipeline(question, session, client, embedder, history, on_progress)
    view = _to_view(state)
    run = run_record(state, view)
    view.query_log_id = _log(session, state, view, run)
    observability.score(state.trace_id, _scores(view, run))
    return view


def run_record(state: QueryState, view: AnswerView) -> dict[str, Any]:
    """How the answer was produced, stored on its audit row for the monitoring page."""
    nodes = [name for name, _ in state.timings]
    basis: dict[str, int] = {}
    for claim in view.claims:
        basis[claim.basis] = basis.get(claim.basis, 0) + 1
    return {
        "trace_id": state.trace_id,
        "steps": [{"node": name, "ms": ms} for name, ms in state.timings],
        "rounds": nodes.count("plan"),
        "searches": nodes.count("rerank_evidence"),
        "repaired": nodes.count("repair"),
        "shortened": nodes.count("shorten"),
        "tokens": {
            model: {"input": i, "output": o, "calls": calls}
            for model, (i, o, calls) in state.usage.by_model.items()
        },
        "waited_ms": state.usage.waited_ms,
        "evidence": len(state.evidence),
        "claims": len(view.claims),
        "criteria": len(view.criteria),
        "basis": basis,
    }


def _scores(view: AnswerView, run: dict[str, Any]) -> dict[str, float | str]:
    """The answer's quality, as Langfuse scores on its trace."""
    scores: dict[str, float | str] = {
        "verdict": view.verdict,
        "rounds": run["rounds"],
        "latency_s": round(view.latency_ms / 1000, 1),
        "rate_limit_wait_s": round(view.waited_ms / 1000, 1),
    }
    if view.quotes_total:
        scores["quote_accuracy"] = (view.quotes_total - view.quotes_dropped) / view.quotes_total
        scores["coverage"] = view.coverage
    return scores


def answer_payload(view: AnswerView) -> dict[str, Any]:
    """An answer as JSON: the ``/api/ask`` response.

    The evaluation harness scores this same function's output when it runs in-process, so
    its two run modes cannot drift apart.
    """
    return {
        "question": view.question,
        "verdict": view.verdict,
        "summary": view.summary,
        "claims": [
            {
                "text": claim.text,
                "basis": claim.basis,
                "citations": [
                    {
                        "citation": c.citation_label,
                        "quote": c.quote,
                        "url": c.deeplink,
                        "version": c.version_label,
                    }
                    for c in claim.citations
                ],
            }
            for claim in view.claims
        ],
        "criteria": [
            {
                "criterion": c["criterion"],
                "status": c["status"],
                "explanation": c["explanation"],
                "citations": [
                    {"citation": q.citation_label, "quote": q.quote, "url": q.deeplink}
                    for q in c["citations"]
                ],
            }
            for c in view.criteria
        ],
        "abstain_reason": view.abstain_reason,
        "unanswered_aspects": view.unanswered_aspects,
        "follow_up_questions": view.follow_up_questions,
        "coverage": view.coverage,
        "quotes_total": view.quotes_total,
        "quotes_dropped": view.quotes_dropped,
        "latency_ms": view.latency_ms,
        "tokens": view.tokens,
        "waited_ms": view.waited_ms,
        "sources": view.sources,
        "query_log_id": view.query_log_id,
    }


def _to_view(state: QueryState) -> AnswerView:
    report = state.report
    versions = {unit.document_version_id: unit.version_label for unit in state.evidence}

    view = AnswerView(
        question=state.question,
        verdict=state.verdict,
        asked=state.asked,
        summary=state.summary,
        criteria=_criteria_view(state),
        abstain_reason=state.abstain_reason,
        unanswered_aspects=state.unanswered_aspects,
        follow_up_questions=state.follow_up_questions,
        intent=state.intent,
        latency_ms=state.latency_ms,
        tokens=state.usage.total_tokens,
        waited_ms=state.usage.waited_ms,
        progress=[{"step": p.step, "detail": p.detail} for p in state.progress],
        sources=[
            {"version_label": label, "version_id": str(vid)} for vid, label in versions.items()
        ],
    )

    if report is not None:
        view.coverage = report.coverage
        view.quotes_total = report.quotes_total
        view.quotes_dropped = report.quotes_dropped
        view.claims_dropped = len(report.dropped_claims)
    view.claims = [
        AnswerClaim(
            text=claim.text,
            citations=[_citation(q, versions) for q in claim.quotes],
            basis=claim.basis,
        )
        for claim in state.claims
    ]
    return view


def _criteria_view(state: QueryState) -> list[dict[str, Any]]:
    versions = {unit.document_version_id: unit.version_label for unit in state.evidence}
    out = []
    for criterion in state.criteria:
        out.append(
            {
                "criterion": criterion["criterion"],
                "status": criterion["status"],
                "explanation": criterion["explanation"],
                "citations": [_citation(q, versions) for q in criterion["quotes"]],
            }
        )
    return out


def _citation(quote: VerifiedQuote, versions: dict[int, str]) -> Citation:
    return Citation(
        citation_label=quote.citation_label,
        quote=quote.quote,
        deeplink=quote.deeplink,
        version_label=versions.get(quote.document_version_id or -1, ""),
        method=quote.method,
    )


def _log(
    session: Session, state: QueryState, view: AnswerView, run: dict[str, Any] | None = None
) -> int | None:
    """Write the audit row. A logging failure must never sink a good answer."""
    try:
        row = QueryLog(
            question=state.question,
            intent=state.intent,
            retrieved_chunk_ids=state.retrieved_chunk_ids,
            document_version_ids=state.document_version_ids,
            answer=state.raw_answer,
            citations=[
                {
                    "citation_label": c.citation_label,
                    "quote": c.quote,
                    "deeplink": c.deeplink,
                    "method": c.method,
                }
                for c in [q for claim in view.claims for q in claim.citations]
                + [q for criterion in view.criteria for q in criterion["citations"]]
            ],
            verdict=state.verdict,
            abstain_reason=state.abstain_reason,
            citation_coverage=view.coverage,
            quotes_total=view.quotes_total,
            quotes_repaired=int(any(p.step == "repairing" for p in state.progress)),
            quotes_dropped=view.quotes_dropped,
            latency_ms=state.latency_ms,
            model=settings.groq_model,
            prompt_version=settings.prompt_version,
            run=run,
        )
        session.add(row)
        session.commit()
        return row.id
    except Exception:  # noqa: BLE001
        log.exception("Failed to write query_log row")
        session.rollback()
        return None


def corpus_status(session: Session) -> list[dict[str, Any]]:
    """Per-source corpus state, for the status page.

    Includes each source's most recent change-detection result, if one has ever run --
    ``check_outcome`` is ``None`` rather than a fabricated value until an administrator
    actually runs a check.
    """
    rows = session.execute(
        select(Source, DocumentVersion)
        .join(DocumentVersion, DocumentVersion.source_id == Source.id)
        .where(DocumentVersion.status == "active")
        .order_by(Source.key)
    ).all()

    out = []
    for source, version in rows:
        chunk_count = session.query(Chunk).filter(
            Chunk.document_version_id == version.id
        ).count()
        last_check = session.scalar(
            select(CheckRun)
            .where(CheckRun.source_id == source.id)
            .order_by(CheckRun.ran_at.desc())
        )
        out.append(
            {
                "key": source.key,
                "title": source.title,
                "publisher": source.publisher,
                "authority": source.authority,
                "landing_url": source.landing_url,
                "version_label": version.version_label,
                "celex": version.celex,
                "doc_date": version.doc_date.isoformat() if version.doc_date else None,
                "format": version.format,
                "ingested_at": version.ingested_at,
                "retrieved_at": version.retrieved_at,
                "chunks": chunk_count,
                "embedded": chunk_count > 0,
                "content_sha256": version.content_sha256,
                "check_outcome": last_check.outcome if last_check else None,
                "checked_at": last_check.ran_at if last_check else None,
            }
        )
    return out
