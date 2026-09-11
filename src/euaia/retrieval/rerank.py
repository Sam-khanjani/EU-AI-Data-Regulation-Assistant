"""Local cross-encoder reranking.

Retrieval optimises for recall -- three legs fused, deliberately over-fetching. Reranking
trades that back for precision, because everything surviving here is spent as context in the
answer call, and irrelevant provisions do measurable harm: they give the model plausible
text to quote in support of the wrong claim.

**This used to be an LLM call and is now a local model.** An LLM reranker puts every
candidate into *one* prompt, so its cost is the sum of all candidates: 20 of our chunks is
~9,000 tokens against a free-tier ceiling of 8,000 tokens per minute. Scoring a full
candidate set was therefore impossible on that tier -- the prompt was capped at 3,600 tokens,
covering about 7 of 20 -- and the reservation consumed ~92% of the minute budget, forcing the
limiter to sleep ~58 seconds before each call.

A cross-encoder scores each ``(query, passage)`` pair as an independent forward pass. There
is no shared context, so there is no shared budget, no per-minute ceiling, and no cutoff.

**What that cutoff actually cost is small, and worth recording so it is not overstated.**
Restricting the cross-encoders to the same top-8-by-retrieval-rank subset the LLM saw leaves
evidence survival unchanged (v2-m3 90% either way; ms-marco-MiniLM 75% either way). RRF
already places the answer near the top, so the 13 unscored candidates were mostly candidates
that did not matter. Measured on equal pools the LLM in fact *ranked better* (P@1 0.80 against
v2-m3's 0.50) and pulled half as many recitals into evidence. The gains from moving local are
zero token cost and no limiter stall -- not better ranking.

**Reranking scores chunks, not articles.** Unchanged, and still for two reasons: a packed
chunk targets ~500 tokens while its article averages 530 and reaches 3,369, and relevance is
a property of the passage that matched rather than of everything else in the same article.
Expansion happens afterwards, to the survivors only.

Scores are advisory -- the sufficiency gate downstream decides whether the best of them is
good enough to answer from at all. See the warning on calibration below before trusting
:attr:`RerankResult.best_score` as that gate.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field, replace

from euaia.config import settings
from euaia.llm.groq_client import Usage
from euaia.retrieval.hybrid import Candidate, RetrievedUnit

log = logging.getLogger(__name__)


class RerankerUnavailable(RuntimeError):
    """The local reranker could not be loaded.

    Raised rather than silently falling back to retrieval order: a pipeline that quietly
    stops reranking looks healthy and answers from worse evidence.
    """


# The model is loaded once per process and reused. Loading is several seconds and the
# weights are gigabytes, so doing it per request would dominate latency and memory.
_model = None
_model_name: str | None = None
_load_lock = threading.Lock()


def load_model(name: str | None = None):
    """Load (or return) the cross-encoder. Safe to call concurrently.

    Call :func:`warm` at application startup rather than paying this on a user's first
    question.
    """
    global _model, _model_name
    wanted = name or settings.rerank_model
    if _model is not None and _model_name == wanted:
        return _model

    with _load_lock:
        if _model is not None and _model_name == wanted:
            return _model
        try:
            import torch
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise RerankerUnavailable(
                "sentence-transformers is not installed; run `uv sync`"
            ) from exc

        started = time.perf_counter()
        try:
            model = CrossEncoder(
                wanted,
                # Sigmoid maps the raw logit to (0, 1). The logit itself is unbounded and
                # its scale differs between models, so normalising here keeps
                # `rerank_min_score` meaningful across a model change.
                activation_fn=torch.nn.Sigmoid(),
                max_length=settings.rerank_max_length,
            )
        except Exception as exc:
            raise RerankerUnavailable(
                f"could not load reranker {wanted!r}: {exc}. "
                "The weights download from the Hugging Face Hub on first use; see the "
                "'Reranker model' section of README.md for offline and pre-fetch options."
            ) from exc

        log.info(
            "Loaded reranker %s on %s in %.1fs",
            wanted, model.model.device, time.perf_counter() - started,
        )
        _model, _model_name = model, wanted
        return _model


def warm() -> None:
    """Pre-load the model. Failures are logged, not raised -- startup should not die."""
    try:
        load_model()
    except RerankerUnavailable:
        log.warning("Reranker could not be pre-loaded; it will retry on first use")


@dataclass(slots=True)
class LabelledChunk:
    """A candidate chunk with the short label used to refer to it."""

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
    """Always zero tokens now. Kept so callers aggregating usage need no special case."""
    best_score: float = 0.0
    """Highest score, in (0, 1). See the calibration warning in :func:`rerank`."""
    scored: int = 0
    skipped: bool = False
    """True when the model was not called at all."""
    latency_ms: int = 0


def rerank(
    question: str,
    candidates: list[Candidate],
    keep: int | None = None,
    min_score: float | None = None,
) -> RerankResult:
    """Score every candidate against the question and keep the best.

    .. warning::

       ``best_score`` is **not usable as an abstention signal**, and the default
       ``rerank_min_score`` is deliberately permissive.

       Measured over all 29 cases in ``eval/questions.yaml``, 20 candidates each, with the
       relevant chunks present for answerable questions and absent for the nine
       ``should_abstain`` ones. Score ranges of the two classes overlap **completely**:

       =========================  ==============  =============  =============
       model                      worst-scoring   best-scoring   clean cutoff?
                                  answerable      abstain
       =========================  ==============  =============  =============
       ms-marco-MiniLM-L-6-v2     0.0002          0.9945         no
       bge-reranker-base          0.0131          0.9966         no
       bge-reranker-v2-m3         0.0016          0.6899         no
       =========================  ==============  =============  =============

       The means look separable (v2-m3: 0.836 answerable vs 0.136 abstain) and that is a
       trap -- the extremes decide a threshold, not the means. The bait case
       ``nonexistent-article`` scores 0.9945 under MiniLM, while the genuine
       ``chatbot-applicability`` question scores 0.0002. Any cutoff either refuses real
       questions or answers invented ones.

       Crucially, **rank survives where score does not**: those same low-scoring questions
       still have MRR 1.0, i.e. the right provision is ranked first with near-zero
       confidence attached. So use this model to order evidence and never to decide whether
       to speak. Correctness is protected downstream by citation verification and the
       coverage gate, which are deterministic and do not depend on this number.
    """
    keep = keep or settings.rerank_keep
    threshold = settings.rerank_min_score if min_score is None else min_score

    if not candidates:
        return RerankResult()

    # Nothing to choose between: skip the work and keep what retrieval found.
    if len(candidates) <= keep:
        log.debug("rerank skipped: %d candidates <= keep=%d", len(candidates), keep)
        return RerankResult(
            kept=candidates,
            best_score=threshold,
            scored=len(candidates),
            skipped=True,
        )

    model = load_model()
    labelled = [
        LabelledChunk(label=f"C{i}", candidate=c) for i, c in enumerate(candidates, start=1)
    ]

    started = time.perf_counter()
    raw = model.predict(
        [(question, lc.candidate.chunk_text) for lc in labelled],
        batch_size=settings.rerank_batch_size,
        show_progress_bar=False,
    )
    elapsed_ms = int((time.perf_counter() - started) * 1000)

    scored = [replace(lc, score=float(s)) for lc, s in zip(labelled, raw, strict=True)]
    scored.sort(key=lambda lc: (lc.score or 0.0), reverse=True)
    passing = [lc for lc in scored if (lc.score or 0.0) >= threshold]
    best = max((lc.score or 0.0) for lc in scored)

    log.debug(
        "rerank: %d scored in %d ms, %d passed threshold %.4f, best %.4f",
        len(scored), elapsed_ms, len(passing), threshold, best,
    )
    return RerankResult(
        kept=[lc.candidate for lc in passing[:keep]],
        best_score=best,
        scored=len(scored),
        latency_ms=elapsed_ms,
    )
