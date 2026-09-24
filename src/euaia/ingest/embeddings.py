"""Gemini embeddings, and the cache that keeps them inside the free tier.

:class:`Embedder` calls the API. :func:`embed_with_cache` sits in front of it for ingestion
and reuses every vector already computed for identical text; a question's one query
embedding goes to the embedder directly. The sections below follow that order.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass

from google import genai
from google.genai import types
from sqlalchemy import select
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from euaia.config import settings
from euaia.db.models import EmbeddingCache
from euaia.db.session import SessionLocal

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------- Gemini
# Three details that are easy to get wrong and expensive to debug:
#
# **Asymmetric task types.** Documents must be embedded with ``RETRIEVAL_DOCUMENT`` and
# queries with ``RETRIEVAL_QUERY``. Using one task type for both is a silent quality
# regression -- nothing errors, retrieval just gets worse. The two entry points here
# (``Embedder.embed_documents``, ``Embedder.embed_query``) exist so the distinction cannot be
# forgotten at a call site.
#
# **Free-tier quota is per item, not per request.** Batching improves latency but not quota:
# the Gemini free tier allows 1,000 ``embed_content`` items per day, and the corpus is 779
# chunks. Reuse is therefore not an optimisation but a requirement -- see
# ``embed_with_cache`` below, which callers should prefer over the embedder directly.
#
# **Full 3072 dimensions.** ``gemini-embedding-001`` supports Matryoshka truncation to
# smaller widths, but unlike ``-002`` it does *not* re-normalise truncated vectors -- callers
# must do it themselves, and forgetting silently corrupts cosine similarity. Staying at the
# native 3072 avoids the trap entirely, and pgvector stores it as ``halfvec`` so the HNSW
# index is still buildable (the ``vector`` type caps out at 2000 dims).

TASK_DOCUMENT = "RETRIEVAL_DOCUMENT"
TASK_QUERY = "RETRIEVAL_QUERY"

# Conservative: the API caps both batch size and total batch tokens, and an over-large
# batch fails the whole call rather than degrading.
DEFAULT_BATCH_SIZE = 16


class EmbeddingError(RuntimeError):
    pass


def is_quota_error(exc: BaseException) -> bool:
    """Whether the provider refused the call because the embedding quota is spent."""
    blob = str(exc)
    return "RESOURCE_EXHAUSTED" in blob or "exceeded your current quota" in blob


def _should_retry(exc: BaseException) -> bool:
    """Retry transient failures only. A spent quota and a malformed response are final."""
    return not is_quota_error(exc) and not isinstance(exc, EmbeddingError)


class Embedder:
    """Thin wrapper over the Gemini embeddings endpoint."""

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        key = api_key or settings.google_api_key
        if not key:
            raise EmbeddingError(
                "GOOGLE_API_KEY is not set. Copy .env.example to .env and fill it in."
            )
        self._client = genai.Client(api_key=key)
        self.model = model or settings.embed_model

    # ---------------------------------------------------------------- public

    def embed_documents(
        self, texts: Sequence[str], batch_size: int = DEFAULT_BATCH_SIZE
    ) -> list[list[float]]:
        """Embed corpus text for indexing."""
        return self._embed_many(texts, TASK_DOCUMENT, batch_size)

    def embed_query(self, text: str) -> list[float]:
        """Embed a user question for retrieval."""
        return self._embed_many([text], TASK_QUERY, batch_size=1)[0]

    # --------------------------------------------------------------- internal

    def _embed_many(
        self, texts: Sequence[str], task_type: str, batch_size: int
    ) -> list[list[float]]:
        if not texts:
            return []

        out: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            try:
                out.extend(self._embed_batch(batch, task_type))
            except Exception as exc:  # noqa: BLE001
                if len(batch) == 1 or is_quota_error(exc):
                    # Never split a batch after a quota rejection. The free tier counts
                    # requests per item, so retrying sixteen items individually spends
                    # sixteen more of an allowance we have just been told is exhausted --
                    # turning a recoverable stop into a deeper hole.
                    raise
                # Other failures (one over-length member, a transient malformed response)
                # genuinely are per-item, so one bad chunk should not cost the other fifteen.
                log.warning(
                    "Batch of %d failed (%s); retrying individually", len(batch), exc
                )
                for text in batch:
                    out.extend(self._embed_batch([text], task_type))
            log.debug("Embedded %d/%d", min(start + batch_size, len(texts)), len(texts))
        return out

    @retry(
        # Quota rejections must NOT be retried: the provider raises a plain ClientError
        # for them, so an exception-type filter would miss it and burn three more of an
        # allowance we have just been told is exhausted. Transient errors are worth a
        # few attempts; a spent daily quota never is.
        retry=retry_if_exception(_should_retry),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=20),
        reraise=True,
    )
    def _embed_batch(self, batch: list[str], task_type: str) -> list[list[float]]:
        response = self._client.models.embed_content(
            model=self.model,
            contents=batch,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=settings.embed_dim,
            ),
        )
        embeddings = response.embeddings or []
        if len(embeddings) != len(batch):
            raise EmbeddingError(
                f"expected {len(batch)} embeddings, got {len(embeddings)}"
            )

        vectors: list[list[float]] = []
        for emb in embeddings:
            values = emb.values
            if not values:
                raise EmbeddingError("embedding response contained an empty vector")
            if len(values) != settings.embed_dim:
                raise EmbeddingError(
                    f"expected {settings.embed_dim} dimensions, got {len(values)}"
                )
            vectors.append(list(values))
        return vectors


# ------------------------------------------------------------------------ cache
# Content-addressed embedding reuse.
#
# The Gemini free tier allows **1,000 embed requests per day, counted per item**. The corpus
# is 779 chunks, so one full ingestion very nearly exhausts a day's allowance and a single
# mistake costs 24 hours. That alone would justify caching, but the reuse is worth having on
# any tier:
#
# * **Amendments are small.** A re-consolidation of the AI Act changes a handful of articles
#   and leaves the rest byte-identical. Keying on the hash of the embedded text means
#   re-ingestion pays only for what actually changed -- turning "the law was amended, re-index
#   everything" into "the law was amended, embed the twelve provisions that moved".
# * **Ingestion becomes resumable.** Work completed before a failure is still in the cache on
#   the next run, so hitting the daily quota half way through costs the remainder, not
#   everything.
#
# The cache is keyed by ``(sha256(text), model, dim)``. Changing the embedding model or width
# therefore misses cleanly rather than silently returning vectors from a different space.
#
# Vectors are stored as ``halfvec`` (fp16), matching ``chunk.embedding``, so a cached vector
# differs from the provider's fp32 original in about the fourth decimal place. That is the
# same value the chunk table would have stored anyway, and cosine similarity is unaffected at
# this magnitude -- but it does mean a cached vector is not bit-identical to a fresh one.
#
# Entries are committed **per batch**, in their own transaction, deliberately separate from
# the ingestion transaction. If ingestion later rolls back, the embeddings stay -- they cost
# quota to produce and are valid regardless of whether the surrounding ingest succeeded.

# A pause longer than this is a daily reset, not a short enforcement window; waiting
# it out inside a command would hang for hours.
MAX_QUOTA_WAIT_SECONDS = 180.0

# A rejection with no retry hint is still usually the per-minute window, not the daily one:
# measured on 2026-09-24, a run gave up on eight sources in a second each while the key still
# had most of the day's allowance. So wait one window before believing the quota is spent.
UNHINTED_QUOTA_WAIT_SECONDS = 60.0

# Consecutive pauses that produce no progress before the quota is taken to be the daily one.
# Any successful batch resets the count, so a long run can wait out many short windows.
MAX_STALLED_WAITS = 2


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class EmbedPlan:
    """What an embedding run would cost, computed without calling the API."""

    total: int
    cached: int
    to_embed: int
    model: str

    def __str__(self) -> str:
        return (
            f"{self.total} chunks: {self.cached} already cached, "
            f"{self.to_embed} to embed with {self.model}"
        )


@dataclass(slots=True)
class EmbedStats:
    reused: int = 0
    computed: int = 0

    @property
    def total(self) -> int:
        return self.reused + self.computed

    def __str__(self) -> str:
        if not self.total:
            return "nothing to embed"
        return (
            f"{self.computed} newly embedded (uses Gemini quota), "
            f"{self.reused} reused from cache (free)"
        )


class QuotaExhausted(RuntimeError):
    """The embedding provider's daily quota is spent.

    Raised in place of the provider's raw error so callers can report what was salvaged.
    Retrying before the quota resets will not help.
    """

    def __init__(self, message: str, stats: EmbedStats) -> None:
        super().__init__(message)
        self.stats = stats


def _retry_delay(exc: BaseException) -> float | None:
    """Seconds Google asked us to wait, from the quota error's RetryInfo.

    Worth honouring. The error names a *daily* quota id but often carries a delay of under
    a minute, which suggests a shorter enforcement window sits in front of it. Waiting the
    stated time and continuing completes an ingestion that would otherwise be spread over
    days; a genuinely daily exhaustion reports a delay far too long to wait out, which is
    how the two cases are told apart.
    """
    match = re.search(r"['\"]retryDelay['\"]:\s*['\"](\d+(?:\.\d+)?)s", str(exc))
    if match:
        return float(match.group(1))
    match = re.search(r"retry in (\d+(?:\.\d+)?)s", str(exc))
    return float(match.group(1)) if match else None


def lookup(hashes: Sequence[str], model: str, dim: int) -> dict[str, list[float]]:
    """Fetch every cached vector for these hashes in one query."""
    if not hashes:
        return {}
    with SessionLocal() as session:
        rows = session.scalars(
            select(EmbeddingCache).where(
                EmbeddingCache.text_sha256.in_(list(set(hashes))),
                EmbeddingCache.model == model,
                EmbeddingCache.dim == dim,
            )
        ).all()
        return {row.text_sha256: list(row.embedding) for row in rows}


def store(entries: dict[str, list[float]], model: str, dim: int) -> None:
    """Persist newly computed vectors, ignoring ones another run already wrote."""
    if not entries:
        return
    with SessionLocal() as session:
        existing = set(
            session.scalars(
                select(EmbeddingCache.text_sha256).where(
                    EmbeddingCache.text_sha256.in_(list(entries)),
                    EmbeddingCache.model == model,
                    EmbeddingCache.dim == dim,
                )
            ).all()
        )
        for digest, vector in entries.items():
            if digest in existing:
                continue
            session.add(
                EmbeddingCache(
                    text_sha256=digest, model=model, dim=dim, embedding=vector
                )
            )
        session.commit()


def plan(texts: Sequence[str], embedder: Embedder | None = None) -> EmbedPlan:
    """What would embedding these cost, without spending anything?

    Worth knowing before you start: the daily allowance is 1,000 items and the corpus is
    779, so "how many will this actually call?" is the difference between a run that
    finishes and one that dies two thirds of the way through.
    """
    embedder = embedder or Embedder()
    hashes = [text_hash(t) for t in texts]
    cached = lookup(hashes, embedder.model, settings.embed_dim)
    unique_new = {h for h in hashes if h not in cached}
    return EmbedPlan(
        total=len(texts),
        cached=sum(1 for h in hashes if h in cached),
        to_embed=len(unique_new),
        model=embedder.model,
    )


def embed_with_cache(
    texts: Sequence[str],
    embedder: Embedder | None = None,
    batch_size: int = 16,
    max_new: int | None = None,
    max_quota_waits: int = 12,
) -> tuple[list[list[float]], EmbedStats]:
    """Embed ``texts``, reusing anything already computed for identical text.

    ``max_new`` caps how many fresh embeddings this call may compute, so a run can be told
    to spend only part of the daily allowance. Exceeding it stops the same way a quota
    rejection does -- cleanly, with everything computed so far cached.

    Returns vectors in the same order as the input, plus what it cost.
    """
    embedder = embedder or Embedder()
    model, dim = embedder.model, settings.embed_dim

    hashes = [text_hash(t) for t in texts]
    cached = lookup(hashes, model, dim)
    stats = EmbedStats(reused=sum(1 for h in hashes if h in cached))

    # Deduplicate within this run too: repeated text costs one request, not several.
    todo: list[str] = []
    todo_hashes: list[str] = []
    seen: set[str] = set()
    for text, digest in zip(texts, hashes, strict=True):
        if digest in cached or digest in seen:
            continue
        seen.add(digest)
        todo.append(text)
        todo_hashes.append(digest)

    if max_new is not None and len(todo) > max_new:
        raise QuotaExhausted(
            f"{len(todo)} new embeddings needed but --max-embeddings is {max_new}. "
            f"Nothing was spent. Raise the cap or run again after the daily quota resets.",
            stats,
        )

    if todo:
        log.info(
            "Embedding %d new chunks (%d reused from cache)...", len(todo), stats.reused
        )
    waits_used = 0
    stalled = 0
    start = 0
    while start < len(todo):
        batch = todo[start : start + batch_size]
        batch_hashes = todo_hashes[start : start + batch_size]
        try:
            vectors = embedder.embed_documents(batch, batch_size=batch_size)
        except Exception as exc:
            if not is_quota_error(exc):
                raise

            hinted = _retry_delay(exc)
            delay = UNHINTED_QUOTA_WAIT_SECONDS if hinted is None else hinted
            if (
                delay <= MAX_QUOTA_WAIT_SECONDS
                and waits_used < max_quota_waits
                and stalled < MAX_STALLED_WAITS
            ):
                waits_used += 1
                stalled += 1
                log.info(
                    "Embedding rate limit hit after %d vectors; waiting %.0fs %s "
                    "(pause %d of %d)",
                    stats.computed, delay,
                    "as instructed" if hinted is not None else "for the per-minute window",
                    waits_used, max_quota_waits,
                )
                time.sleep(delay + 1)
                continue  # retry the same batch; nothing was consumed from `todo`

            # The wait is too long to be a short window, waiting has stopped producing
            # progress, or we have paused enough times: treat it as the daily quota.
            # Everything embedded so far is committed, so the next run resumes.
            raise QuotaExhausted(
                f"Gemini's daily embedding quota is used up ({stats.computed} embedded this "
                f"run, {len(todo) - stats.computed} still to do). Work done so far is saved.",
                stats,
            ) from exc

        fresh = dict(zip(batch_hashes, vectors, strict=True))
        store(fresh, model, dim)  # commit per batch: a later failure cannot lose this
        cached.update(fresh)
        stats.computed += len(fresh)
        stalled = 0
        start += batch_size

    return [cached[h] for h in hashes], stats
