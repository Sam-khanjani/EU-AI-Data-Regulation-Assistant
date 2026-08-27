"""Content-addressed embedding reuse.

The Gemini free tier allows **1,000 embed requests per day, counted per item**. The corpus
is 779 chunks, so one full ingestion very nearly exhausts a day's allowance and a single
mistake costs 24 hours. That alone would justify caching, but the reuse is worth having on
any tier:

* **Amendments are small.** A re-consolidation of the AI Act changes a handful of articles
  and leaves the rest byte-identical. Keying on the hash of the embedded text means
  re-ingestion pays only for what actually changed -- turning "the law was amended, re-index
  everything" into "the law was amended, embed the twelve provisions that moved".
* **Ingestion becomes resumable.** Work completed before a failure is still in the cache on
  the next run, so hitting the daily quota half way through costs the remainder, not
  everything.

The cache is keyed by ``(sha256(text), model, dim)``. Changing the embedding model or width
therefore misses cleanly rather than silently returning vectors from a different space.

Vectors are stored as ``halfvec`` (fp16), matching ``chunk.embedding``, so a cached vector
differs from the provider's fp32 original in about the fourth decimal place. That is the
same value the chunk table would have stored anyway, and cosine similarity is unaffected at
this magnitude -- but it does mean a cached vector is not bit-identical to a fresh one.

Entries are committed **per batch**, in their own transaction, deliberately separate from
the ingestion transaction. If ingestion later rolls back, the embeddings stay -- they cost
quota to produce and are valid regardless of whether the surrounding ingest succeeded.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import select

from euaia.config import settings
from euaia.db.models import EmbeddingCache
from euaia.db.session import SessionLocal
from euaia.ingest.embedder import Embedder

log = logging.getLogger(__name__)

# A pause longer than this is a daily reset, not a short enforcement window; waiting
# it out inside a command would hang for hours.
MAX_QUOTA_WAIT_SECONDS = 180.0


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
        saved = f"{self.reused / self.total:.0%}"
        return (
            f"{self.computed} embedded, {self.reused} reused from cache ({saved} of the "
            f"daily quota saved)"
        )


class QuotaExhausted(RuntimeError):
    """The embedding provider's daily quota is spent.

    Raised in place of the provider's raw error so callers can report what was salvaged.
    Retrying before the quota resets will not help.
    """

    def __init__(self, message: str, stats: EmbedStats) -> None:
        super().__init__(message)
        self.stats = stats


def _is_quota_error(exc: BaseException) -> bool:
    blob = str(exc)
    return "RESOURCE_EXHAUSTED" in blob or "exceeded your current quota" in blob


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


def embed_documents(
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
            "Embedding %d new chunks (%d reused from cache)", len(todo), stats.reused
        )
    waits_used = 0
    start = 0
    while start < len(todo):
        batch = todo[start : start + batch_size]
        batch_hashes = todo_hashes[start : start + batch_size]
        try:
            vectors = embedder.embed_documents(batch, batch_size=batch_size)
        except Exception as exc:
            if not _is_quota_error(exc):
                raise

            delay = _retry_delay(exc)
            if (
                delay is not None
                and delay <= MAX_QUOTA_WAIT_SECONDS
                and waits_used < max_quota_waits
            ):
                waits_used += 1
                log.info(
                    "Embedding quota hit after %d vectors; waiting %.0fs as instructed "
                    "(pause %d of %d)",
                    stats.computed, delay, waits_used, max_quota_waits,
                )
                time.sleep(delay + 1)
                continue  # retry the same batch; nothing was consumed from `todo`

            # Either the wait is too long to be a short window, or we have paused enough
            # times. Everything embedded so far is committed, so the next run resumes.
            raise QuotaExhausted(
                f"embedding quota exhausted after {stats.computed} new vectors this run; "
                f"{len(todo) - stats.computed} still to do. Already-computed embeddings "
                f"are cached, so re-running continues from here.",
                stats,
            ) from exc

        fresh = dict(zip(batch_hashes, vectors, strict=True))
        store(fresh, model, dim)  # commit per batch: a later failure cannot lose this
        cached.update(fresh)
        stats.computed += len(fresh)
        start += batch_size

    return [cached[h] for h in hashes], stats
