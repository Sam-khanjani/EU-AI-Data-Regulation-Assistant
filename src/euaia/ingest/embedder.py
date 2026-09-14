"""Gemini embeddings.

Two details that are easy to get wrong and expensive to debug:

**Asymmetric task types.** Documents must be embedded with ``RETRIEVAL_DOCUMENT`` and
queries with ``RETRIEVAL_QUERY``. Using one task type for both is a silent quality
regression -- nothing errors, retrieval just gets worse. The two entry points here
(:func:`embed_documents`, :func:`embed_query`) exist so the distinction cannot be
forgotten at a call site.

**Free-tier quota is per item, not per request.** Batching improves latency but not quota:
the Gemini free tier allows 1,000 ``embed_content`` items per day, and the corpus is 779
chunks. Reuse is therefore not an optimisation but a requirement -- see
:mod:`euaia.ingest.embedding_cache`, which callers should prefer over this module directly.

**Full 3072 dimensions.** ``gemini-embedding-001`` supports Matryoshka truncation to
smaller widths, but unlike ``-002`` it does *not* re-normalise truncated vectors -- callers
must do it themselves, and forgetting silently corrupts cosine similarity. Staying at the
native 3072 avoids the trap entirely, and pgvector stores it as ``halfvec`` so the HNSW
index is still buildable (the ``vector`` type caps out at 2000 dims).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from google import genai
from google.genai import types
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from euaia.config import settings

log = logging.getLogger(__name__)

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
