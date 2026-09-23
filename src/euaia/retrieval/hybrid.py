"""Hybrid retrieval: dense vectors, full-text search, and direct structural lookup.

Three legs, because each one fails at something the others cover:

**Dense** (pgvector cosine over ``halfvec``) handles paraphrase -- "how do I know if my
system is high-risk" finding Article 6 without sharing vocabulary with it.

**Full text** (Postgres ``tsvector``) handles the exact tokens that dense retrieval
reliably under-weights. Legal questions are full of them: "Annex III", "GPAI",
"Article 50", "conformity assessment". Embeddings smear these into their neighbourhoods;
lexical search does not. This leg is not optional.

**Structural** short-circuits the case where the user simply names a provision. If someone
asks "what does Article 50 say", the right answer is Article 50, not whatever is nearest in
embedding space. Ranking cannot be trusted to get this right, so it is not asked to.

The three are fused with Reciprocal Rank Fusion, which combines rankings without needing
their scores to be commensurable -- cosine similarity and ``ts_rank_cd`` are not.

Ordering matters here. Retrieval and reranking both work on **paragraph-level chunks**;
only the survivors are **expanded to their parent article** before the answer model sees
them. We embed paragraphs because articles exceed the embedding model's token limit, we
rank paragraphs because relevance belongs to the passage that matched, and we reason over
whole articles because a paragraph read alone is often meaningless.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace

from sqlalchemy import text
from sqlalchemy.orm import Session

from euaia.config import settings
from euaia.llm.ratelimit import estimate_tokens

log = logging.getLogger(__name__)

# Standard RRF damping. Large enough that the top few ranks do not dominate, small enough
# that deep results stay negligible.
RRF_K = 60

# Restricting a search to particular sources. Useful in its own right -- answering
# "what does the operative text say" without recitals, or scoping to one regulation
# once more are indexed -- and it is what makes retrieval tests deterministic against
# a database that also holds the real corpus.
_SOURCE_FILTER = """
          AND (CAST(:source_keys AS text[]) IS NULL OR s.key = ANY(:source_keys))"""


@dataclass(slots=True)
class Candidate:
    """A retrieved chunk with its provenance and fused score."""

    chunk_id: int
    unit_id: int
    unit_path: str
    unit_type: str
    unit_number: str | None
    heading: str | None
    document_version_id: int
    version_label: str
    source_key: str
    authority: str
    """``law`` | ``guidance`` | ``code`` -- see :data:`euaia.ingest.ec_documents.Authority`."""

    text: str
    """The structural unit's own text -- what quotes are verified against."""

    chunk_text: str = ""
    """The embedded chunk (breadcrumb + body) -- what reranking scores."""

    deeplink: str | None = None

    token_count: int = 0
    score: float = 0.0
    legs: dict[str, int] = field(default_factory=dict)
    """Which legs found it, and at what rank. Useful for debugging retrieval quality."""


@dataclass(slots=True)
class RetrievedUnit:
    """A citable unit, expanded to article granularity, ready to become Evidence."""

    unit_id: int
    unit_path: str
    unit_type: str
    unit_number: str | None
    heading: str | None
    citation_label: str
    text: str
    document_version_id: int
    version_label: str
    source_key: str
    authority: str
    deeplink: str | None
    score: float
    matched_chunk_ids: list[int] = field(default_factory=list)
    label: str = ""
    """The handle the answer model cites (E1, E2, ...), assigned by :func:`label_units`."""


# What every retrieval leg starts from: chunks of active document versions, with their unit,
# version and source. Each leg appends its own condition, the source filter and an ordering.
_ACTIVE_CHUNKS = """
    SELECT c.id            AS chunk_id,
           c.text          AS chunk_text,
           c.token_count   AS chunk_tokens,
           su.id           AS unit_id,
           su.unit_path    AS unit_path,
           su.unit_type    AS unit_type,
           su.unit_number  AS unit_number,
           su.heading      AS heading,
           su.text         AS unit_text,
           su.eurlex_deeplink AS deeplink,
           dv.id           AS document_version_id,
           dv.version_label AS version_label,
           s.key           AS source_key,
           s.authority     AS authority
    FROM chunk c
    JOIN structural_unit su ON su.id = c.structural_unit_id
    JOIN document_version dv ON dv.id = c.document_version_id
    JOIN source s ON s.id = dv.source_id
    WHERE dv.status = 'active'"""


def dense_search(
    session: Session,
    embedding: list[float],
    limit: int,
    source_keys: list[str] | None = None,
) -> list[Candidate]:
    """Nearest chunks by cosine distance over the HNSW index."""
    sql = text(
        _ACTIVE_CHUNKS
        + _SOURCE_FILTER
        + """
        ORDER BY c.embedding <=> CAST(:embedding AS halfvec)
        LIMIT :limit
        """
    )
    rows = session.execute(
        sql,
        {
            "embedding": _vector_literal(embedding),
            "limit": limit,
            "source_keys": source_keys,
        },
    ).mappings()
    return [_candidate(row) for row in rows]


def fulltext_search(
    session: Session, query: str, limit: int, source_keys: list[str] | None = None
) -> list[Candidate]:
    """Lexical search over the generated ``tsvector`` column."""
    sql = text(
        _ACTIVE_CHUNKS
        + """
          AND c.fts @@ websearch_to_tsquery('english', :query)"""
        + _SOURCE_FILTER
        + """
        ORDER BY ts_rank_cd(c.fts, websearch_to_tsquery('english', :query)) DESC
        LIMIT :limit
        """
    )
    rows = session.execute(
        sql, {"query": query, "limit": limit, "source_keys": source_keys}
    ).mappings()
    return [_candidate(row) for row in rows]


def structural_search(
    session: Session,
    articles: list[str],
    annexes: list[str],
    limit: int = 40,
    source_keys: list[str] | None = None,
) -> list[Candidate]:
    """Fetch provisions the user named outright, bypassing ranking entirely."""
    if not articles and not annexes:
        return []

    sql = text(
        _ACTIVE_CHUNKS
        + """
          AND (
                (su.unit_type = 'article' AND su.unit_number = ANY(:articles))
             OR (su.unit_type = 'annex'   AND su.unit_number = ANY(:annexes))
             OR (su.unit_type = 'paragraph' AND split_part(
                    split_part(su.unit_path, 'ART_', 2), '/', 1) = ANY(:articles))
          )"""
        + _SOURCE_FILTER
        + """
        ORDER BY su.ordinal
        LIMIT :limit
        """
    )
    rows = session.execute(
        sql,
        {
            "articles": [a.strip() for a in articles],
            "annexes": [a.strip().upper() for a in annexes],
            "limit": limit,
            "source_keys": source_keys,
        },
    ).mappings()
    return [_candidate(row) for row in rows]


def reciprocal_rank_fusion(
    legs: dict[str, list[Candidate]], weights: dict[str, float] | None = None
) -> list[Candidate]:
    """Fuse ranked lists by RRF, keeping which leg contributed each hit.

    RRF combines rankings rather than scores, so cosine similarity and ``ts_rank_cd`` never
    have to be made comparable.
    """
    weights = weights or {}
    fused: dict[int, Candidate] = {}

    for leg_name, candidates in legs.items():
        weight = weights.get(leg_name, 1.0)
        for rank, candidate in enumerate(candidates, start=1):
            existing = fused.get(candidate.chunk_id)
            if existing is None:
                existing = candidate
                existing.score = 0.0
                existing.legs = {}
                fused[candidate.chunk_id] = existing
            existing.score += weight / (RRF_K + rank)
            existing.legs[leg_name] = rank

    return sorted(fused.values(), key=lambda c: c.score, reverse=True)


def expand_to_units(
    session: Session, candidates: list[Candidate], max_unit_tokens: int | None = None
) -> list[RetrievedUnit]:
    """Collapse chunks onto the units to cite, promoting paragraphs to their article.

    Two paragraphs of Article 6 becoming two separate pieces of evidence would waste
    context and invite the model to cite the same provision twice. They become one
    Article 6.

    Promotion is skipped when the parent article is longer than ``max_unit_tokens``.
    Article 5 is roughly 1,900 tokens on its own -- promoting a single matched paragraph
    into the whole of it spends most of a minute's budget on text the question did not ask
    about, and crowds out other provisions. For those, the matched paragraph is the better
    unit of evidence; it is still precisely citable.
    """
    if not candidates:
        return []

    target_ids: dict[int, int] = {}  # candidate unit id -> unit id to cite
    for candidate in candidates:
        target_ids[candidate.unit_id] = candidate.unit_id

    ceiling = settings.max_expand_tokens if max_unit_tokens is None else max_unit_tokens
    paragraph_ids = [c.unit_id for c in candidates if c.unit_type == "paragraph"]
    if paragraph_ids:
        rows = session.execute(
            text(
                """
                SELECT child.id AS child_id, parent.id AS parent_id, parent.text AS parent_text
                FROM structural_unit child
                JOIN structural_unit parent ON parent.id = child.parent_id
                WHERE child.id = ANY(:ids) AND parent.unit_type = 'article'
                """
            ),
            {"ids": paragraph_ids},
        ).mappings()
        for row in rows:
            if estimate_tokens(row["parent_text"]) <= ceiling:
                target_ids[row["child_id"]] = row["parent_id"]

    best: dict[int, RetrievedUnit] = {}
    order: list[int] = []
    for candidate in candidates:
        unit_id = target_ids[candidate.unit_id]
        if unit_id in best:
            best[unit_id].score = max(best[unit_id].score, candidate.score)
            best[unit_id].matched_chunk_ids.append(candidate.chunk_id)
            continue
        best[unit_id] = RetrievedUnit(
            unit_id=unit_id,
            unit_path=candidate.unit_path,
            unit_type=candidate.unit_type,
            unit_number=candidate.unit_number,
            heading=candidate.heading,
            citation_label="",
            text=candidate.text,
            document_version_id=candidate.document_version_id,
            version_label=candidate.version_label,
            source_key=candidate.source_key,
            authority=candidate.authority,
            deeplink=candidate.deeplink,
            score=candidate.score,
            matched_chunk_ids=[candidate.chunk_id],
        )
        order.append(unit_id)

    _hydrate_units(session, [best[i] for i in order])
    return [best[i] for i in order]


def _hydrate_units(session: Session, units: list[RetrievedUnit]) -> None:
    """Replace each unit's text with the full text of the unit actually being cited."""
    if not units:
        return
    rows = session.execute(
        text(
            """
            SELECT id, unit_path, unit_type, unit_number, heading, text, eurlex_deeplink
            FROM structural_unit WHERE id = ANY(:ids)
            """
        ),
        {"ids": [u.unit_id for u in units]},
    ).mappings()
    by_id = {row["id"]: row for row in rows}

    for unit in units:
        row = by_id.get(unit.unit_id)
        if row is None:
            continue
        unit.unit_path = row["unit_path"]
        unit.unit_type = row["unit_type"]
        unit.unit_number = row["unit_number"]
        unit.heading = row["heading"]
        unit.text = row["text"]
        unit.deeplink = row["eurlex_deeplink"]
        # source_key comes from the unit, not the row: it is already known from the candidate,
        # and this query reads structural_unit alone.
        unit.citation_label = citation_label(
            row["unit_type"], row["unit_number"], row["unit_path"], unit.source_key
        )


def fit_token_budget(
    units: list[RetrievedUnit], budget: int, estimator=None
) -> list[RetrievedUnit]:
    """Take units in the order given until the token budget is spent.

    Callers pass them ordered by authority, so binding text has first claim on the budget and
    guidance fills whatever is left. Spending in pure relevance order instead would let the
    Act be reranked into the evidence and then dropped for want of room.

    Free-tier tokens-per-minute is the scarcest resource in the pipeline, and a single
    article can be 3,369 tokens. Without a budget, one long provision in the top results
    pushes the answer call over the minute limit and the request fails outright. Taking
    fewer, better provisions is the right trade: precision matters more than breadth once
    reranking has done its job.

    The first unit is kept even when it alone exceeds the budget -- abstaining for lack of
    context we actually retrieved would be worse, and the provisions that matter most are
    often the longest. Article 5 is 3,714 tokens against a 2,400 budget, and it is the whole
    answer to "which practices are prohibited?"; dropping it in favour of two short recitals
    that happen to fit would be a worse answer, not a cheaper one.

    That exemption is safe only because callers order by authority first, which is what puts
    a binding provision in the position that receives it. Ordered by relevance instead, it
    would be handed to whatever scored highest -- including a Commission Q&A page, which has
    no heading structure and is therefore a single ~5,700-token unit that would evict
    everything else in the evidence set.
    """
    estimate = estimator or estimate_tokens
    kept: list[RetrievedUnit] = []
    spent = 0

    for unit in units:
        cost = estimate(unit.text)
        if kept and spent + cost > budget:
            continue
        kept.append(unit)
        spent += cost

    log.debug("evidence budget: kept %d/%d units, ~%d tokens", len(kept), len(units), spent)
    return kept


AUTHORITY_ORDER: tuple[str, ...] = ("law", "guidance", "code")
"""Most authoritative first. See :data:`euaia.ingest.ec_documents.Authority`."""


def by_authority(units: list[RetrievedUnit]) -> list[RetrievedUnit]:
    """Re-order ranked units so binding text comes first, keeping rank within each tier.

    Retrieval and reranking answer "what is most relevant?", which is the right question for
    them and the wrong one to hand an answer model unmodified. Asked what the law requires,
    a voluntary code of practice can easily out-rank the provision it implements: it is
    longer, more concrete, and repeats the question's vocabulary. Whichever evidence block
    lands at E1 is what the answer is built around.

    So relevance decides *which* provisions survive, and this decides what order they are
    read in. Nothing is dropped -- the guidance and the codes still reach the model, after
    the law they are interpreting.
    """
    rank = {name: index for index, name in enumerate(AUTHORITY_ORDER)}
    return sorted(units, key=lambda unit: rank.get(unit.authority, len(rank)))


def label_units(units: list[RetrievedUnit]) -> list[RetrievedUnit]:
    """Assign E1, E2, ... in rank order.

    Labels are deliberately opaque handles rather than database ids: the model never sees an
    id it could invent a plausible-looking variant of, and a label we never issued is
    trivially detected during verification.
    """
    return [replace(unit, label=f"E{i}") for i, unit in enumerate(units, start=1)]


def citation_label(
    unit_type: str, unit_number: str | None, unit_path: str, source_key: str = ""
) -> str:
    """Human citation string, e.g. 'Article 6', 'Annex III', 'GPAI Guidelines 2.1'.

    A provision of the Act is cited by its number alone, because there is only one Act and
    "Article 6" is how it is referred to everywhere. Everything else must name its document:
    "Section 2.1" is meaningless across eleven Commission publications, and a reader has to be
    able to see at a glance that a citation points at a voluntary code rather than at the law.
    """
    from euaia.ingest.sources import BY_KEY

    spec = BY_KEY.get(source_key)
    if spec and spec.authority != "law":
        return f"{spec.doc_title} {unit_number}" if unit_number else spec.doc_title

    pretty = {
        "article": "Article",
        "annex": "Annex",
        "recital": "Recital",
        "chapter": "Chapter",
        "section": "Section",
        "paragraph": "Paragraph",
    }.get(unit_type)
    if pretty and unit_number:
        return f"{pretty} {unit_number}"
    return unit_path


def retrieve(
    session: Session,
    *,
    query: str,
    embedding: list[float],
    search_queries: list[str] | None = None,
    articles: list[str] | None = None,
    annexes: list[str] | None = None,
    candidates: int | None = None,
    source_keys: list[str] | None = None,
) -> list[Candidate]:
    """Run all three legs and fuse them, returning **chunk-level** candidates.

    Expansion to whole articles deliberately does *not* happen here. It runs after
    reranking (:func:`euaia.retrieval.hybrid.expand_to_units`), for two reasons:

    * **Correctness** -- relevance is a property of the passage that matched, not of the
      whole article it happens to sit in. Scoring a 3,000-token article because one of its
      paragraphs matched dilutes the signal.
    * **Cost** -- 20 expanded articles is roughly 16,000 tokens, twice Groq's free-tier
      minute budget. 20 paragraph chunks is under 2,000.
    """
    limit = candidates or settings.retrieve_candidates
    legs: dict[str, list[Candidate]] = {
        "dense": dense_search(session, embedding, limit, source_keys),
        "fulltext": fulltext_search(session, query, limit, source_keys),
    }

    # Extra lexical passes for the analyser's reformulations; they cost one index scan each.
    for i, extra in enumerate(search_queries or []):
        if extra and extra.strip() and extra.strip().lower() != query.strip().lower():
            legs[f"fulltext_{i}"] = fulltext_search(session, extra, limit, source_keys)

    structural = structural_search(
        session, articles or [], annexes or [], source_keys=source_keys
    )
    if structural:
        legs["structural"] = structural

    # A named provision outranks a merely similar one.
    fused = reciprocal_rank_fusion(legs, weights={"structural": 2.0})
    log.debug(
        "retrieval: %s -> %d fused candidates",
        {name: len(c) for name, c in legs.items()},
        len(fused),
    )
    return fused[:limit]


def _candidate(row) -> Candidate:
    return Candidate(
        chunk_id=row["chunk_id"],
        unit_id=row["unit_id"],
        unit_path=row["unit_path"],
        unit_type=row["unit_type"],
        unit_number=row["unit_number"],
        heading=row["heading"],
        document_version_id=row["document_version_id"],
        version_label=row["version_label"],
        source_key=row["source_key"],
        authority=row["authority"],
        text=row["unit_text"],
        chunk_text=row["chunk_text"],
        token_count=row["chunk_tokens"],
        deeplink=row["deeplink"],
    )


def _vector_literal(embedding: list[float]) -> str:
    """pgvector accepts a bracketed literal; this avoids a driver-level type registration."""
    return "[" + ",".join(f"{v:.7g}" for v in embedding) + "]"
