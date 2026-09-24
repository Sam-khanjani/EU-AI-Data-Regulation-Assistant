"""SQLAlchemy models.

Design note: document versions are *immutable snapshots*. Re-ingesting a source never
mutates an existing ``document_version`` row -- it inserts a new one and flips the previous
one to ``superseded``. Retrieval only ever reads ``status='active'``, so a partial or failed
ingestion can never be queried. This is what makes "which version produced this answer?"
answerable after the fact.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from pgvector.sqlalchemy import HALFVEC
from sqlalchemy import (
    BigInteger,
    Boolean,
    Computed,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TSVECTOR
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from euaia.config import settings

# Stored as VARCHAR rather than native PG enums: far less migration pain when a new value is
# added. Note that `native_enum=False` alone enforces *nothing* -- SQLAlchemy's
# `create_constraint` has defaulted to False since 1.4, and `validate_strings` is off too, so
# these render as bare VARCHAR and accept any string at either layer. `DocFormat` opts back
# into the CHECK because a format with no reader in the tree must not be storable; the others
# remain documentation. Which *reader* runs is `SourceSpec.parser`, not this.
SourceType = Enum("eurlex", "ec_page", name="source_type", native_enum=False)
DocFormat = Enum("pdf", name="doc_format", native_enum=False, create_constraint=True)
VersionStatus = Enum(
    "discovered", "ingesting", "active", "superseded", "failed",
    name="version_status", native_enum=False,
)
UnitType = Enum(
    "recital", "chapter", "section", "article", "paragraph", "point", "annex",
    name="unit_type", native_enum=False,
)
# How much weight a source's text carries. Ordered, and CHECK-constrained for the same reason
# `DocFormat` is: an unrecognised tier would silently rank as though it were the lowest, and
# answering "what is required?" from a voluntary code is exactly the failure to prevent.
# See `euaia.ingest.ec_documents.Authority`.
Authority = Enum(
    "law", "guidance", "code", name="authority", native_enum=False, create_constraint=True
)
CheckOutcome = Enum(
    "unchanged", "new_version", "error", name="check_outcome", native_enum=False
)
RunStatus = Enum("running", "succeeded", "failed", name="run_status", native_enum=False)
Verdict = Enum("answered", "partial", "abstained", name="verdict", native_enum=False)


class Base(DeclarativeBase):
    pass


def _now_col() -> Mapped[dt.datetime]:
    return mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class Source(Base):
    """An authoritative document we track over time."""

    __tablename__ = "source"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    key: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    publisher: Mapped[str] = mapped_column(Text, nullable=False)
    source_type: Mapped[str] = mapped_column(SourceType, nullable=False)
    # Defaulted to the strictest tier: a source added without stating its authority is
    # treated as binding, which is wrong loudly rather than wrong quietly.
    authority: Mapped[str] = mapped_column(Authority, nullable=False, server_default="law")

    # EUR-Lex sources: the stable identifiers we resolve versions through.
    eli_uri: Mapped[str | None] = mapped_column(Text)
    celex_base: Mapped[str | None] = mapped_column(String(32))
    # Non-EUR-Lex sources (Commission pages) have only a landing page to poll.
    landing_url: Mapped[str | None] = mapped_column(Text)

    check_interval_hours: Mapped[int] = mapped_column(Integer, default=24, nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[dt.datetime] = _now_col()

    versions: Mapped[list[DocumentVersion]] = relationship(
        back_populates="source", cascade="all, delete-orphan"
    )


class DocumentVersion(Base):
    """An immutable snapshot of one source at one point in time."""

    __tablename__ = "document_version"
    __table_args__ = (
        UniqueConstraint("source_id", "celex", name="uq_document_version_source_celex"),
        # At most one active version per source, enforced by the database rather than
        # by convention -- a half-finished ingestion must never become queryable.
        Index(
            "ix_document_version_one_active",
            "source_id",
            unique=True,
            postgresql_where=text("status = 'active'"),
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_id: Mapped[int] = mapped_column(
        ForeignKey("source.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Human-readable, e.g. "consolidated 2026-07-27".
    version_label: Mapped[str] = mapped_column(Text, nullable=False)
    celex: Mapped[str | None] = mapped_column(String(64))
    cellar_id: Mapped[str | None] = mapped_column(String(128))
    doc_date: Mapped[dt.date | None] = mapped_column()

    retrieved_at: Mapped[dt.datetime] = _now_col()
    content_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    raw_path: Mapped[str | None] = mapped_column(Text)
    format: Mapped[str] = mapped_column(DocFormat, nullable=False)

    status: Mapped[str] = mapped_column(VersionStatus, nullable=False, default="discovered")
    superseded_by: Mapped[int | None] = mapped_column(
        ForeignKey("document_version.id", ondelete="SET NULL")
    )
    ingested_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))

    source: Mapped[Source] = relationship(back_populates="versions")
    units: Mapped[list[StructuralUnit]] = relationship(
        back_populates="document_version", cascade="all, delete-orphan"
    )


class StructuralUnit(Base):
    """A node of the parsed legal tree -- the thing a citation points at."""

    __tablename__ = "structural_unit"
    __table_args__ = (
        Index("ix_structural_unit_lookup", "document_version_id", "unit_type", "unit_number"),
        Index("ix_structural_unit_path", "document_version_id", "unit_path"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    document_version_id: Mapped[int] = mapped_column(
        ForeignKey("document_version.id", ondelete="CASCADE"), nullable=False, index=True
    )
    parent_id: Mapped[int | None] = mapped_column(
        ForeignKey("structural_unit.id", ondelete="CASCADE"), index=True
    )

    unit_type: Mapped[str] = mapped_column(UnitType, nullable=False)
    # Official numbering exactly as published: "6", "III", "5(b)", "S1 Measure 1.1".
    unit_number: Mapped[str | None] = mapped_column(String(128))
    # Machine path for stable addressing: "CH_III/ART_6/PAR_2".
    unit_path: Mapped[str] = mapped_column(String(255), nullable=False)
    heading: Mapped[str | None] = mapped_column(Text)
    # Where the unit sits and what it is, where numbering cannot say (the codes of practice):
    # "Commitment 1: Marking ... › Measure 1.1: ... | implements Article 50(2) | optional".
    context: Mapped[str | None] = mapped_column(Text)

    text: Mapped[str] = mapped_column(Text, nullable=False)
    # NFKC-folded copy used for quote verification; see verify/normalize.py.
    text_normalized: Mapped[str] = mapped_column(Text, nullable=False)

    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    page: Mapped[int | None] = mapped_column(Integer)
    """1-based page in the source PDF. Nullable because a unit whose heading could not be
    located still gets a row; NULL means "not established", not "no pages exist"."""
    eurlex_deeplink: Mapped[str | None] = mapped_column(Text)

    document_version: Mapped[DocumentVersion] = relationship(back_populates="units")
    parent: Mapped[StructuralUnit | None] = relationship(remote_side=[id])
    chunks: Mapped[list[Chunk]] = relationship(
        back_populates="structural_unit", cascade="all, delete-orphan"
    )


class Chunk(Base):
    """An embedded retrieval unit (paragraph granularity)."""

    __tablename__ = "chunk"
    __table_args__ = (
        Index(
            "ix_chunk_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "halfvec_cosine_ops"},
        ),
        Index("ix_chunk_fts", "fts", postgresql_using="gin"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    structural_unit_id: Mapped[int] = mapped_column(
        ForeignKey("structural_unit.id", ondelete="CASCADE"), nullable=False, index=True
    )
    document_version_id: Mapped[int] = mapped_column(
        ForeignKey("document_version.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # What was embedded: breadcrumb header + unit text.
    text: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)

    # 3072 dims exceeds pgvector's 2000-dim ceiling for the `vector` type's HNSW index;
    # halfvec indexes to 4000 dims at half the storage.
    embedding: Mapped[Any] = mapped_column(HALFVEC(settings.embed_dim))
    embed_model: Mapped[str] = mapped_column(String(64), nullable=False)

    fts: Mapped[str] = mapped_column(
        TSVECTOR, Computed("to_tsvector('english', text)", persisted=True)
    )
    created_at: Mapped[dt.datetime] = _now_col()

    structural_unit: Mapped[StructuralUnit] = relationship(back_populates="chunks")


class EmbeddingCache(Base):
    """Embeddings keyed by the exact text that produced them.

    Two reasons this exists, one practical and one structural.

    **Quota.** The Gemini free tier allows 1,000 embed requests per day, counted per item.
    The corpus is 779 chunks, so a single re-ingestion very nearly exhausts a day. Without
    reuse, any mistake costs 24 hours.

    **Amendments are small.** When the AI Act is re-consolidated, a handful of articles
    change and the rest are byte-identical. Keying on content hash means a re-ingestion
    pays only for what actually changed -- which is exactly the property the "safely update
    when the regulation changes" workflow needs.

    It also makes ingestion resumable: work completed before a failure is still here on the
    next run.
    """

    __tablename__ = "embedding_cache"
    __table_args__ = (
        UniqueConstraint("text_sha256", "model", "dim", name="uq_embedding_cache_key"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    text_sha256: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    model: Mapped[str] = mapped_column(String(64), nullable=False)
    dim: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding: Mapped[Any] = mapped_column(HALFVEC(settings.embed_dim), nullable=False)
    created_at: Mapped[dt.datetime] = _now_col()


class CheckRun(Base):
    """One execution of the change detector against a source."""

    __tablename__ = "check_run"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    source_id: Mapped[int] = mapped_column(
        ForeignKey("source.id", ondelete="CASCADE"), nullable=False, index=True
    )
    ran_at: Mapped[dt.datetime] = _now_col()
    outcome: Mapped[str] = mapped_column(CheckOutcome, nullable=False)
    detail: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)


class IngestionRun(Base):
    """One ingestion job, successful or not."""

    __tablename__ = "ingestion_run"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    document_version_id: Mapped[int] = mapped_column(
        ForeignKey("document_version.id", ondelete="CASCADE"), nullable=False, index=True
    )
    started_at: Mapped[dt.datetime] = _now_col()
    finished_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(RunStatus, nullable=False, default="running")
    units_written: Mapped[int] = mapped_column(Integer, default=0)
    chunks_written: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text)
    log: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)


class QueryLog(Base):
    """Audit trail. Written on every terminal path, abstentions included."""

    __tablename__ = "query_log"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    asked_at: Mapped[dt.datetime] = _now_col()
    question: Mapped[str] = mapped_column(Text, nullable=False)
    intent: Mapped[str | None] = mapped_column(String(32))

    retrieved_chunk_ids: Mapped[list[int]] = mapped_column(ARRAY(BigInteger), default=list)
    # The provenance answer to "which version produced this?".
    document_version_ids: Mapped[list[int]] = mapped_column(ARRAY(Integer), default=list)

    answer: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    citations: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list)
    verdict: Mapped[str] = mapped_column(Verdict, nullable=False)
    abstain_reason: Mapped[str | None] = mapped_column(Text)

    citation_coverage: Mapped[float | None] = mapped_column(Float)
    quotes_total: Mapped[int] = mapped_column(Integer, default=0)
    # 0 or 1: whether a repair round-trip was made, not a count of repaired quotes.
    # Nothing is ever repaired on the model's behalf -- it is asked to quote again.
    quotes_repaired: Mapped[int] = mapped_column(Integer, default=0)
    quotes_dropped: Mapped[int] = mapped_column(Integer, default=0)

    latency_ms: Mapped[int | None] = mapped_column(Integer)
    model: Mapped[str | None] = mapped_column(String(64))
    prompt_version: Mapped[str | None] = mapped_column(String(32))
