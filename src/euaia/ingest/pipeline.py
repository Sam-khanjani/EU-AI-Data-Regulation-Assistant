"""End-to-end ingestion: resolve a version, fetch it, parse it, embed it, activate it.

The invariant this module protects is that **a partially ingested version is never
queryable**. A new version is written with ``status='ingesting'`` and only flipped to
``active`` in a final transaction that simultaneously supersedes the previous one. A
partial unique index in the schema enforces at most one active version per source, so a
bug here fails loudly rather than silently serving half a corpus.

Nothing is ever overwritten. Re-ingesting produces a new ``document_version`` row, which is
what makes "which version produced this answer?" answerable months later.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import logging
import sys
import time
import zlib
from contextlib import contextmanager
from dataclasses import dataclass

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from euaia.config import settings
from euaia.db.models import Chunk, DocumentVersion, IngestionRun, Source, StructuralUnit
from euaia.db.session import SessionLocal, ensure_extensions, session_scope
from euaia.ingest import deeplinks, embedding_cache, sources
from euaia.ingest.cellar import CellarClient, Manifestation, VersionRef
from euaia.ingest.chunker import TokenCounter, chunk_document
from euaia.ingest.embedder import Embedder
from euaia.ingest.formex import ParsedDocument, parse
from euaia.ingest.sources import SourceSpec
from euaia.verify.normalize import normalize_text

log = logging.getLogger(__name__)


@dataclass(slots=True)
class IngestResult:
    source_key: str
    status: str
    version_label: str
    celex: str | None = None
    units: int = 0
    chunks: int = 0
    detail: str = ""

    def __str__(self) -> str:
        head = f"{self.source_key}: {self.status} [{self.version_label}]"
        if self.status == "ingested":
            return f"{head} - {self.units} units, {self.chunks} chunks"
        return f"{head} - {self.detail}" if self.detail else head


def ensure_source(session: Session, spec: SourceSpec) -> Source:
    """Insert the source row on first use; keep its descriptive fields current."""
    source = session.scalar(select(Source).where(Source.key == spec.key))
    if source is None:
        source = Source(key=spec.key)
        session.add(source)
    source.title = spec.title
    source.publisher = spec.publisher
    source.source_type = spec.source_type
    source.eli_uri = spec.eli_uri
    source.celex_base = spec.celex_base
    source.landing_url = spec.landing_url
    source.check_interval_hours = spec.check_interval_hours
    source.active = True
    session.flush()
    return source


def resolve_target(client: CellarClient, spec: SourceSpec) -> VersionRef:
    """Which version of this source should be live?"""
    if spec.use_consolidated:
        return client.latest_consolidated(spec.celex_base or "")
    return client.resolve_base_work(spec.celex_base or "")


def fetch_content(client: CellarClient, spec: SourceSpec, ref: VersionRef) -> Manifestation:
    """Fetch, caching raw bytes so re-runs and tests do not re-hit CELLAR."""
    settings.raw_data_dir.mkdir(parents=True, exist_ok=True)
    tag = ("consolidated_" if spec.use_consolidated else "original_") + ref.celex
    path = settings.raw_data_dir / f"{tag}.xml"

    if path.exists():
        content = path.read_bytes()
        log.info("Using cached %s (%d bytes)", path.name, len(content))
        return Manifestation(
            content=content,
            fmt="formex",
            sha256=hashlib.sha256(content).hexdigest(),
            filename=path.name,
        )

    manifestation = client.fetch(ref.cellar_id)
    path.write_bytes(manifestation.content)
    log.info(
        "Fetched %s as %s (%d bytes)", ref.celex, manifestation.fmt, len(manifestation.content)
    )
    return manifestation


class ConcurrentIngestion(RuntimeError):
    """Another process is already ingesting this source."""


@contextmanager
def _source_lock(source_key: str):
    """Hold a Postgres advisory lock for the duration of one source's ingestion.

    Ingestion spans several transactions -- create the version, write units, embed, then
    activate -- so two concurrent runs interleave and both write the same version. That is
    not hypothetical: running a full ingest and a single-source ingest at the same time
    produced exactly 360 structural units for a 180-recital document, every path duplicated.

    A session-level advisory lock makes the second run fail fast with a clear message
    instead of silently doubling the corpus. ``try`` rather than blocking, because waiting
    behind a run that is itself waiting out an embedding quota window helps nobody.
    """
    key = zlib.crc32(source_key.encode("utf-8"))
    session = SessionLocal()
    try:
        acquired = session.execute(
            text("SELECT pg_try_advisory_lock(:k)"), {"k": key}
        ).scalar()
        if not acquired:
            raise ConcurrentIngestion(
                f"another process is already ingesting {source_key!r}. "
                "Concurrent ingestion of one source would duplicate its content."
            )
        yield
    finally:
        session.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": key})
        session.commit()
        session.close()


def ingest_source(
    spec: SourceSpec,
    *,
    force: bool = False,
    skip_embeddings: bool = False,
    max_embeddings: int | None = None,
) -> IngestResult:
    """Ingest one source. Idempotent: an unchanged, already-active version is a no-op."""
    ensure_extensions()
    with _source_lock(spec.key):
        return _ingest_source_locked(
            spec,
            force=force,
            skip_embeddings=skip_embeddings,
            max_embeddings=max_embeddings,
        )


def _ingest_source_locked(
    spec: SourceSpec,
    *,
    force: bool,
    skip_embeddings: bool,
    max_embeddings: int | None,
) -> IngestResult:

    with CellarClient() as client:
        ref = resolve_target(client, spec)
        manifestation = fetch_content(client, spec, ref)

    if manifestation.fmt != "formex":
        raise NotImplementedError(
            f"{spec.key}: only Formex ingestion is implemented, got {manifestation.fmt}. "
            "An XHTML fallback parser is needed for sources CELLAR does not publish in Formex."
        )

    with session_scope() as session:
        source = ensure_source(session, spec)
        existing = session.scalar(
            select(DocumentVersion).where(
                DocumentVersion.source_id == source.id,
                DocumentVersion.celex == ref.celex,
                DocumentVersion.status == "active",
            )
        )
        if existing and existing.content_sha256 == manifestation.sha256 and not force:
            return IngestResult(
                source_key=spec.key,
                status="up-to-date",
                version_label=existing.version_label,
                celex=ref.celex,
                detail="content hash unchanged",
            )

        version = _create_version(session, source, spec, ref, manifestation)
        session.add(IngestionRun(document_version_id=version.id, status="running"))

    try:
        units, chunks = _ingest_content(
            spec, ref, manifestation,
            skip_embeddings=skip_embeddings, max_embeddings=max_embeddings,
        )
    except embedding_cache.QuotaExhausted as exc:
        # Not a bug and not worth a stack trace: the vectors computed before the stop are
        # cached, so the next run resumes rather than restarting.
        with session_scope() as session:
            _mark_failed(session, spec.key, ref.celex, str(exc))
        return IngestResult(
            source_key=spec.key,
            status="quota-exhausted",
            version_label=ref.label,
            celex=ref.celex,
            detail=str(exc),
        )
    except Exception as exc:
        with session_scope() as session:
            _mark_failed(session, spec.key, ref.celex, str(exc))
        raise

    with session_scope() as session:
        _activate(session, spec.key, ref.celex)

    return IngestResult(
        source_key=spec.key,
        status="ingested",
        version_label=ref.label,
        celex=ref.celex,
        units=units,
        chunks=chunks,
    )


def _create_version(
    session: Session,
    source: Source,
    spec: SourceSpec,
    ref: VersionRef,
    manifestation: Manifestation,
) -> DocumentVersion:
    """Insert, or reset, the pending version row for this CELEX."""
    version = session.scalar(
        select(DocumentVersion).where(
            DocumentVersion.source_id == source.id,
            DocumentVersion.celex == ref.celex,
        )
    )
    if version is None:
        version = DocumentVersion(source_id=source.id, celex=ref.celex)
        session.add(version)
    else:
        # Re-ingesting the same CELEX: drop old content so generations never mix.
        session.query(Chunk).filter(Chunk.document_version_id == version.id).delete()
        session.query(StructuralUnit).filter(
            StructuralUnit.document_version_id == version.id
        ).delete()

    version.version_label = ref.label
    version.cellar_id = ref.cellar_id
    version.doc_date = ref.doc_date
    version.retrieved_at = dt.datetime.now(dt.UTC)
    version.content_sha256 = manifestation.sha256
    version.raw_path = manifestation.filename
    version.format = manifestation.fmt
    version.status = "ingesting"
    session.flush()
    return version


def _ingest_content(
    spec: SourceSpec,
    ref: VersionRef,
    manifestation: Manifestation,
    *,
    skip_embeddings: bool,
    max_embeddings: int | None = None,
) -> tuple[int, int]:
    """Parse, persist units, chunk and embed. Returns (unit count, chunk count)."""
    doc = parse(manifestation.content)
    drafts = chunk_document(doc, spec.doc_title or spec.title, TokenCounter(), spec.unit_types)

    with session_scope() as session:
        version = _pending_version(session, spec.key, ref.celex)
        path_to_id = _write_units(session, version, doc, ref, spec.unit_types)

        if skip_embeddings:
            log.warning("skip-embeddings: %d chunks left unembedded", len(drafts))
            return len(path_to_id), 0

        embedder = Embedder()
        # Goes through the cache rather than the embedder directly: the free tier allows
        # 1,000 items a day and the corpus is 779, so re-embedding unchanged text would
        # make routine re-ingestion impossible.
        vectors, embed_stats = embedding_cache.embed_documents(
            [d.text for d in drafts], embedder, max_new=max_embeddings
        )
        log.info("Embeddings: %s", embed_stats)

        written = 0
        for draft, vector in zip(drafts, vectors, strict=True):
            unit_id = path_to_id.get(draft.unit_path)
            if unit_id is None:
                log.warning("Chunk references unknown unit path %s", draft.unit_path)
                continue
            session.add(
                Chunk(
                    structural_unit_id=unit_id,
                    document_version_id=version.id,
                    text=draft.text,
                    token_count=draft.token_count,
                    embedding=vector,
                    embed_model=embedder.model,
                )
            )
            written += 1
        session.flush()
        return len(path_to_id), written


def _selected_units(doc: ParsedDocument, unit_types: frozenset[str] | None) -> list:
    """Units this source is authoritative for, plus the ancestors needed to place them.

    Restricting this matters for correctness, not storage. The recitals source parses the
    *original* act, which still contains an unamended Article 6. Persisting it would leave
    two Article 6 rows in the database -- one current, one superseded -- and the direct
    article lookup in retrieval would happily return the stale one. Only the unit types a
    source is authoritative for are written.
    """
    if unit_types is None:
        return list(doc.units)

    by_path = {u.unit_path: u for u in doc.units}
    keep: set[str] = set()
    for unit in doc.units:
        if unit.unit_type not in unit_types:
            continue
        keep.add(unit.unit_path)
        path = unit.parent_path
        while path and path in by_path and path not in keep:
            keep.add(path)
            path = by_path[path].parent_path
    return [u for u in doc.units if u.unit_path in keep]


def _write_units(
    session: Session,
    version: DocumentVersion,
    doc: ParsedDocument,
    ref: VersionRef,
    unit_types: frozenset[str] | None = None,
) -> dict[str, int]:
    """Persist the structural tree, resolving parent links by path.

    Written in two passes rather than one. Flushing per row to learn each parent's id
    costs a database round trip per unit -- about 1,500 for the AI Act, which dominated
    ingestion time. Instead every row is inserted parentless in a single flush, then
    parent ids are filled in from the resulting path map. Both passes are inside the
    caller's transaction, so a failure between them rolls back cleanly.
    """
    selected = _selected_units(doc, unit_types)
    rows: list[StructuralUnit] = []
    for unit in selected:
        rows.append(
            StructuralUnit(
                document_version_id=version.id,
                unit_type=unit.unit_type,
                unit_number=unit.unit_number,
                unit_path=unit.unit_path,
                heading=unit.heading,
                text=unit.text,
                text_normalized=normalize_text(unit.text),
                ordinal=unit.ordinal,
                eurlex_deeplink=deeplinks.build(
                    ref.celex, unit.unit_type, unit.unit_number, unit.unit_path
                ),
            )
        )

    session.add_all(rows)
    session.flush()

    path_to_id = {row.unit_path: row.id for row in rows}
    for unit, row in zip(selected, rows, strict=True):
        if unit.parent_path:
            row.parent_id = path_to_id.get(unit.parent_path)
    session.flush()
    return path_to_id


def _pending_version(session: Session, source_key: str, celex: str) -> DocumentVersion:
    version = session.scalar(
        select(DocumentVersion)
        .join(Source)
        .where(Source.key == source_key, DocumentVersion.celex == celex)
    )
    if version is None:
        raise RuntimeError(f"no pending version for {source_key} {celex}")
    return version


def _activate(session: Session, source_key: str, celex: str) -> None:
    """Flip the new version live and supersede the old one, atomically.

    Order matters: the previous version must leave 'active' before the new one enters it,
    or the partial unique index rejects the write.
    """
    incoming = _pending_version(session, source_key, celex)
    previous = session.scalars(
        select(DocumentVersion).where(
            DocumentVersion.source_id == incoming.source_id,
            DocumentVersion.status == "active",
            DocumentVersion.id != incoming.id,
        )
    ).all()

    for old in previous:
        old.status = "superseded"
        old.superseded_by = incoming.id
    session.flush()

    incoming.status = "active"
    incoming.ingested_at = dt.datetime.now(dt.UTC)

    run = _latest_run(session, incoming.id)
    if run is not None:
        run.status = "succeeded"
        run.finished_at = dt.datetime.now(dt.UTC)
        run.units_written = (
            session.query(StructuralUnit)
            .filter(StructuralUnit.document_version_id == incoming.id)
            .count()
        )
        run.chunks_written = (
            session.query(Chunk).filter(Chunk.document_version_id == incoming.id).count()
        )


def _latest_run(session: Session, version_id: int) -> IngestionRun | None:
    return session.scalars(
        select(IngestionRun)
        .where(IngestionRun.document_version_id == version_id)
        .order_by(IngestionRun.id.desc())
    ).first()


def _mark_failed(session: Session, source_key: str, celex: str, error: str) -> None:
    try:
        version = _pending_version(session, source_key, celex)
    except RuntimeError:
        return
    version.status = "failed"
    run = _latest_run(session, version.id)
    if run is not None:
        run.status = "failed"
        run.finished_at = dt.datetime.now(dt.UTC)
        run.error = error[:4000]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest an official document into the corpus.")
    parser.add_argument(
        "--source", action="append", help="source key (repeatable); default: all registered"
    )
    parser.add_argument("--force", action="store_true", help="re-ingest even if unchanged")
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="parse and store structure without calling the embedding API",
    )
    parser.add_argument(
        "--max-embeddings",
        type=int,
        default=None,
        help=(
            "cap how many NEW embeddings a run may compute. The Gemini free tier allows "
            "1000 items/day and the full corpus is 779, so this bounds a run's share."
        ),
    )
    parser.add_argument(
        "--estimate",
        action="store_true",
        help="report how many embeddings would be needed, without calling the API",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="check CELLAR for a newer version of each source, without ingesting",
    )
    parser.add_argument(
        "--resume",
        type=int,
        metavar="N",
        default=0,
        help=(
            "on hitting the embedding quota, wait and retry up to N times. Safe to loop "
            "because embeddings are cached: each attempt pays only for the remainder, so a "
            "corpus larger than one quota window still completes."
        ),
    )
    parser.add_argument(
        "--resume-wait",
        type=int,
        default=90,
        metavar="SECONDS",
        help="seconds to wait between resume attempts (default 90)",
    )
    parser.add_argument("--list", action="store_true", help="list registered sources and exit")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-7s %(name)s: %(message)s",
    )

    if args.list:
        for spec in sources.ALL_SOURCES:
            print(f"{spec.key:24s} {spec.title}")
            print(f"{'':24s} {spec.notes}")
        return 0

    if args.estimate:
        return _estimate(args.source or [s.key for s in sources.ALL_SOURCES])

    if args.check:
        from euaia.ingest.check import check_all  # deferred: avoids a circular import

        specs = [sources.get(k) for k in (args.source or [s.key for s in sources.ALL_SOURCES])]
        with session_scope() as session:
            for result in check_all(session, specs):
                flag = " <-- newer version available" if result.outcome == "new_version" else ""
                print(f"{result.source_key:24s} {result.outcome}{flag}")
                if result.detail:
                    print(f"{'':24s} {result.detail}")
        return 0

    failures = 0
    for key in args.source or [s.key for s in sources.ALL_SOURCES]:
        spec = sources.get(key)
        try:
            result = _ingest_with_resume(
                spec,
                force=args.force,
                skip_embeddings=args.skip_embeddings,
                max_embeddings=args.max_embeddings,
                attempts=args.resume,
                wait=args.resume_wait,
            )
            print(result)
            if result.status == "quota-exhausted":
                failures += 1
        except Exception as exc:  # noqa: BLE001
            failures += 1
            log.exception("Ingestion failed for %s", key)
            print(f"{key}: FAILED - {exc}", file=sys.stderr)
    return 1 if failures else 0


def _ingest_with_resume(
    spec: SourceSpec,
    *,
    force: bool,
    skip_embeddings: bool,
    max_embeddings: int | None,
    attempts: int,
    wait: int,
) -> IngestResult:
    """Ingest, optionally waiting out the embedding quota and picking up where it stopped.

    Only safe to loop like this because embeddings are cached: each attempt re-reads what
    the previous one computed and pays only for the remainder, so retrying costs time
    rather than quota.
    """
    result = ingest_source(
        spec, force=force, skip_embeddings=skip_embeddings, max_embeddings=max_embeddings
    )
    for attempt in range(attempts):
        if result.status != "quota-exhausted":
            return result
        log.info(
            "Quota reached; waiting %ds before resume attempt %d of %d",
            wait, attempt + 1, attempts,
        )
        time.sleep(wait)
        result = ingest_source(
            spec,
            force=force,
            skip_embeddings=skip_embeddings,
            max_embeddings=max_embeddings,
        )
    return result


def _estimate(keys: list[str]) -> int:
    """Report embedding cost per source without spending any quota."""
    counter = TokenCounter()
    total_new = 0
    with CellarClient() as client:
        for key in keys:
            spec = sources.get(key)
            ref = resolve_target(client, spec)
            manifestation = fetch_content(client, spec, ref)
            doc = parse(manifestation.content)
            drafts = chunk_document(
                doc, spec.doc_title or spec.title, counter, spec.unit_types
            )
            estimate = embedding_cache.plan([d.text for d in drafts])
            total_new += estimate.to_embed
            print(f"{spec.key:24s} {estimate}")

    print()
    print(f"{'TOTAL new embeddings':24s} {total_new}")
    print(f"{'Gemini free tier/day':24s} 1000 (shared with query embeddings)")
    if total_new > 1000:
        print(
            "\nThis exceeds one day's allowance. Use --max-embeddings to split it across "
            "days;\ncached vectors are reused, so each run resumes where the last stopped."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
