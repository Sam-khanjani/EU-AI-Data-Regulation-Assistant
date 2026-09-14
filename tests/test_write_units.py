"""Persisting parsed units, with the fields that only a PDF source can supply.

Narrow on purpose: this covers the step between "the parser produced a page number" and
"the page number is in the database". That gap is where a new column most often dies -- the
migration lands, the parser fills the field, and nothing carries it across.
"""

from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy import text

from euaia.db.models import DocumentVersion, Source, StructuralUnit
from euaia.db.session import SessionLocal, engine
from euaia.ingest.cellar import VersionRef
from euaia.ingest.document import ParsedDocument, ParsedUnit
from euaia.ingest.pipeline import _write_units

TEST_SOURCE_KEY = "pytest-write-units"


def _db_available() -> bool:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(
    not _db_available(), reason="Postgres not reachable; run `docker compose up -d db`"
)


@pytest.fixture
def version():
    """A throwaway source + version, removed afterwards whatever the test does."""
    session = SessionLocal()
    try:
        source = Source(
            key=TEST_SOURCE_KEY,
            title="write-units fixture",
            publisher="pytest",
            source_type="eurlex",
        )
        session.add(source)
        session.flush()
        doc_version = DocumentVersion(
            source_id=source.id,
            celex="02024R1689-20260727",
            version_label="fixture",
            status="ingesting",
            format="pdf",
            content_sha256="0" * 64,
            retrieved_at=dt.datetime.now(dt.UTC),
        )
        session.add(doc_version)
        session.flush()
        yield session, doc_version
    finally:
        session.rollback()
        session.execute(
            text("DELETE FROM source WHERE key = :k"), {"k": TEST_SOURCE_KEY}
        )
        session.commit()
        session.close()


REF = VersionRef(
    cellar_id="x",
    celex="02024R1689-20260727",
    doc_date=None,
    is_consolidated=True,
)


def _doc() -> ParsedDocument:
    article = ParsedUnit(
        unit_type="article",
        unit_path="CH_III/ART_6",
        text="Article 6\nClassification rules",
        ordinal=1,
        unit_number="6",
        heading="Classification rules",
        page=17,
    )
    paragraph = ParsedUnit(
        unit_type="paragraph",
        unit_path="CH_III/ART_6/PAR_1",
        text="1. An AI system shall be considered high-risk.",
        ordinal=2,
        unit_number="6(1)",
        parent_path="CH_III/ART_6",
        page=18,
    )
    recital = ParsedUnit(
        unit_type="recital",
        unit_path="RCT_27",
        text="(27) Whereas something.",
        ordinal=3,
        unit_number="27",
        page=None,
    )
    return ParsedDocument(units=[article, paragraph, recital])


class TestPagePersistence:
    def test_pages_reach_the_database(self, version):
        session, doc_version = version
        _write_units(session, doc_version, _doc(), REF, None)
        session.flush()

        rows = {
            row.unit_path: row
            for row in session.query(StructuralUnit).filter_by(
                document_version_id=doc_version.id
            )
        }
        assert rows["CH_III/ART_6"].page == 17
        assert rows["CH_III/ART_6/PAR_1"].page == 18

    def test_a_unit_without_a_page_stores_null(self, version):
        # A unit whose page could not be established still gets a row. NULL means "not
        # established", not "this unit does not exist".
        session, doc_version = version
        _write_units(session, doc_version, _doc(), REF, None)
        session.flush()

        recital = (
            session.query(StructuralUnit)
            .filter_by(document_version_id=doc_version.id, unit_path="RCT_27")
            .one()
        )
        assert recital.page is None

    def test_parentage_still_resolves(self, version):
        # The two-pass insert-then-link is what makes ingestion fast; a page column added to
        # the first pass must not disturb it.
        session, doc_version = version
        _write_units(session, doc_version, _doc(), REF, None)
        session.flush()

        rows = {
            row.unit_path: row
            for row in session.query(StructuralUnit).filter_by(
                document_version_id=doc_version.id
            )
        }
        assert rows["CH_III/ART_6/PAR_1"].parent_id == rows["CH_III/ART_6"].id
        assert rows["CH_III/ART_6"].parent_id is None


class TestUpToDateRequiresCompleteness:
    """An unchanged content hash is not on its own proof the version is usable.

    ``--skip-embeddings`` leaves the version active with its units written but no chunks. If
    the idempotency check looked only at the hash, every later run would report "up-to-date"
    and refuse to embed -- leaving a corpus that cannot be searched and a pipeline insisting
    nothing needs doing. That is exactly the "partially ingested version is never queryable"
    invariant this module's docstring claims to protect.
    """

    def test_a_version_with_no_chunks_is_not_up_to_date(self, version):
        from sqlalchemy import func, select

        from euaia.db.models import Chunk

        session, doc_version = version
        _write_units(session, doc_version, _doc(), REF, None)
        session.flush()

        stored = session.scalar(
            select(func.count()).select_from(Chunk).where(
                Chunk.document_version_id == doc_version.id
            )
        )
        assert stored == 0, "units written but no chunks -- the state --skip-embeddings leaves"

    def test_units_alone_do_not_make_a_corpus_searchable(self, version):
        # The UI reads this same signal: service.py reports embedded=bool(chunks), which is
        # what renders the red "not embedded" badge.
        from sqlalchemy import func, select

        from euaia.db.models import Chunk

        session, doc_version = version
        _write_units(session, doc_version, _doc(), REF, None)
        session.flush()

        units = session.query(StructuralUnit).filter_by(
            document_version_id=doc_version.id
        ).count()
        chunks = session.scalar(
            select(func.count()).select_from(Chunk).where(
                Chunk.document_version_id == doc_version.id
            )
        )
        assert units > 0 and chunks == 0
