"""Change detection compares what CELLAR resolves now against what is active.

The CELLAR client is faked throughout -- these tests are about the comparison and the
``check_run`` bookkeeping, not about SPARQL. ``test_pipeline_live.py`` and the ``cellar.py``
docstring cover the real endpoint.
"""

from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy import select
from sqlalchemy import text as sql_text

from euaia.db.models import CheckRun, DocumentVersion, Source
from euaia.db.session import engine, session_scope
from euaia.ingest.cellar import CellarError, VersionRef
from euaia.ingest.check import check_all, check_source
from euaia.ingest.sources import SourceSpec

TEST_KEY = "pytest-check-source"
TEST_KEY_2 = "pytest-check-source-2"


def _db_available() -> bool:
    try:
        with engine.connect() as conn:
            conn.execute(sql_text("SELECT 1"))
        return True
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(
    not _db_available(), reason="Postgres not reachable; run `docker compose up -d db`"
)


def _spec(key: str = TEST_KEY) -> SourceSpec:
    return SourceSpec(
        key=key,
        title="Test source",
        publisher="pytest",
        source_type="eurlex",
        celex_base="99999X9999",
        use_consolidated=True,
    )


def _ref(celex: str) -> VersionRef:
    return VersionRef(
        cellar_id="test-cellar-id",
        celex=celex,
        doc_date=dt.date(2026, 1, 1),
        is_consolidated=True,
    )


class FakeClient:
    """Stands in for CellarClient: returns a fixed VersionRef, or raises."""

    def __init__(self, ref: VersionRef | None = None, error: Exception | None = None):
        self._ref = ref
        self._error = error
        self.calls = 0

    def latest_consolidated(self, base_celex: str) -> VersionRef:
        self.calls += 1
        if self._error:
            raise self._error
        assert self._ref is not None
        return self._ref


@pytest.fixture(autouse=True)
def _purge():
    with session_scope() as session:
        session.execute(
            sql_text("DELETE FROM source WHERE key IN (:a, :b)"),
            {"a": TEST_KEY, "b": TEST_KEY_2},
        )
    yield
    with session_scope() as session:
        session.execute(
            sql_text("DELETE FROM source WHERE key IN (:a, :b)"),
            {"a": TEST_KEY, "b": TEST_KEY_2},
        )


def _add_active_version(session, key: str, celex: str) -> None:
    source = Source(
        key=key, title="Test source", publisher="pytest",
        source_type="eurlex", celex_base="99999X9999",
    )
    session.add(source)
    session.flush()
    session.add(
        DocumentVersion(
            source_id=source.id,
            version_label=f"consolidated {celex}",
            celex=celex,
            content_sha256="a" * 64,
            format="pdf",
            status="active",
        )
    )


class TestNoPriorVersion:
    def test_first_check_is_a_new_version(self):
        with session_scope() as session:
            result = check_source(session, _spec(), FakeClient(ref=_ref("TEST-1")))
        assert result.outcome == "new_version"
        assert result.active_celex is None
        assert result.latest_celex == "TEST-1"

    def test_it_creates_the_source_row(self):
        with session_scope() as session:
            check_source(session, _spec(), FakeClient(ref=_ref("TEST-1")))
            assert session.scalar(select(Source).where(Source.key == TEST_KEY)) is not None


class TestMatchingVersion:
    def test_same_celex_is_unchanged(self):
        with session_scope() as session:
            _add_active_version(session, TEST_KEY, "TEST-1")
        with session_scope() as session:
            result = check_source(session, _spec(), FakeClient(ref=_ref("TEST-1")))
        assert result.outcome == "unchanged"
        assert result.active_celex == "TEST-1"
        assert result.latest_celex == "TEST-1"


class TestNewerVersion:
    def test_different_celex_is_flagged(self):
        with session_scope() as session:
            _add_active_version(session, TEST_KEY, "TEST-1")
        with session_scope() as session:
            result = check_source(session, _spec(), FakeClient(ref=_ref("TEST-2")))
        assert result.outcome == "new_version"
        assert result.active_celex == "TEST-1"
        assert result.latest_celex == "TEST-2"


class TestCellarFailure:
    def test_an_error_is_recorded_not_raised(self):
        # A SPARQL timeout must not crash the check loop or leave silence where a record
        # of the failure belongs.
        with session_scope() as session:
            result = check_source(
                session, _spec(), FakeClient(error=CellarError("SPARQL timed out"))
            )
        assert result.outcome == "error"
        assert result.latest_celex is None
        assert "timed out" in result.detail


class TestCheckRunBookkeeping:
    def test_every_check_writes_a_check_run_row(self):
        with session_scope() as session:
            check_source(session, _spec(), FakeClient(ref=_ref("TEST-1")))
            check_source(session, _spec(), FakeClient(ref=_ref("TEST-1")))
            source = session.scalar(select(Source).where(Source.key == TEST_KEY))
            count = (
                session.query(CheckRun).filter(CheckRun.source_id == source.id).count()
            )
        assert count == 2

    def test_failed_checks_are_also_recorded(self):
        with session_scope() as session:
            check_source(session, _spec(), FakeClient(error=CellarError("down")))
            source = session.scalar(select(Source).where(Source.key == TEST_KEY))
            run = session.scalar(
                select(CheckRun).where(CheckRun.source_id == source.id)
            )
        assert run.outcome == "error"


def test_check_all_shares_one_client(monkeypatch):
    fake = FakeClient(ref=_ref("TEST-1"))

    class _FakeCtx:
        def __enter__(self):
            return fake

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr("euaia.ingest.check.CellarClient", lambda: _FakeCtx())
    with session_scope() as session:
        results = check_all(session, [_spec(TEST_KEY), _spec(TEST_KEY_2)])
    assert len(results) == 2
    assert fake.calls == 2
