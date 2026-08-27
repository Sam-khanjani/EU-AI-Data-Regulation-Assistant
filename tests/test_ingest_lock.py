"""Concurrent ingestion of one source must be refused, not silently interleaved.

Ingestion spans several transactions, so two overlapping runs both write the same version.
This is not theoretical: it happened, and produced 360 structural units for a 180-recital
document with every path duplicated. Retrieval then had two copies of every provision.
"""

from __future__ import annotations

import pytest
from sqlalchemy import text

from euaia.db.session import engine
from euaia.ingest.pipeline import ConcurrentIngestion, _source_lock


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


class TestSourceLock:
    def test_a_second_holder_is_refused(self):
        with _source_lock("pytest-lock-source"):
            with pytest.raises(ConcurrentIngestion, match="already ingesting"):
                with _source_lock("pytest-lock-source"):
                    pass

    def test_the_lock_is_released_afterwards(self):
        with _source_lock("pytest-lock-source"):
            pass
        # Must be re-acquirable, or one crashed run would block ingestion for good.
        with _source_lock("pytest-lock-source"):
            pass

    def test_the_lock_is_released_even_when_the_body_raises(self):
        with pytest.raises(ValueError):  # noqa: PT012
            with _source_lock("pytest-lock-source"):
                raise ValueError("ingestion blew up")
        with _source_lock("pytest-lock-source"):
            pass

    def test_different_sources_do_not_block_each_other(self):
        # The AI Act and its recitals are separate sources and may ingest in parallel.
        with _source_lock("pytest-lock-a"), _source_lock("pytest-lock-b"):
            pass

    def test_the_error_explains_why_it_matters(self):
        with _source_lock("pytest-lock-source"):
            with pytest.raises(ConcurrentIngestion) as exc:
                with _source_lock("pytest-lock-source"):
                    pass
        assert "duplicate" in str(exc.value)
