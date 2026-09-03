"""Database engine and session factory."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker

from euaia.config import settings

# Without connect_timeout, a connection attempt to an unreachable database (e.g. Docker not
# running) hangs for a long time before failing -- the OS keeps waiting rather than giving
# up quickly, and it looks like the program is stuck rather than erroring out. 5 seconds is
# ample for a healthy local Postgres to respond and short enough that a genuinely-down
# database fails fast.
engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    future=True,
    connect_args={"connect_timeout": 5},
)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)

UNAVAILABLE_MESSAGE = (
    "Could not reach the database. Is Postgres running? "
    "Start it with: docker compose up -d db"
)


class DatabaseUnavailable(RuntimeError):
    """The database could not be reached at all -- distinct from a query that failed.

    Raised in place of the raw driver error so every entry point (CLI, web app) can show
    one clear, actionable message instead of a connection-pool stack trace.
    """


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope: commits on success, rolls back on failure."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except OperationalError as exc:
        session.rollback()
        raise DatabaseUnavailable(UNAVAILABLE_MESSAGE) from exc
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@contextmanager
def db_session() -> Iterator[Session]:
    """Yield a session without managing commit/rollback -- for callers (like FastAPI
    request handlers) that decide their own transaction boundaries. Only exists to turn a
    connection failure into DatabaseUnavailable rather than a raw driver error.
    """
    session = SessionLocal()
    try:
        yield session
    except OperationalError as exc:
        raise DatabaseUnavailable(UNAVAILABLE_MESSAGE) from exc
    finally:
        session.close()


def ensure_extensions() -> None:
    """Create the pgvector extension. Idempotent; safe to call before migrations."""
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
