"""Change detection: has EUR-Lex published a newer version of a tracked source?

Deliberately separate from ingestion. A check is one SPARQL query; ingestion is a full
fetch/parse/chunk/embed pipeline that spends real quota. An administrator (or, later, a
scheduler) needs to ask "has this changed?" on a cheap, frequent cadence without paying for
a re-ingest just to find out the answer is no.

The signal is the resolved CELEX, not a content hash: a consolidated act that changes gets a
new CELEX of its own (``CellarClient`` docstring), so comparing it to the active version's
CELEX is exactly the check-vs-ingest split -- fetching the document body is not required to
know whether one exists.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from euaia.db.models import CheckRun, DocumentVersion
from euaia.ingest.cellar import CellarClient, CellarError
from euaia.ingest.pipeline import ensure_source, resolve_target
from euaia.ingest.sources import SourceSpec

log = logging.getLogger(__name__)


@dataclass(slots=True)
class CheckResult:
    source_key: str
    outcome: str
    """'unchanged' | 'new_version' | 'error'."""

    active_celex: str | None
    latest_celex: str | None
    detail: str = ""


def check_source(session: Session, spec: SourceSpec, client: CellarClient) -> CheckResult:
    """Resolve the version CELLAR currently serves and compare it to what's active.

    Writes one ``check_run`` row regardless of outcome -- a failed check (CELLAR down, a
    SPARQL timeout) is itself a fact worth recording, not just logging, since a string of
    failed checks is what should eventually page someone rather than a single one.
    """
    source = ensure_source(session, spec)
    active = session.scalar(
        select(DocumentVersion).where(
            DocumentVersion.source_id == source.id,
            DocumentVersion.status == "active",
        )
    )
    active_celex = active.celex if active else None

    try:
        ref = resolve_target(client, spec)
    except CellarError as exc:
        result = CheckResult(
            source_key=spec.key,
            outcome="error",
            active_celex=active_celex,
            latest_celex=None,
            detail=str(exc),
        )
    else:
        outcome = "unchanged" if ref.celex == active_celex else "new_version"
        result = CheckResult(
            source_key=spec.key,
            outcome=outcome,
            active_celex=active_celex,
            latest_celex=ref.celex,
            detail=f"latest is {ref.label}" if ref.doc_date else "",
        )

    session.add(
        CheckRun(
            source_id=source.id,
            outcome=result.outcome,
            detail={
                "active_celex": result.active_celex,
                "latest_celex": result.latest_celex,
                "note": result.detail,
            },
        )
    )
    session.flush()
    return result


def check_all(session: Session, specs: list[SourceSpec]) -> list[CheckResult]:
    """Check every given source against one shared CELLAR client and connection."""
    with CellarClient() as client:
        return [check_source(session, spec, client) for spec in specs]
