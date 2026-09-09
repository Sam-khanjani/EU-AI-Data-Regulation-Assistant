"""doc_format: pdf only

Revision ID: c7d2e91b4a08
Revises: b1f4a7c92e30
Create Date: 2026-09-08

Both sources are now read from PDF, so ``formex`` is no longer a format anything can
produce. The Formex parser has been removed from the codebase entirely.

That makes any surviving ``format='formex'`` row unreproducible: nothing left in the tree
could parse its source bytes again, so it cannot be re-ingested in place and cannot be
verified against its origin. Rather than leave the constraint loose enough to keep admitting
a format with no reader, this drops those versions. Cascades remove their structural units
and chunks with them, and re-running ingestion rebuilds the recitals from the as-adopted
PDF -- which is where they come from now.
"""
from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = 'c7d2e91b4a08'
down_revision: str | None = 'b1f4a7c92e30'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Must precede the constraint change: these rows would violate it.
    op.execute("DELETE FROM document_version WHERE format = 'formex'")
    op.execute("ALTER TABLE document_version DROP CONSTRAINT IF EXISTS doc_format")
    op.create_check_constraint("doc_format", "document_version", "format IN ('pdf')")


def downgrade() -> None:
    # The deleted rows are not restored; re-ingest instead.
    op.drop_constraint("doc_format", "document_version", type_="check")
    op.create_check_constraint(
        "doc_format", "document_version", "format IN ('formex', 'pdf')"
    )
