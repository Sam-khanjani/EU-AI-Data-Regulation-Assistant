"""pdf ingestion: page numbers, and doc_format without xhtml

Revision ID: b1f4a7c92e30
Revises: 8c397f77a5cd
Create Date: 2026-09-07

Two changes, both consequences of the consolidated act now being ingested from PDF rather
than Formex.

``structural_unit.page``
    The PDF parser reads each unit's starting page from the publisher's own bookmark outline.
    Nullable, and permanently so: Formex has no concept of a page, so the recitals source
    will always leave it NULL. A NULL here means "this unit came from a source with no
    pagination", not "we failed to work it out".

``doc_format``
    ``xhtml`` was never written and never read. The constraint is *created* here rather than
    altered, because ``native_enum=False`` renders a bare VARCHAR -- ``Enum.create_constraint``
    has defaulted to False since SQLAlchemy 1.4, so the baseline migration never created one
    despite the model comment claiming it had. Dropping the dead value is not tidying: the
    format column selects which parser runs, and an unenforced value in it is one some future
    code path will eventually set.
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = 'b1f4a7c92e30'
down_revision: str | None = '8c397f77a5cd'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column('structural_unit', sa.Column('page', sa.Integer(), nullable=True))

    # The baseline never created this constraint, but a hand-applied schema might have.
    # IF EXISTS keeps the migration idempotent across both, and across downgrade/upgrade.
    op.execute('ALTER TABLE document_version DROP CONSTRAINT IF EXISTS doc_format')
    op.create_check_constraint(
        'doc_format',
        'document_version',
        "format IN ('formex', 'pdf')",
    )


def downgrade() -> None:
    # Not re-adding 'xhtml': the prior state had no constraint at all, so dropping it is the
    # exact inverse.
    op.drop_constraint('doc_format', 'document_version', type_='check')
    op.drop_column('structural_unit', 'page')
